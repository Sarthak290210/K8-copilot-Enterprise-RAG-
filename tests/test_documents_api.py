import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.main import app
from app.middleware.auth import create_access_token

client = TestClient(app)


def auth(is_admin):
    return {"Authorization": f"Bearer {create_access_token(1, 'u', is_admin)}"}


def upload(name, data, is_admin=True):
    return client.post("/documents/upload", files={"file": (name, data)}, headers=auth(is_admin))


def test_upload_requires_admin():
    assert upload("pods.txt", b"Pods.", is_admin=False).status_code == 403


def test_bad_extension_rejected():
    r = upload("payload.exe", b"MZ")
    assert r.status_code == 400 and "Unsupported file type" in r.json()["detail"]


def test_empty_file_rejected():
    assert upload("pods.txt", b"").status_code == 400


def test_too_large_rejected():
    assert upload("big.txt", b"x" * (documents.MAX_UPLOAD_BYTES + 1)).status_code == 413


class Calls(list):
    """What the upload did, in order; `indexed` = the fake Qdrant (chunks per source)."""

    indexed: dict[str, int]


@pytest.fixture
def fake_ingest(monkeypatch):
    calls = Calls()
    calls.indexed = {}

    class FakeProcessor:
        def process_document(self, path, source):
            assert path.name == source  # saved under its base name only
            calls.append(("process", source))
            return [{"text": "Pods are small.", "source": source, "metadata": {"chunk_index": 0}}]

    async def fake_embed(texts):
        return [[0.1] for _ in texts]

    async def fake_delete(source):
        calls.append(("delete", source))
        calls.indexed.pop(source, None)

    async def fake_upsert(chunks, embeddings):
        calls.append(("upsert", len(chunks)))
        for chunk in chunks:
            calls.indexed[chunk["source"]] = calls.indexed.get(chunk["source"], 0) + 1
        return len(chunks)

    async def fake_count(source):
        return calls.indexed.get(source, 0)

    monkeypatch.setattr(documents, "get_processor", lambda: FakeProcessor())
    monkeypatch.setattr(documents.embedding_service, "embed_texts", fake_embed)
    monkeypatch.setattr(documents.vector_store, "delete_source", fake_delete)
    monkeypatch.setattr(documents.vector_store, "upsert_chunks", fake_upsert)
    monkeypatch.setattr(documents.vector_store, "count_source", fake_count)
    return calls


def test_successful_upload_replaces_old_version(fake_ingest):
    # A path in the client-supplied name must not survive.
    r = upload("../../etc/pods.txt", b"Pods are small.")
    assert r.status_code == 201
    assert r.json() == {"source": "pods.txt", "chunks_indexed": 1, "already_ingested": False}
    assert fake_ingest == [("process", "pods.txt"), ("delete", "pods.txt"), ("upsert", 1)]


def test_same_bytes_are_not_ingested_twice(fake_ingest):
    assert upload("pods.txt", b"Pods are small.").status_code == 201
    fake_ingest.clear()
    r = upload("copy-of-pods.txt", b"Pods are small.")  # same content, another name
    assert r.status_code == 200
    assert r.json() == {"source": "pods.txt", "chunks_indexed": 1, "already_ingested": True}
    assert fake_ingest == []  # no Docling, no embedding, no Qdrant write


def test_recorded_but_no_longer_indexed_is_ingested_again(fake_ingest):
    """The records outlive Qdrant (e.g. `docker compose down -v` keeps ./cache)."""
    assert upload("pods.txt", b"Pods are small.").status_code == 201
    fake_ingest.indexed.clear()  # Qdrant wiped
    r = upload("pods.txt", b"Pods are small.")
    assert r.status_code == 201 and not r.json()["already_ingested"]
    assert fake_ingest.indexed == {"pods.txt": 1}


def test_changed_content_is_ingested_again(fake_ingest):
    assert upload("pods.txt", b"Pods v1.").status_code == 201
    assert upload("pods.txt", b"Pods v2.").status_code == 201
    # Back to v1: pods.txt now holds v2's chunks, so v1 must be ingested again.
    fake_ingest.clear()
    r = upload("pods.txt", b"Pods v1.")
    assert r.status_code == 201 and not r.json()["already_ingested"]
    assert ("upsert", 1) in fake_ingest


def test_upload_drops_cached_answers(fake_ingest, cache_on):
    import asyncio

    asyncio.run(cache_on.set_rag_answer("q", {"f": 1}, {"response": {"answer": "old"}}))
    assert upload("pods.txt", b"Pods are small.").status_code == 201
    assert asyncio.run(cache_on.get_rag_answer("q", {"f": 1})) is None


def test_list_documents(monkeypatch):
    async def fake_list():
        return ["a.pdf", "pods.html"]

    monkeypatch.setattr(documents.vector_store, "list_sources", fake_list)
    r = client.get("/documents", headers=auth(False))
    assert r.status_code == 200 and r.json() == {"sources": ["a.pdf", "pods.html"]}
