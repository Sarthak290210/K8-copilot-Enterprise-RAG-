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


def test_successful_upload_replaces_old_version(monkeypatch):
    calls = []

    class FakeProcessor:
        def process_document(self, path, source):
            assert path.name == source  # saved under its base name only
            return [{"text": "Pods are small.", "source": source, "metadata": {"chunk_index": 0}}]

    async def fake_embed(texts):
        return [[0.1] for _ in texts]

    async def fake_delete(source):
        calls.append(("delete", source))

    async def fake_upsert(chunks, embeddings):
        calls.append(("upsert", len(chunks)))
        return len(chunks)

    monkeypatch.setattr(documents, "get_processor", lambda: FakeProcessor())
    monkeypatch.setattr(documents.embedding_service, "embed_texts", fake_embed)
    monkeypatch.setattr(documents.vector_store, "delete_source", fake_delete)
    monkeypatch.setattr(documents.vector_store, "upsert_chunks", fake_upsert)

    # A path in the client-supplied name must not survive.
    r = upload("../../etc/pods.txt", b"Pods are small.")
    assert r.status_code == 201
    assert r.json() == {"source": "pods.txt", "chunks_indexed": 1}
    assert calls == [("delete", "pods.txt"), ("upsert", 1)]


def test_list_documents(monkeypatch):
    async def fake_list():
        return ["a.pdf", "pods.html"]

    monkeypatch.setattr(documents.vector_store, "list_sources", fake_list)
    r = client.get("/documents", headers=auth(False))
    assert r.status_code == 200 and r.json() == {"sources": ["a.pdf", "pods.html"]}
