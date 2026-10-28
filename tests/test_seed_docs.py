import pytest

from app.services import document_processor, embedding_service, vector_store
from scripts import seed_db


def test_corpus_selection_is_reproducible():
    signal_a, noise_a = seed_db._select_corpus(10)
    signal_b, noise_b = seed_db._select_corpus(10)
    assert noise_a == noise_b  # seeded: every run and every eval uses the same noise sample
    assert len(noise_a) == 10
    # all signal docs, always, plus the Phase 6 attack test docs
    assert signal_a == seed_db._collect_files("true_data") + seed_db._collect_files("attack_data")


def test_missing_folder_has_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_db, "DOCS_DIR", tmp_path)
    assert seed_db._collect_files("attack_data") == []


def test_all_and_oversized_samples_take_everything():
    _, everything = seed_db._select_corpus("all")
    _, oversized = seed_db._select_corpus(len(everything) + 5)
    assert everything == oversized == seed_db._collect_files("noisy_data")


def test_collect_files_skips_unsupported():
    files = seed_db._collect_files("true_data")
    assert files and all(f.suffix.lower() in {".pdf", ".docx", ".html", ".txt"} for f in files)
    assert all(f.name != ".gitkeep" for f in files)


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    """A tiny seed/docs with fake Docling, embeddings and Qdrant; records what was ingested."""
    (tmp_path / "true_data").mkdir()
    (tmp_path / "noisy_data").mkdir()
    (tmp_path / "true_data" / "pods.txt").write_bytes(b"Pods.")
    (tmp_path / "true_data" / "old.txt").write_bytes(b"Ingested before Phase 5.")
    state = {"sources": {"old.txt": 4}, "processed": [], "deleted": []}

    class FakeProcessor:
        def __init__(self, table_structure=True):
            pass

        def process_document(self, src, max_pages=None):
            state["processed"].append(src.name)
            text = src.read_text()
            return [{"text": text, "source": src.name, "metadata": {"chunk_index": 0}}]

    class FakeClient:
        async def close(self):
            pass

    def get_client():
        return FakeClient()

    get_client.cache_clear = lambda: None  # type: ignore[attr-defined]

    async def noop():
        pass

    async def list_sources():
        return list(state["sources"])

    async def count_source(source):
        return state["sources"].get(source, 0)

    async def delete_source(source):
        state["deleted"].append(source)

    async def upsert_chunks(chunks, vectors):
        state["sources"][chunks[0]["source"]] = len(chunks)
        return len(chunks)

    async def count_points():
        return sum(state["sources"].values())

    monkeypatch.setattr(seed_db, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(document_processor, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(embedding_service, "embed_texts_sync", lambda texts: [[0.1]] * len(texts))
    monkeypatch.setattr(vector_store, "get_client", get_client)
    monkeypatch.setattr(vector_store, "ensure_collection", noop)
    monkeypatch.setattr(vector_store, "list_sources", list_sources)
    monkeypatch.setattr(vector_store, "count_source", count_source)
    monkeypatch.setattr(vector_store, "delete_source", delete_source)
    monkeypatch.setattr(vector_store, "upsert_chunks", upsert_chunks)
    monkeypatch.setattr(vector_store, "count_points", count_points)
    return state


def test_reseeding_skips_files_already_ingested(fake_corpus):
    first = seed_db.seed_docs(noise_sample_size=0)
    # old.txt was in Qdrant without a hash record: recorded, not re-ingested (user's choice).
    assert fake_corpus["processed"] == ["pods.txt"]
    assert first["files"] == 1 and first["recorded"] == 1 and first["skipped"] == 0
    second = seed_db.seed_docs(noise_sample_size=0)
    assert fake_corpus["processed"] == ["pods.txt"]  # nothing processed again
    assert second["skipped"] == 2 and second["files"] == 0


def test_changed_file_is_reingested_replacing_its_chunks(fake_corpus):
    seed_db.seed_docs(noise_sample_size=0)
    (seed_db.DOCS_DIR / "true_data" / "pods.txt").write_bytes(b"Pods, updated.")
    counters = seed_db.seed_docs(noise_sample_size=0)
    assert fake_corpus["processed"] == ["pods.txt", "pods.txt"]
    assert fake_corpus["deleted"] == ["pods.txt"]  # old chunks removed first
    assert counters["files"] == 1 and counters["skipped"] == 1


def test_recorded_files_are_reingested_when_qdrant_lost_them(fake_corpus):
    """`docker compose down -v` wipes Qdrant but keeps ./cache: records alone mustn't skip."""
    seed_db.seed_docs(noise_sample_size=0)
    fake_corpus["sources"].clear()
    counters = seed_db.seed_docs(noise_sample_size=0)
    assert counters["skipped"] == 0 and counters["files"] == 2
    assert fake_corpus["sources"] == {"pods.txt": 1, "old.txt": 1}
