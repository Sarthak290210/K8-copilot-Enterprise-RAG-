"""vector_store against an in-memory Qdrant (no server needed)."""

import pytest
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.services import vector_store


def chunk(source, index, text):
    return {"text": text, "source": source, "metadata": {"chunk_index": index}}


def filler(n):
    """Unrelated chunks, so test words are rare (BM25 gives no weight to a word in half the docs)."""
    return [chunk("filler.pdf", i, f"graphics shader pipeline number {i}") for i in range(n)]


def vec(*values):
    """A tiny vector padded to the configured dimension."""
    return list(values) + [0.0] * (settings.embedding_dim - len(values))


@pytest.fixture
async def client():
    vector_store.invalidate_sparse_index()  # the BM25 cache is module-level
    qdrant = AsyncQdrantClient(location=":memory:")
    await vector_store.ensure_collection(qdrant)
    yield qdrant
    await qdrant.close()
    vector_store.invalidate_sparse_index()


async def test_ensure_collection_is_idempotent(client):
    await vector_store.ensure_collection(client)  # second call must not fail
    assert await client.collection_exists(settings.qdrant_collection)


async def test_search_returns_nearest_first(client):
    await vector_store.upsert_chunks(
        [chunk("pods.html", 0, "pods"), chunk("noise.pdf", 0, "hashing")],
        [vec(1.0, 0.0), vec(0.0, 1.0)],
        client,
    )
    hits = await vector_store.search(vec(0.9, 0.1), top_k=2, client=client)
    assert [h.source for h in hits] == ["pods.html", "noise.pdf"]
    assert hits[0].text == "pods" and hits[0].metadata == {"chunk_index": 0}
    assert hits[0].score > hits[1].score


async def test_reingesting_overwrites_instead_of_duplicating(client):
    for _ in range(2):
        await vector_store.upsert_chunks([chunk("pods.html", 0, "v1")], [vec(1.0)], client)
    assert await vector_store.count_points(client) == 1
    assert vector_store.point_id("pods.html", 0) == vector_store.point_id("pods.html", 0)
    assert vector_store.point_id("pods.html", 0) != vector_store.point_id("pods.html", 1)


async def test_delete_source(client):
    await vector_store.upsert_chunks(
        [chunk("a.pdf", 0, "a"), chunk("a.pdf", 1, "a2"), chunk("b.pdf", 0, "b")],
        [vec(1.0), vec(0.5, 0.5), vec(0.0, 1.0)],
        client,
    )
    await vector_store.delete_source("a.pdf", client)
    assert await vector_store.count_points(client) == 1


async def test_sparse_search_index_is_built_once_and_rebuilt_after_changes(client, monkeypatch):
    docs = [chunk("debug.txt", 0, "container in CrashLoopBackOff"), *filler(6)]
    await vector_store.upsert_chunks(docs, [vec(1.0)] * len(docs), client)
    builds = []
    real_fit = vector_store.SparseVectorIndex.fit

    def counting_fit(self, documents):
        builds.append(len(documents))
        real_fit(self, documents)

    monkeypatch.setattr(vector_store.SparseVectorIndex, "fit", counting_fit)
    for _ in range(3):
        hits = await vector_store.sparse_search("CrashLoopBackOff", top_k=5, client=client)
        assert [h.source for h in hits] == ["debug.txt"]
    assert builds == [7]  # one build for three queries

    await vector_store.upsert_chunks(
        [chunk("new.txt", 0, "CrashLoopBackOff again")], [vec(0.5)], client
    )
    hits = await vector_store.sparse_search("CrashLoopBackOff", top_k=5, client=client)
    assert builds == [7, 8]  # upsert invalidated the index
    assert {h.source for h in hits} == {"debug.txt", "new.txt"}

    await vector_store.delete_source("new.txt", client)
    await vector_store.sparse_search("CrashLoopBackOff", client=client)
    assert builds == [7, 8, 7]


async def test_scroll_pages_through_everything(client, monkeypatch):
    monkeypatch.setattr(vector_store, "SCROLL_PAGE_SIZE", 2)  # force several pages
    n = 5
    await vector_store.upsert_chunks(
        [chunk("a.txt", i, f"word{i}") for i in range(n)], [vec(1.0)] * n, client
    )
    payloads = await vector_store._scroll_all_payloads(client)
    assert sorted(p["metadata"]["chunk_index"] for p in payloads) == list(range(n))


async def test_hybrid_search_fuses_dense_and_sparse(client):
    docs = [
        chunk("dense-only.html", 0, "pods and containers"),
        chunk("both.txt", 0, "CrashLoopBackOff pods"),
        chunk("sparse-only.txt", 0, "CrashLoopBackOff"),
        *filler(6),
    ]
    vectors = [vec(1.0, 0.0), vec(0.9, 0.1), vec(0.0, 1.0)] + [vec(0.0, 0.0, 1.0)] * 6
    await vector_store.upsert_chunks(docs, vectors, client)
    hits = await vector_store.hybrid_search(
        "CrashLoopBackOff", vec(1.0, 0.0), top_k=3, client=client
    )
    assert hits[0].source == "both.txt"  # found by both searches -> highest RRF score
    assert {h.source for h in hits} == {"dense-only.html", "both.txt", "sparse-only.txt"}
    assert hits[0].metadata["dense_rank"] == 2 and "sparse_rank" in hits[0].metadata


async def test_mismatched_lengths_rejected(client):
    with pytest.raises(ValueError, match="1 chunks but 0 embeddings"):
        await vector_store.upsert_chunks([chunk("a", 0, "a")], [], client)
