"""vector_store against an in-memory Qdrant (no server needed)."""

import pytest
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.services import vector_store


def chunk(source, index, text):
    return {"text": text, "source": source, "metadata": {"chunk_index": index}}


def vec(*values):
    """A tiny vector padded to the configured dimension."""
    return list(values) + [0.0] * (settings.embedding_dim - len(values))


@pytest.fixture
async def client():
    qdrant = AsyncQdrantClient(location=":memory:")
    await vector_store.ensure_collection(qdrant)
    yield qdrant
    await qdrant.close()


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


async def test_mismatched_lengths_rejected(client):
    with pytest.raises(ValueError, match="1 chunks but 0 embeddings"):
        await vector_store.upsert_chunks([chunk("a", 0, "a")], [], client)
