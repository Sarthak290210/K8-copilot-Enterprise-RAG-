"""Read and write vectors in Qdrant."""

import uuid
from functools import lru_cache
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.models import RetrievedChunk

UPSERT_BATCH_SIZE = 256
# Fixed namespace: the same (source, chunk_index) always maps to the same point ID.
POINT_ID_NAMESPACE = uuid.UUID("6f1c2d5e-8b0a-4f43-9c7e-2a5d1b3e4f60")


@lru_cache
def get_client() -> AsyncQdrantClient:
    """Shared client; main.py's lifespan closes it and clears this cache."""
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=30)


def point_id(source: str, chunk_index: int) -> str:
    """Deterministic ID, so re-ingesting a file overwrites its chunks instead of duplicating."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{source}::{chunk_index}"))


async def ensure_collection(client: AsyncQdrantClient | None = None) -> None:
    """Create the collection and its `source` index if missing. Safe on every startup."""
    client = client or get_client()
    name = settings.qdrant_collection
    if not await client.collection_exists(name):
        await client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        logger.info("Created Qdrant collection {} (dim {})", name, settings.embedding_dim)
    # Keyword index: fast filtering/faceting by source (list and delete documents).
    await client.create_payload_index(
        name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
    )


async def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    client: AsyncQdrantClient | None = None,
) -> int:
    """Write chunks ({text, source, metadata}) with their vectors. Returns points written."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    client = client or get_client()
    points = [
        models.PointStruct(
            id=point_id(chunk["source"], chunk["metadata"]["chunk_index"]),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "metadata": chunk["metadata"],
            },
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        await client.upsert(
            settings.qdrant_collection, points=points[start : start + UPSERT_BATCH_SIZE], wait=True
        )
    return len(points)


async def delete_source(source: str, client: AsyncQdrantClient | None = None) -> None:
    """Remove every chunk of one document (before re-ingesting a changed file)."""
    client = client or get_client()
    await client.delete(
        settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
        wait=True,
    )


async def search(
    query_embedding: list[float], top_k: int = 5, client: AsyncQdrantClient | None = None
) -> list[RetrievedChunk]:
    """Nearest neighbours by cosine similarity, best first."""
    client = client or get_client()
    result = await client.query_points(
        settings.qdrant_collection, query=query_embedding, limit=top_k, with_payload=True
    )
    return [
        RetrievedChunk(
            text=(point.payload or {}).get("text", ""),
            source=(point.payload or {}).get("source", "unknown"),
            score=point.score,
            metadata=(point.payload or {}).get("metadata", {}),
        )
        for point in result.points
    ]


async def list_sources(client: AsyncQdrantClient | None = None, limit: int = 10_000) -> list[str]:
    """Distinct document names in the collection (via the keyword index's facet counts)."""
    client = client or get_client()
    result = await client.facet(settings.qdrant_collection, key="source", limit=limit, exact=True)
    return sorted(str(hit.value) for hit in result.hits)


async def count_points(client: AsyncQdrantClient | None = None) -> int:
    client = client or get_client()
    return (await client.count(settings.qdrant_collection, exact=True)).count
