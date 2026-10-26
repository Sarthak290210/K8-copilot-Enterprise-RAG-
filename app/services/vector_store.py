"""Read and write vectors in Qdrant, plus BM25 keyword search and hybrid (dense + BM25) search."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.models import RetrievedChunk
from app.services.sparse_vector_service import SparseVectorIndex, fuse_rrf

UPSERT_BATCH_SIZE = 256
SCROLL_PAGE_SIZE = 1_000

# BM25 index over every chunk: built on first use, reused by every query, and dropped when
# documents change in this process. (A separate process, e.g. the seed script, can't reach it:
# restart the API after a bulk ingest.)
_sparse_index: SparseVectorIndex | None = None
_sparse_lock = asyncio.Lock()
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
    invalidate_sparse_index()
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
    invalidate_sparse_index()


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


async def _scroll_all_payloads(client: AsyncQdrantClient) -> list[dict[str, Any]]:
    """Every point's payload, page by page (no cap), without the vectors."""
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            settings.qdrant_collection,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(
            {
                "text": (p.payload or {}).get("text", ""),
                "source": (p.payload or {}).get("source", "unknown"),
                "metadata": (p.payload or {}).get("metadata", {}),
            }
            for p in points
        )
        if offset is None:
            return payloads


async def _build_sparse_index(client: AsyncQdrantClient | None = None) -> SparseVectorIndex:
    """The cached BM25 index, built from all chunks on first use (one build at a time)."""
    global _sparse_index
    if _sparse_index is not None:
        return _sparse_index
    async with _sparse_lock:
        if _sparse_index is None:  # another request may have built it while we waited
            start = time.perf_counter()
            payloads = await _scroll_all_payloads(client or get_client())
            index = SparseVectorIndex()
            await asyncio.to_thread(index.fit, payloads)  # tokenising ~7k chunks is CPU work
            _sparse_index = index
            logger.info(
                "Built BM25 index over {} chunks in {:.1f}s",
                len(index),
                time.perf_counter() - start,
            )
    return _sparse_index


def invalidate_sparse_index() -> None:
    """Drop the BM25 index; the next sparse search rebuilds it with the current chunks."""
    global _sparse_index
    _sparse_index = None


async def sparse_search(
    query_text: str, top_k: int = 20, client: AsyncQdrantClient | None = None
) -> list[RetrievedChunk]:
    """BM25 keyword search over all chunks, best first."""
    index = await _build_sparse_index(client)
    return await asyncio.to_thread(index.search, query_text, top_k)


DenseRetriever = Callable[[int], Awaitable[list[RetrievedChunk]]]  # depth -> ranked chunks


async def hybrid_search(
    query_text: str,
    query_embedding: list[float] | None,
    top_k: int = 20,
    client: AsyncQdrantClient | None = None,
    dense_search: DenseRetriever | None = None,
) -> list[RetrievedChunk]:
    """Dense and BM25 candidates (each `reranker_initial_top_k` deep, run concurrently), merged
    with RRF. `dense_search` replaces the plain dense search (HyDE plugs in here); BM25 always
    uses the question itself."""
    depth = max(top_k, settings.reranker_initial_top_k)
    if dense_search is not None:
        dense_candidates = dense_search(depth)
    elif query_embedding is not None:
        dense_candidates = search(query_embedding, top_k=depth, client=client)
    else:
        raise ValueError("hybrid_search needs query_embedding or dense_search")
    dense, sparse = await asyncio.gather(
        dense_candidates, sparse_search(query_text, top_k=depth, client=client)
    )
    return fuse_rrf(dense, sparse, k=settings.rrf_k, top_k=top_k)


async def list_sources(client: AsyncQdrantClient | None = None, limit: int = 10_000) -> list[str]:
    """Distinct document names in the collection (via the keyword index's facet counts)."""
    client = client or get_client()
    result = await client.facet(settings.qdrant_collection, key="source", limit=limit, exact=True)
    return sorted(str(hit.value) for hit in result.hits)


async def count_points(client: AsyncQdrantClient | None = None) -> int:
    client = client or get_client()
    return (await client.count(settings.qdrant_collection, exact=True)).count


async def count_source(source: str, client: AsyncQdrantClient | None = None) -> int:
    """Chunks indexed for one document."""
    client = client or get_client()
    source_filter = models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    )
    return (
        await client.count(settings.qdrant_collection, count_filter=source_filter, exact=True)
    ).count
