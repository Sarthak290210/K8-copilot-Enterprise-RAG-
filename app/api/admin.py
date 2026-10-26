import asyncio
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from tavily import AsyncTavilyClient
from upstash_redis.asyncio import Redis

from app.api.auth import get_pool
from app.middleware.auth import require_admin
from app.services.query_cache_service import Tier, query_cache

PING_TIMEOUT_SECONDS = 3


class ServiceStatus(BaseModel):
    service: str
    status: Literal["up", "down"]
    latency_ms: float | None = None
    error: str | None = None


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]  # "degraded" if any service is down
    services: list[ServiceStatus]


def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant


def get_redis(request: Request) -> Redis | None:
    return request.app.state.redis


def get_llm(request: Request) -> AsyncOpenAI:
    return request.app.state.llm


def get_tavily(request: Request) -> AsyncTavilyClient | None:
    return getattr(request.app.state, "tavily", None)  # None when TAVILY_API_KEY isn't set


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


async def _select_one(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute("SELECT 1")


async def ping_postgres(pool: AsyncConnectionPool) -> ServiceStatus:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(_select_one(pool), timeout=PING_TIMEOUT_SECONDS)
    except Exception as exc:
        # Only the exception type: psycopg messages can include the connection string.
        return ServiceStatus(service="postgres", status="down", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(service="postgres", status="up", latency_ms=latency_ms)


async def ping_qdrant(client: AsyncQdrantClient) -> ServiceStatus:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(client.get_collections(), timeout=PING_TIMEOUT_SECONDS)
    except Exception as exc:
        return ServiceStatus(service="qdrant", status="down", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(service="qdrant", status="up", latency_ms=latency_ms)


async def ping_redis(client: Redis | None) -> ServiceStatus:
    if client is None:
        return ServiceStatus(service="redis", status="down", error="NotConfigured")
    start = time.perf_counter()
    try:
        await asyncio.wait_for(client.ping(), timeout=PING_TIMEOUT_SECONDS)
    except Exception as exc:
        return ServiceStatus(service="redis", status="down", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(service="redis", status="up", latency_ms=latency_ms)


async def ping_cerebras(client: AsyncOpenAI) -> ServiceStatus:
    # Listing models checks reachability + API key without spending tokens.
    start = time.perf_counter()
    try:
        await asyncio.wait_for(client.models.list(), timeout=PING_TIMEOUT_SECONDS)
    except Exception as exc:
        return ServiceStatus(service="cerebras", status="down", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(service="cerebras", status="up", latency_ms=latency_ms)


async def ping_tavily(client: AsyncTavilyClient | None) -> ServiceStatus:
    """A one-result search: checks reachability + API key. Costs 1 Tavily credit."""
    if client is None:
        return ServiceStatus(service="tavily", status="down", error="NotConfigured")
    start = time.perf_counter()
    try:
        await asyncio.wait_for(
            client.search("kubernetes", max_results=1, search_depth="basic"),
            timeout=PING_TIMEOUT_SECONDS * 2,  # a real search is slower than a list call
        )
    except Exception as exc:
        return ServiceStatus(service="tavily", status="down", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(service="tavily", status="up", latency_ms=latency_ms)


@router.get("/health/postgres", response_model=ServiceStatus)
async def postgres_health(
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> ServiceStatus:
    return await ping_postgres(pool)


@router.get("/health/qdrant", response_model=ServiceStatus)
async def qdrant_health(
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
) -> ServiceStatus:
    return await ping_qdrant(client)


@router.get("/health/redis", response_model=ServiceStatus)
async def redis_health(
    client: Annotated[Redis | None, Depends(get_redis)],
) -> ServiceStatus:
    return await ping_redis(client)


@router.get("/health/cerebras", response_model=ServiceStatus)
async def cerebras_health(
    client: Annotated[AsyncOpenAI, Depends(get_llm)],
) -> ServiceStatus:
    return await ping_cerebras(client)


@router.get("/health/tavily", response_model=ServiceStatus)
async def tavily_health(
    client: Annotated[AsyncTavilyClient | None, Depends(get_tavily)],
) -> ServiceStatus:
    return await ping_tavily(client)


@router.get("/health", response_model=HealthReport)
async def services_health(
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    redis: Annotated[Redis | None, Depends(get_redis)],
    llm: Annotated[AsyncOpenAI, Depends(get_llm)],
) -> HealthReport:
    # Tavily is left out on purpose: its ping is a real search (1 credit of 1,000/month), so a
    # dashboard polling this route would burn the credits. Check it at /admin/health/tavily.
    services = list(
        await asyncio.gather(
            ping_postgres(pool),
            ping_qdrant(qdrant),
            ping_redis(redis),
            ping_cerebras(llm),
        )
    )
    overall = "ok" if all(s.status == "up" for s in services) else "degraded"
    return HealthReport(status=overall, services=services)


# ---------------- cache admin (Phase 5) ----------------


class TierStats(BaseModel):
    hits: int
    misses: int
    sets: int
    hit_rate: str | None  # e.g. "75.0%", None before the first lookup


class CacheStats(BaseModel):
    enabled: bool
    backend: str  # "redis+memory" | "memory"
    memory_entries: int
    tiers: dict[str, TierStats]


class CacheClearResult(BaseModel):
    cleared: dict[str, int]  # tier -> entries deleted


def _tier(counts: dict) -> TierStats:
    rate = counts.get("hit_rate")
    return TierStats(
        hits=counts["hits"],
        misses=counts["misses"],
        sets=counts["sets"],
        hit_rate=f"{rate * 100:.1f}%" if rate is not None else None,
    )


@router.get("/cache/stats", response_model=CacheStats)
async def cache_stats() -> CacheStats:
    """Hits / misses / sets per tier, counted by this API process since start (or the last
    clear). An eval run counts in its own process and records its stats in the result file."""
    stats = query_cache.stats()
    return CacheStats(
        enabled=stats["enabled"],
        backend=stats["backend"],
        memory_entries=stats["memory_entries"],
        tiers={name: _tier(counts) for name, counts in stats["tiers"].items()},
    )


@router.post("/cache/clear", response_model=CacheClearResult)
async def cache_clear(tier: Tier | None = None) -> CacheClearResult:
    """Delete every cached entry (or one tier's) from Redis and memory."""
    try:
        cleared = await query_cache.clear(tier)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Cache clear failed ({type(exc).__name__})"
        ) from None
    return CacheClearResult(cleared=cleared)
