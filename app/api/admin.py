import asyncio
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from openai import AsyncOpenAI
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from upstash_redis.asyncio import Redis

from app.api.auth import get_pool
from app.middleware.auth import require_admin

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


@router.get("/health", response_model=HealthReport)
async def services_health(
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    redis: Annotated[Redis | None, Depends(get_redis)],
    llm: Annotated[AsyncOpenAI, Depends(get_llm)],
) -> HealthReport:
    services = list(
        await asyncio.gather(
            ping_postgres(pool), ping_qdrant(qdrant), ping_redis(redis), ping_cerebras(llm)
        )
    )
    overall = "ok" if all(s.status == "up" for s in services) else "degraded"
    return HealthReport(status=overall, services=services)
