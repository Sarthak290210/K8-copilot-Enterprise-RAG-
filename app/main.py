import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from groq import AsyncGroq
from loguru import logger
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient

from app.api import admin, auth
from app.config import settings
from app.middleware.rate_limiter import get_redis_client


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, serialize=settings.log_json)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Starting k8-copilot API")
    # The exit stack closes everything opened so far, even if startup fails midway.
    async with AsyncExitStack() as stack:
        pool = AsyncConnectionPool(
            settings.database_url.get_secret_value(),
            connection_class=AsyncConnection,  # explicit so type checkers know the pool's type
            open=False,
        )
        # Fail fast (10s) instead of psycopg_pool's 30s default if Postgres is unreachable.
        await pool.open(wait=True, timeout=10)
        stack.push_async_callback(pool.close)
        await auth.ensure_users_table(pool)
        app.state.db_pool = pool

        app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=3)
        stack.push_async_callback(app.state.qdrant.close)

        app.state.redis = get_redis_client()
        if app.state.redis is not None:
            stack.push_async_callback(app.state.redis.close)

        # AsyncGroq appends /openai/v1 itself, so strip it from the configured URL.
        app.state.groq = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url.removesuffix("/openai/v1"),
            timeout=10,
        )
        stack.push_async_callback(app.state.groq.close)

        try:
            yield
        finally:
            logger.info("Shutting down k8-copilot API")


app = FastAPI(
    title="k8-copilot",
    description="Advanced RAG: Text2SQL + Core RAG + Caching + LLM Security",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(admin.router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
