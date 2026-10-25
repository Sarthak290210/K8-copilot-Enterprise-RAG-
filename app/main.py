import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.api import admin, auth, documents, query
from app.config import settings
from app.core.graph import build_graph
from app.middleware.rate_limiter import get_redis_client
from app.services import llm_service, vector_store, web_search


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
        await query.ensure_query_threads_table(pool)
        app.state.db_pool = pool

        # Agent graph (Phase 4) with a Postgres checkpointer: a run paused for SQL approval
        # survives until POST /query/sql/execute resumes it (setup() creates its tables).
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url.get_secret_value())
        )
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)

        # Shared with the services (vector_store / llm_service own the cached clients).
        app.state.qdrant = vector_store.get_client()
        stack.push_async_callback(app.state.qdrant.close)
        stack.callback(vector_store.get_client.cache_clear)
        await vector_store.ensure_collection()
        try:
            # Build the BM25 index now, so the first sparse/hybrid query doesn't pay for it.
            await vector_store.sparse_search("warm up", top_k=1)
        except Exception as exc:
            # Not fatal: the index builds on the first sparse/hybrid query instead.
            logger.warning("Could not warm the BM25 index ({})", type(exc).__name__)
        stack.callback(vector_store.invalidate_sparse_index)

        app.state.redis = get_redis_client()
        if app.state.redis is not None:
            stack.push_async_callback(app.state.redis.close)
            # Drop the closed client from the cache so a later startup (e.g. tests) gets a new one.
            stack.callback(get_redis_client.cache_clear)

        app.state.llm = llm_service.get_client()  # Cerebras (OpenAI-compatible)
        stack.push_async_callback(app.state.llm.close)
        stack.callback(llm_service.get_client.cache_clear)

        app.state.tavily = web_search.get_client()  # None without TAVILY_API_KEY
        if app.state.tavily is not None:
            stack.push_async_callback(app.state.tavily.close)
        stack.callback(web_search.get_client.cache_clear)

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
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"name": "k8-copilot", "status": "ok"}


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
