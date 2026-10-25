"""The question endpoints (Phase 4: through the agent graph).

POST /query               -> answer, or a pending SQL block to approve first
POST /query/sql/execute   -> approve (optionally edited) or reject that SQL; only the asker can
GET  /query/defaults      -> the default flags (public)
"""

import uuid
from pathlib import Path
from typing import Annotated, Any, LiteralString, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg_pool import AsyncConnectionPool

from app.api.auth import get_pool
from app.core.graph import is_waiting_for_approval, resume_agent, run_agent
from app.middleware.auth import User, get_current_user
from app.middleware.rate_limiter import RateLimiter
from app.models import ChatResponse, QueryRequest, SqlExecuteRequest
from app.services import rag_service

router = APIRouter(prefix="/query", tags=["query"])

# Per-user limit (RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS): each query costs LLM tokens.
query_limiter = RateLimiter(name="query")

QUERY_THREADS_MIGRATION = (
    Path(__file__).resolve().parents[2] / "seed" / "migrations" / "004_query_threads.sql"
)


async def ensure_query_threads_table(pool: AsyncConnectionPool) -> None:
    """Run at startup, like the users table: the ownership check needs it."""
    async with pool.connection() as conn:
        await conn.execute(cast(LiteralString, QUERY_THREADS_MIGRATION.read_text(encoding="utf-8")))


def get_graph(request: Request) -> Any:
    return request.app.state.graph


async def _check_rate_limit(user: User) -> None:
    limit = await query_limiter.is_allowed_user(user.id)
    if not limit.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many queries, try again later",
            headers={"Retry-After": str(limit.retry_after)},
        )


@router.get("/defaults")
async def query_defaults() -> dict[str, Any]:
    """The flags a query uses when it doesn't set them (from the `*_ENABLED_BY_DEFAULT`
    settings). Public: no secrets, and the UI needs them before login to start its toggles."""
    return dict(rag_service.DEFAULT_FLAGS)


@router.post("", response_model=ChatResponse)
async def query(
    body: QueryRequest,
    user: Annotated[User, Depends(get_current_user)],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
    graph: Annotated[Any, Depends(get_graph)],
) -> ChatResponse:
    await _check_rate_limit(user)
    query_id = uuid.uuid4()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO query_threads (thread_id, user_id) VALUES (%s, %s)", (query_id, user.id)
        )
    flags = body.flags.model_dump(exclude_none=True) if body.flags else None
    try:
        result = await run_agent(graph, body.question, flags, query_id, user.id)
    except ValueError as exc:  # e.g. invalid flags
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    return result.response


@router.post("/sql/execute", response_model=ChatResponse)
async def execute_sql(
    body: SqlExecuteRequest,
    user: Annotated[User, Depends(get_current_user)],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
    graph: Annotated[Any, Depends(get_graph)],
) -> ChatResponse:
    """Fix vs reference: same auth and rate limit as /query, and only the user who asked may
    decide (another user's query_id is a 404, so its existence isn't revealed)."""
    await _check_rate_limit(user)
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT user_id FROM query_threads WHERE thread_id = %s", (body.query_id,)
            )
        ).fetchone()
    if row is None or row[0] != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown query_id")
    if not await is_waiting_for_approval(graph, body.query_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This query is not waiting for approval"
        )
    result = await resume_agent(graph, body.query_id, body.approved, body.edited_sql)
    return result.response
