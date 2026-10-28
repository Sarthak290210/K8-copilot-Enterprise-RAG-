"""The question endpoints (Phase 4: through the agent graph).

POST /query               -> answer, or a pending SQL block to approve first
POST /query/sql/execute   -> approve (optionally edited) or reject that SQL; only the asker can
GET  /query/defaults      -> the default flags (public)

Phase 6 layer order (plan.md), the same on both POST endpoints where a layer applies:
L1 request validation (QueryRequest) -> L4a JWT -> L4b rate limit -> L6 token budget check
-> L5 restructure + L2 input guard + input PII redaction (security/pipeline.py)
-> agent (L8 spotlighting, L3 prompt, L9, L10 grounding) -> L7 moderation + PII redaction
-> L6 consume the actual tokens. Refusals are 4xx with a reason code in `detail.reason`:
input_blocked (400), budget_exceeded (429), rate_limited (429).
"""

import uuid
from pathlib import Path
from typing import Annotated, Any, LiteralString, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg_pool import AsyncConnectionPool

from app.api.auth import get_pool
from app.config import settings
from app.core.graph import is_waiting_for_approval, resume_agent, run_agent
from app.middleware.auth import User, get_current_user
from app.middleware.rate_limiter import RateLimiter
from app.models import ChatResponse, QueryRequest, SqlExecuteRequest
from app.security.input_restructuring import count_tokens
from app.security.pipeline import InputBlocked, screen_question, screen_response
from app.security.token_budget import check_budget, consume_budget
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


def _refusal(code: int, reason: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(code, detail={"reason": reason, "message": message, **extra})


async def _check_rate_limit(user: User) -> None:
    limit = await query_limiter.is_allowed_user(user.id)
    if not limit.allowed:
        error = _refusal(429, "rate_limited", "Too many queries, try again later")
        error.headers = {"Retry-After": str(limit.retry_after)}
        raise error


def _estimate_tokens(question: str) -> int:
    """For the budget pre-check only; the actual usage is what gets consumed."""
    return (
        count_tokens(question) + settings.reserved_context_tokens + settings.reserved_output_tokens
    )


async def _check_budget(user: User, question: str = "") -> None:
    budget = await check_budget(user.id, _estimate_tokens(question))
    if not budget.allowed:
        raise _refusal(
            429, "budget_exceeded", "Daily token budget used up; it resets tomorrow.",
            used=budget.used, limit=settings.max_tokens_per_user_daily,
        )  # fmt: skip


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
    await _check_budget(user, body.question)
    try:
        screened = await screen_question(body.question)
    except InputBlocked as exc:
        raise _refusal(400, "input_blocked", exc.message, category=exc.reason) from None
    query_id = uuid.uuid4()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO query_threads (thread_id, user_id) VALUES (%s, %s)", (query_id, user.id)
        )
    flags = body.flags.model_dump(exclude_none=True) if body.flags else None
    try:
        result = await run_agent(graph, screened.question, flags, query_id, user.id)
    except ValueError as exc:  # e.g. invalid flags
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    response = await screen_response(result.response, screened)
    await consume_budget(user.id, response.metadata.tokens_used)
    return response


@router.post("/sql/execute", response_model=ChatResponse)
async def execute_sql(
    body: SqlExecuteRequest,
    user: Annotated[User, Depends(get_current_user)],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
    graph: Annotated[Any, Depends(get_graph)],
) -> ChatResponse:
    """Fix vs reference: same auth, rate limit, token budget and output layers as /query, and
    only the user who asked may decide (another user's query_id is a 404, so its existence isn't
    revealed). Its only text input is edited SQL, which sqlglot re-validates (graph)."""
    await _check_rate_limit(user)
    await _check_budget(user)
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
    response = await screen_response(result.response)
    await consume_budget(user.id, response.metadata.tokens_used)
    return response
