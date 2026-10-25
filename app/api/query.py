from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.middleware.auth import User, get_current_user
from app.middleware.rate_limiter import RateLimiter
from app.models import ChatResponse, QueryRequest
from app.services import rag_service
from app.services.rag_service import run_rag

router = APIRouter(prefix="/query", tags=["query"])

# Per-user limit (RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS): each query costs LLM tokens.
query_limiter = RateLimiter(name="query")


@router.get("/defaults")
async def query_defaults() -> dict[str, Any]:
    """The flags a query uses when it doesn't set them (from the `*_ENABLED_BY_DEFAULT`
    settings). Public: no secrets, and the UI needs them before login to start its toggles."""
    return dict(rag_service.DEFAULT_FLAGS)


@router.post("", response_model=ChatResponse)
async def query(
    body: QueryRequest, user: Annotated[User, Depends(get_current_user)]
) -> ChatResponse:
    limit = await query_limiter.is_allowed_user(user.id)
    if not limit.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many queries, try again later",
            headers={"Retry-After": str(limit.retry_after)},
        )
    try:
        flags = body.flags.model_dump(exclude_none=True) if body.flags else None
        return await run_rag(body.question, flags)
    except ValueError as exc:  # e.g. a search_mode that doesn't exist yet
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
