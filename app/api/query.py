from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.middleware.auth import User, get_current_user
from app.middleware.rate_limiter import RateLimiter
from app.models import ChatResponse, QueryRequest
from app.services.rag_service import run_rag

router = APIRouter(prefix="/query", tags=["query"])

# Per-user limit (RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS): each query costs LLM tokens.
query_limiter = RateLimiter(name="query")


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
        return await run_rag(body.question, body.flags)
    except ValueError as exc:  # e.g. a search_mode that doesn't exist yet
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
