import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from fastapi import HTTPException, Request, Response, status
from loguru import logger
from upstash_redis.asyncio import Redis

from app.config import settings


@lru_cache
def get_redis_client() -> Redis | None:
    if not (settings.upstash_redis_url and settings.upstash_redis_token):
        logger.warning("Upstash Redis is not configured; rate limiting is disabled")
        return None
    return Redis(
        url=settings.upstash_redis_url,
        token=settings.upstash_redis_token.get_secret_value(),
        allow_telemetry=False,
    )


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    count: int
    retry_after: int  # seconds until a slot frees up in the window


class RateLimiter:
    """Fixed-window rate limiter backed by Upstash Redis."""

    def __init__(
        self,
        max_requests: int = settings.rate_limit_requests,
        window_seconds: int = settings.rate_limit_window_seconds,
        redis: Redis | None = None,
        name: str = "default",
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis = redis if redis is not None else get_redis_client()
        # Separate limiters (e.g. login vs register) keep separate counters.
        self.name = name

    async def is_allowed(self, key: str) -> RateLimitResult:
        """Sliding window: a sorted set of request timestamps covering the last window."""
        now = time.time()
        if self.redis is None:
            return RateLimitResult(True, self.max_requests, 0, 0)

        redis_key = f"ratelimit:{self.name}:{key}"
        # Unique member so concurrent requests at the same timestamp are all counted.
        member = f"{now}:{uuid.uuid4().hex}"
        try:
            # One round trip: Upstash sends the whole pipeline as a single HTTP request.
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(redis_key, 0, now - self.window_seconds)
            pipe.zadd(redis_key, {member: now})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, self.window_seconds)
            pipe.zrange(redis_key, 0, 0, withscores=True)  # oldest entry, for retry_after
            results = await pipe.exec()
            count = int(cast(int, results[2]))
            oldest = cast(list[tuple[str, float]], results[4])
        except Exception as exc:
            # Fail open: a Redis outage should not take the whole API down.
            logger.warning("Rate limiter Redis error ({}); allowing request", type(exc).__name__)
            return RateLimitResult(True, self.max_requests, 0, 0)

        # The window frees a slot when the oldest request in it ages out.
        oldest_ts = float(oldest[0][1]) if oldest else now
        retry_after = max(math.ceil(oldest_ts + self.window_seconds - now), 1)
        return RateLimitResult(
            allowed=count <= self.max_requests,
            remaining=max(self.max_requests - count, 0),
            count=count,
            retry_after=retry_after,
        )

    async def is_allowed_ip(self, ip: str) -> RateLimitResult:
        return await self.is_allowed(f"ip:{ip}")

    async def is_allowed_user(self, user_id: int | str) -> RateLimitResult:
        return await self.is_allowed(f"user:{user_id}")


def rate_limit_by_ip(limiter: RateLimiter) -> Callable[[Request, Response], Awaitable[None]]:
    """FastAPI dependency: 429 when the client IP exceeds `limiter`, else add X-RateLimit headers."""

    async def dependency(request: Request, response: Response) -> None:
        ip = request.client.host if request.client else "unknown"
        result = await limiter.is_allowed_ip(ip)
        headers = {
            "X-RateLimit-Limit": str(limiter.max_requests),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.retry_after),
        }
        if not result.allowed:
            logger.warning("Rate limit exceeded: {} for ip {}", limiter.name, ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests, try again later",
                headers={**headers, "Retry-After": str(result.retry_after)},
            )
        response.headers.update(headers)

    return dependency
