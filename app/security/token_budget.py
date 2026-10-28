"""L6 token budget (Phase 6): cap each user's LLM tokens per day.

`check_budget` runs before the agent with an estimate (question + reserved context and output
tokens): a user whose used + estimate would pass `MAX_TOKENS_PER_USER_DAILY` gets a 429.
`consume` runs after, with the **actual** tokens of every LLM call in the request
(`metadata.tokens_used`; fix vs reference, which consumed the estimate).

Counters live in Upstash (`budget:{user_id}:{YYYY-MM-DD}`, INCRBY + EXPIRE 2 days, so a day
resets by key name). Without Redis, or if it errors/hangs, a per-process memory counter takes
over, like the rate limiter's fallback: cost control keeps working, just not shared.
"""

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import cast

from loguru import logger
from upstash_redis.asyncio import Redis

from app.config import settings
from app.middleware.rate_limiter import REDIS_TIMEOUT_SECONDS, get_redis_client

BUDGET_TTL_SECONDS = 2 * 24 * 3600

_memory_used: dict[str, int] = {}


@dataclass(frozen=True)
class BudgetStatus:
    allowed: bool
    used: int
    remaining: int


class TokenBudget:
    def __init__(self, max_tokens: int | None = None, redis: Redis | None = None) -> None:
        self.max_tokens = (
            max_tokens if max_tokens is not None else settings.max_tokens_per_user_daily
        )
        self._redis = redis

    @property
    def redis(self) -> Redis | None:
        return self._redis if self._redis is not None else get_redis_client()

    @staticmethod
    def _key(user_id: int, day: dt.date | None = None) -> str:
        return f"budget:{user_id}:{(day or dt.date.today()).isoformat()}"

    async def used(self, user_id: int) -> int:
        key = self._key(user_id)
        redis = self.redis
        if redis is not None:
            try:
                value = await asyncio.wait_for(redis.get(key), REDIS_TIMEOUT_SECONDS)
                return int(value or 0)
            except Exception as exc:
                logger.warning("Token budget read failed ({}); using memory", type(exc).__name__)
        return _memory_used.get(key, 0)

    async def check_budget(self, user_id: int, estimated_tokens: int) -> BudgetStatus:
        used = await self.used(user_id)
        remaining = max(self.max_tokens - used, 0)
        return BudgetStatus(used + estimated_tokens <= self.max_tokens, used, remaining)

    async def consume(self, user_id: int, actual_tokens: int) -> BudgetStatus:
        """Adds the tokens this request really spent; returns the new totals."""
        key = self._key(user_id)
        if actual_tokens <= 0:
            used = await self.used(user_id)
        else:
            used = await self._add(key, actual_tokens)
        return BudgetStatus(used <= self.max_tokens, used, max(self.max_tokens - used, 0))

    async def _add(self, key: str, tokens: int) -> int:
        redis = self.redis
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.incrby(key, tokens)
                pipe.expire(key, BUDGET_TTL_SECONDS)
                results = await asyncio.wait_for(pipe.exec(), REDIS_TIMEOUT_SECONDS)
                return int(cast(int, results[0]))
            except Exception as exc:
                logger.warning("Token budget write failed ({}); using memory", type(exc).__name__)
        _memory_used[key] = _memory_used.get(key, 0) + tokens
        return _memory_used[key]


token_budget = TokenBudget()


async def check_budget(user_id: int, estimated_tokens: int) -> BudgetStatus:
    return await token_budget.check_budget(user_id, estimated_tokens)


async def consume_budget(user_id: int, actual_tokens: int) -> BudgetStatus:
    return await token_budget.consume(user_id, actual_tokens)
