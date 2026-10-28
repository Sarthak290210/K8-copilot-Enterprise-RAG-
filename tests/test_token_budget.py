"""L6: per-user daily token budget (memory fallback; Redis is faked elsewhere)."""

import pytest

from app.security import token_budget as tb


@pytest.fixture(autouse=True)
def _fresh():
    tb._memory_used.clear()
    yield
    tb._memory_used.clear()


async def test_check_then_consume_actual_tokens():
    budget = tb.TokenBudget(max_tokens=1000)
    assert (await budget.check_budget(1, 600)).allowed
    after = await budget.consume(1, 700)  # actual usage, not the estimate
    assert (after.used, after.remaining) == (700, 300)
    status = await budget.check_budget(1, 400)
    assert not status.allowed and status.remaining == 300


async def test_users_and_days_are_separate():
    budget = tb.TokenBudget(max_tokens=100)
    await budget.consume(1, 100)
    assert (await budget.check_budget(2, 50)).allowed
    assert tb.TokenBudget._key(1) != tb.TokenBudget._key(2)
    assert tb.TokenBudget._key(1).startswith("budget:1:")


async def test_zero_tokens_changes_nothing():
    budget = tb.TokenBudget(max_tokens=100)
    assert (await budget.consume(1, 0)).used == 0


class BrokenRedis:
    def pipeline(self):
        raise ConnectionError("down")

    async def get(self, key):
        raise ConnectionError("down")


async def test_redis_errors_fall_back_to_memory():
    budget = tb.TokenBudget(max_tokens=100, redis=BrokenRedis())  # type: ignore[arg-type]
    assert (await budget.consume(1, 60)).used == 60
    assert not (await budget.check_budget(1, 50)).allowed
