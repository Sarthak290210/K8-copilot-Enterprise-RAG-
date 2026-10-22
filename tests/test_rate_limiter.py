import asyncio

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.middleware.rate_limiter import RateLimiter, rate_limit_by_ip


async def test_21st_call_in_window_is_blocked():
    limiter = RateLimiter(max_requests=20, window_seconds=60, name="t")
    results = [await limiter.is_allowed_user(42) for _ in range(21)]
    assert all(r.allowed for r in results[:20])
    blocked = results[20]
    assert not blocked.allowed
    assert blocked.remaining == 0
    assert 1 <= blocked.retry_after <= 60


async def test_keys_are_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60, name="t")
    assert (await limiter.is_allowed_ip("1.1.1.1")).allowed
    assert (await limiter.is_allowed_ip("2.2.2.2")).allowed
    assert not (await limiter.is_allowed_ip("1.1.1.1")).allowed
    # A different limiter name keeps its own counters.
    assert (await RateLimiter(1, 60, name="other").is_allowed_ip("1.1.1.1")).allowed


async def test_window_slides(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.middleware.rate_limiter.time.time", lambda: clock[0])
    limiter = RateLimiter(max_requests=2, window_seconds=10, name="t")
    assert (await limiter.is_allowed("k")).allowed
    assert (await limiter.is_allowed("k")).allowed
    assert not (await limiter.is_allowed("k")).allowed
    clock[0] += 11  # all three timestamps are now outside the window
    assert (await limiter.is_allowed("k")).allowed


class _BrokenRedis:
    def pipeline(self):
        raise ConnectionError("redis down")


class _HangingPipeline:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    async def exec(self):
        await asyncio.sleep(60)


class _HangingRedis:
    def pipeline(self):
        return _HangingPipeline()


async def test_redis_error_fails_open():
    result = await RateLimiter(1, 60, redis=_BrokenRedis(), name="t").is_allowed("k")  # type: ignore[arg-type]
    assert result.allowed


async def test_hung_redis_times_out_and_fails_open(monkeypatch):
    monkeypatch.setattr("app.middleware.rate_limiter.REDIS_TIMEOUT_SECONDS", 0.05)
    result = await RateLimiter(1, 60, redis=_HangingRedis(), name="t").is_allowed("k")  # type: ignore[arg-type]
    assert result.allowed


def test_dependency_returns_429_with_headers():
    app = FastAPI()
    limiter = RateLimiter(max_requests=5, window_seconds=60, name="login-test")

    @app.post("/login", dependencies=[Depends(rate_limit_by_ip(limiter))])
    async def login() -> dict:
        return {"ok": True}

    client = TestClient(app)
    for _ in range(5):
        r = client.post("/login")
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Limit"] == "5"
    sixth = client.post("/login")
    assert sixth.status_code == 429
    assert int(sixth.headers["Retry-After"]) >= 1
    assert sixth.headers["X-RateLimit-Remaining"] == "0"
