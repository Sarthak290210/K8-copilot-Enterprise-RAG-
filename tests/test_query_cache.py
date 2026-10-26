"""The 5-tier query cache: memory and (fake) Redis, expiry, stats, clear, per-request trace."""

import time

import pytest

from app.config import settings
from app.services import query_cache_service as qc
from app.services.query_cache_service import QueryCacheService, cache_scope


class FakePipeline:
    def __init__(self, redis):
        self.redis, self.ops = redis, []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.ops.append((name, args, kwargs))

        return queue

    async def exec(self):
        return [await getattr(self.redis, name)(*a, **kw) for name, a, kw in self.ops]


class FakeRedis:
    """Just enough of upstash_redis.asyncio.Redis (strings with TTL, sets)."""

    def __init__(self):
        self.data, self.expiry, self.sets, self.calls = {}, {}, {}, 0
        self.fail = False

    def pipeline(self):
        return FakePipeline(self)

    def _check(self):
        self.calls += 1
        if self.fail:
            raise ConnectionError("redis down")

    async def get(self, key):
        self._check()
        return self.data.get(key)

    async def ttl(self, key):
        return self.expiry.get(key, -2)

    async def set(self, key, value, ex=None):
        self._check()
        self.data[key], self.expiry[key] = value, ex
        return True

    async def sadd(self, key, *members):
        self.sets.setdefault(key, set()).update(members)
        return len(members)

    async def expire(self, key, seconds):
        return 1

    async def smembers(self, key):
        self._check()
        return list(self.sets.get(key, set()))

    async def delete(self, *keys):
        n = 0
        for key in keys:
            n += int(self.data.pop(key, None) is not None or self.sets.pop(key, None) is not None)
        return n


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", True)


@pytest.fixture
def memory_cache(enabled):
    return QueryCacheService(use_shared_redis=False)


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def redis_cache(enabled, redis):
    return QueryCacheService(redis=redis)


async def test_set_then_get_each_tier(memory_cache):
    await memory_cache.set_intent("How many pods?", "sql", 120)
    await memory_cache.set_rag_answer(
        "q", {"top_k": 5}, {"response": {"answer": "a"}, "chunks": []}
    )
    await memory_cache.set_sql_generation(
        "q", {"sql": "SELECT 1", "explanation": "", "tokens_used": 9}
    )
    await memory_cache.set_sql_result("SELECT 1", [{"n": 1}])
    await memory_cache.set_embedding("text", [0.1, 0.2])
    assert await memory_cache.get_intent("  how many   PODS? ") == {"intent": "sql", "tokens": 120}
    assert (await memory_cache.get_rag_answer("q", {"top_k": 5}))["response"]["answer"] == "a"
    assert (await memory_cache.get_sql_generation("Q"))["sql"] == "SELECT 1"
    assert await memory_cache.get_sql_result("select   1 ;") == [{"n": 1}]  # normalised SQL
    assert await memory_cache.get_embedding("text") == [0.1, 0.2]
    assert await memory_cache.get_embedding("Text") is None  # embeddings: exact text only


async def test_different_flags_never_share_an_answer(memory_cache):
    await memory_cache.set_rag_answer("q", {"top_k": 5}, {"response": {"answer": "a"}})
    assert await memory_cache.get_rag_answer("q", {"top_k": 6}) is None


async def test_expired_entries_are_misses(memory_cache, monkeypatch):
    monkeypatch.setattr(settings, "cache_ttl_intent", 10)
    await memory_cache.set_intent("q", "rag", 1)
    now = time.time()
    monkeypatch.setattr(qc.time, "time", lambda: now + 11)
    assert await memory_cache.get_intent("q") is None
    assert memory_cache.stats()["memory_entries"] == 0  # the stale entry was deleted


async def test_memory_is_bounded_lru(memory_cache, monkeypatch):
    monkeypatch.setattr(settings, "cache_memory_max_entries", 2)
    for text in ("a", "b"):
        await memory_cache.set_embedding(text, [1.0])
    await memory_cache.get_embedding("a")  # "a" is now the most recently used
    await memory_cache.set_embedding("c", [1.0])
    assert await memory_cache.get_embedding("b") is None
    assert await memory_cache.get_embedding("a") == [1.0]


async def test_stats_count_hits_misses_sets(memory_cache):
    await memory_cache.get_intent("q")
    await memory_cache.set_intent("q", "rag", 1)
    await memory_cache.get_intent("q")
    await memory_cache.get_intent("q")
    intent = memory_cache.stats()["tiers"]["intent"]
    assert intent == {"hits": 2, "misses": 1, "sets": 1, "hit_rate": round(2 / 3, 4)}
    assert memory_cache.stats()["tiers"]["sql_gen"]["hit_rate"] is None


async def test_disabled_cache_does_nothing(memory_cache, monkeypatch):
    await memory_cache.set_intent("q", "rag", 1)
    monkeypatch.setattr(settings, "cache_enabled", False)
    assert await memory_cache.get_intent("q") is None
    await memory_cache.set_intent("other", "rag", 1)
    assert memory_cache.stats()["tiers"]["intent"]["sets"] == 1


async def test_corrupt_payload_is_a_miss(redis_cache, redis):
    redis.data[redis_cache.intent_key("q")] = "{not json"
    assert await redis_cache.get_intent("q") is None


async def test_redis_hit_fills_memory(redis_cache, redis):
    await redis_cache.set_intent("q", "sql", 5)
    other_process = QueryCacheService(redis=redis)  # same Redis, empty memory
    assert await other_process.get_intent("q") == {"intent": "sql", "tokens": 5}
    calls = redis.calls
    assert await other_process.get_intent("q") is not None
    assert redis.calls == calls  # second lookup served from memory, no Redis round trip


async def test_redis_errors_fall_back_to_memory(redis_cache, redis):
    redis.fail = True
    await redis_cache.set_intent("q", "rag", 1)  # Redis write fails, memory keeps it
    assert await redis_cache.get_intent("q") == {"intent": "rag", "tokens": 1}
    assert await redis_cache.get_intent("never-set") is None  # a failed read is a miss


async def test_clear_really_deletes_redis_keys(redis_cache, redis):
    """Fix vs reference: clear() used to do nothing for Redis."""
    await redis_cache.set_intent("q1", "rag", 1)
    await redis_cache.set_intent("q2", "sql", 1)
    await redis_cache.set_embedding("t", [0.5])
    assert await redis_cache.clear("intent") == {"intent": 2}
    assert redis_cache.intent_key("q1") not in redis.data
    assert "cachekeys:intent" not in redis.sets
    assert await redis_cache.get_embedding("t") == [0.5]  # other tiers untouched
    cleared = await redis_cache.clear()
    assert cleared["embedding"] == 1 and redis.data == {}
    assert redis_cache.stats()["memory_entries"] == 0
    assert redis_cache.stats()["tiers"]["intent"] == {
        "hits": 0, "misses": 0, "sets": 0, "hit_rate": None,
    }  # fmt: skip


async def test_trace_records_hits_and_tokens_saved(memory_cache, monkeypatch):
    monkeypatch.setattr(settings, "llm_cost_per_million_tokens", 0.5)
    await memory_cache.set_intent("q", "rag", 1000)
    with cache_scope(fresh=True) as trace:
        hit = await memory_cache.get_intent("q")
        qc.note_tokens_saved(hit["tokens"])
        await memory_cache.get_rag_answer("q", {})
        with cache_scope() as inner:  # nested: the same request's trace
            assert inner is trace
    assert trace.hits == {"intent": True, "rag_answer": False}
    assert trace.tokens_saved == 1000 and trace.cost_saved_usd == 0.0005
    assert qc.current_trace() is None  # reset after the scope


def test_normalize_sql_keeps_quoted_text_exact():
    """Postgres compares 'Running' and 'running' as different: never one cached result."""
    a = qc.normalize_sql("SELECT  count(*) FROM pods WHERE status = 'Running';")
    assert a == "select count(*) from pods where status = 'Running'"
    assert a != qc.normalize_sql("select count(*) from pods where status = 'running'")
    assert qc.normalize_sql("SELECT 'a  b'") != qc.normalize_sql("SELECT 'a b'")
    assert qc.normalize_sql('SELECT "Col" FROM t') != qc.normalize_sql("SELECT col FROM t")
    assert qc.normalize_sql("SELECT 'it''s A'  FROM t") == "select 'it''s A' from t"
