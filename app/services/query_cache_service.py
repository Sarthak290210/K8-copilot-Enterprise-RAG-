"""The 5-tier query cache (Phase 5): remember expensive results so repeats are instant and free.

| tier       | key built from                          | TTL (settings)         |
|------------|-----------------------------------------|------------------------|
| intent     | normalised question + grader model      | CACHE_TTL_INTENT 24 h  |
| embedding  | exact text + embedding model            | CACHE_TTL_EMBEDDINGS 7 d |
| rag_answer | normalised question + flags + models + pipeline settings | CACHE_TTL_RAG 1 h      |
| sql_gen    | normalised question + answer model + date + schema prompt | CACHE_TTL_SQL_GEN 24 h |
| sql_result | SQL, case/space folded outside quotes   | CACHE_TTL_SQL_RESULT 15 min |

Keys are `cache:<tier>:<sha256>`. Lookups go memory -> Redis (user's choice 2026-10-26): each
process keeps a small LRU copy (CACHE_MEMORY_MAX_ENTRIES) so a repeat answer skips Redis's
~44 ms round trips; Redis (Upstash) is shared across restarts and processes (API, eval). If
Redis isn't configured or errors, the memory copy alone keeps the app working. Only question
vectors are cached in the embedding tier, never document chunks (user's choice: the file-hash
dedup already skips unchanged files, and ~15 KB per vector would fill Upstash).

Every Redis key is also SADDed to `cachekeys:<tier>`, so `clear()` really deletes them (fix vs
reference, whose clear() did nothing for Redis).

Per request, a `CacheTrace` (a ContextVar) collects which tiers hit and how many LLM tokens the
hits saved; the API response reports them as `cache_hits` and `cost_saved_usd`.
"""

import asyncio
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from upstash_redis.asyncio import Redis

from app.config import settings
from app.middleware.rate_limiter import REDIS_TIMEOUT_SECONDS, get_redis_client

Tier = Literal["intent", "embedding", "rag_answer", "sql_gen", "sql_result"]
TIERS: tuple[Tier, ...] = ("intent", "embedding", "rag_answer", "sql_gen", "sql_result")
_DELETE_BATCH = 500


def _ttl(tier: Tier) -> int:
    return {
        "intent": settings.cache_ttl_intent,
        "embedding": settings.cache_ttl_embeddings,
        "rag_answer": settings.cache_ttl_rag,
        "sql_gen": settings.cache_ttl_sql_gen,
        "sql_result": settings.cache_ttl_sql_result,
    }[tier]


def normalize_question(question: str) -> str:
    return " ".join(question.split()).lower()


# Quoted text: 'string literals' and "quoted identifiers" ('' / "" escape a quote inside).
_SQL_QUOTED = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


def normalize_sql(query: str) -> str:
    """Case and whitespace are folded only outside quotes: Postgres compares 'Running' and
    'running' (and "Name" vs name) as different, so they must never share cached rows."""
    query = query.strip().rstrip(";").strip()
    parts, last = [], 0
    for match in _SQL_QUOTED.finditer(query):
        parts += [re.sub(r"\s+", " ", query[last : match.start()]).lower(), match.group()]
        last = match.end()
    parts.append(re.sub(r"\s+", " ", query[last:]).lower())
    return "".join(parts)


def tokens_to_usd(tokens: int) -> float:
    return round(tokens * settings.llm_cost_per_million_tokens / 1_000_000, 6)


@dataclass
class CacheTrace:
    """What the cache did for one request: tier -> hit (True if any lookup of it hit)."""

    hits: dict[str, bool] = field(default_factory=dict)
    tokens_saved: int = 0

    @property
    def cost_saved_usd(self) -> float:
        return tokens_to_usd(self.tokens_saved)


_trace: ContextVar[CacheTrace | None] = ContextVar("cache_trace", default=None)


@contextmanager
def cache_scope(fresh: bool = False) -> Iterator[CacheTrace]:
    """The trace of the current request: reuses the enclosing one (the agent's, when the RAG
    service runs inside the graph) unless `fresh`. Tasks started inside share the object."""
    existing = _trace.get()
    if existing is not None and not fresh:
        yield existing
        return
    trace = CacheTrace()
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def current_trace() -> CacheTrace | None:
    return _trace.get()


def note_tokens_saved(tokens: int) -> None:
    if (trace := _trace.get()) is not None:
        trace.tokens_saved += max(int(tokens or 0), 0)


async def _with_timeout[T](awaitable: Awaitable[T]) -> T:
    # The Upstash client has no timeout of its own (see rate_limiter).
    return await asyncio.wait_for(awaitable, REDIS_TIMEOUT_SECONDS)


class QueryCacheService:
    def __init__(self, redis: Redis | None = None, use_shared_redis: bool = True) -> None:
        # redis=None + use_shared_redis -> the app's Upstash client, resolved per call (never a
        # closed one after a lifespan restart). Tests pass a fake, or use_shared_redis=False.
        self._redis = redis
        self._use_shared_redis = use_shared_redis
        self._memory: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {tier: {"hits": 0, "misses": 0, "sets": 0} for tier in TIERS}

    @property
    def redis(self) -> Redis | None:
        if self._redis is not None:
            return self._redis
        return get_redis_client() if self._use_shared_redis else None

    @property
    def enabled(self) -> bool:
        return settings.cache_enabled

    # ---------------- keys ----------------

    @staticmethod
    def _key(namespace: Tier, raw: str) -> str:
        return f"cache:{namespace}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    def intent_key(self, question: str) -> str:
        return self._key("intent", f"{settings.llm_model_grader}\n{normalize_question(question)}")

    def rag_answer_key(self, question: str, cache_context: dict[str, Any]) -> str:
        # The context holds the flags that change the answer (and the models): a different
        # profile must never share an answer.
        raw = json.dumps(
            {"question": normalize_question(question), "context": cache_context},
            sort_keys=True,
            default=str,
        )
        return self._key("rag_answer", raw)

    def sql_gen_key(self, question: str, context: str = "") -> str:
        # context = what else the SQL prompt told the model (today's date, the schema and its
        # date ranges): "last month" asked tomorrow, or after a re-seed, is a different query.
        return self._key(
            "sql_gen",
            f"{settings.llm_model_answer}\n{context}\n{normalize_question(question)}",
        )

    def sql_result_key(self, query: str) -> str:
        return self._key("sql_result", normalize_sql(query))

    def embedding_key(self, text: str, model: str | None = None) -> str:
        return self._key("embedding", f"{model or settings.embedding_model}\n{text}")

    # ---------------- core ----------------

    def _record(self, tier: Tier, field_name: str) -> None:
        with self._lock:
            self._stats[tier][field_name] += 1
        if field_name in ("hits", "misses") and (trace := _trace.get()) is not None:
            trace.hits[tier] = trace.hits.get(tier, False) or field_name == "hits"

    def _memory_get(self, key: str) -> str | None:
        with self._lock:
            entry = self._memory.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= time.time():
                del self._memory[key]  # stale
                return None
            self._memory.move_to_end(key)
            return value

    def _memory_set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._memory[key] = (time.time() + ttl, value)
            self._memory.move_to_end(key)
            while len(self._memory) > max(settings.cache_memory_max_entries, 0):
                self._memory.popitem(last=False)  # least recently used

    async def _get(self, tier: Tier, key: str) -> str | None:
        if not self.enabled:
            return None
        value = self._memory_get(key)
        redis = self.redis
        if value is None and redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.get(key)
                pipe.ttl(key)
                raw, remaining = await _with_timeout(pipe.exec())
            except Exception as exc:
                logger.warning("Cache read failed ({}); treating as a miss", type(exc).__name__)
            else:
                if raw is not None:
                    value = str(raw)
                    ttl = int(remaining) if isinstance(remaining, int) and remaining > 0 else 0
                    self._memory_set(key, value, ttl or _ttl(tier))
        self._record(tier, "hits" if value is not None else "misses")
        return value

    async def _set(self, tier: Tier, key: str, value: str, ttl: int | None = None) -> None:
        if not self.enabled:
            return
        ttl = ttl or _ttl(tier)
        self._memory_set(key, value, ttl)
        redis = self.redis
        if redis is not None:
            tracker = f"cachekeys:{tier}"
            try:
                pipe = redis.pipeline()
                pipe.set(key, value, ex=ttl)
                pipe.sadd(tracker, key)  # so clear() can find it
                pipe.expire(tracker, ttl)  # same tier, same TTL: outlives every member
                await _with_timeout(pipe.exec())
            except Exception as exc:
                logger.warning("Cache write failed ({}); kept in memory only", type(exc).__name__)
        self._record(tier, "sets")

    async def _get_json(self, tier: Tier, key: str) -> Any:
        raw = await self._get(tier, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            logger.warning("Corrupt {} cache entry; ignoring it", tier)
            return None

    async def _set_json(self, tier: Tier, key: str, value: Any) -> None:
        await self._set(tier, key, json.dumps(value, separators=(",", ":"), default=str))

    # ---------------- typed tiers ----------------

    async def get_intent(self, question: str) -> dict[str, Any] | None:
        """{"intent", "tokens"} (tokens = what the original routing call cost)."""
        data = await self._get_json("intent", self.intent_key(question))
        return data if isinstance(data, dict) and isinstance(data.get("intent"), str) else None

    async def set_intent(self, question: str, intent: str, tokens: int) -> None:
        await self._set_json(
            "intent", self.intent_key(question), {"intent": intent, "tokens": tokens}
        )

    async def get_rag_answer(
        self, question: str, cache_context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """{"response": ChatResponse JSON, "chunks": [RetrievedChunk JSON]}."""
        data = await self._get_json("rag_answer", self.rag_answer_key(question, cache_context))
        return data if isinstance(data, dict) and isinstance(data.get("response"), dict) else None

    async def set_rag_answer(
        self, question: str, cache_context: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        await self._set_json("rag_answer", self.rag_answer_key(question, cache_context), payload)

    async def get_sql_generation(self, question: str, context: str = "") -> dict[str, Any] | None:
        """{"sql", "explanation", "tokens_used"}."""
        data = await self._get_json("sql_gen", self.sql_gen_key(question, context))
        return data if isinstance(data, dict) and isinstance(data.get("sql"), str) else None

    async def set_sql_generation(
        self, question: str, generated: dict[str, Any], context: str = ""
    ) -> None:
        await self._set_json("sql_gen", self.sql_gen_key(question, context), generated)

    async def get_sql_result(self, query: str) -> list[dict[str, Any]] | None:
        data = await self._get_json("sql_result", self.sql_result_key(query))
        return data if isinstance(data, list) else None

    async def set_sql_result(self, query: str, rows: list[dict[str, Any]]) -> None:
        await self._set_json("sql_result", self.sql_result_key(query), rows)

    async def get_embedding(self, text: str, model: str | None = None) -> list[float] | None:
        data = await self._get_json("embedding", self.embedding_key(text, model))
        return data if isinstance(data, list) else None

    async def set_embedding(self, text: str, vector: list[float], model: str | None = None) -> None:
        await self._set_json("embedding", self.embedding_key(text, model), vector)

    # ---------------- admin ----------------

    def stats(self) -> dict[str, Any]:
        """Per tier: hits, misses, sets, hit_rate (hits / lookups, None before any lookup).
        Counted per process since start (or the last clear)."""
        with self._lock:
            tiers = {}
            for tier, counts in self._stats.items():
                lookups = counts["hits"] + counts["misses"]
                tiers[tier] = {
                    **counts,
                    "hit_rate": round(counts["hits"] / lookups, 4) if lookups else None,
                }
            memory_entries = len(self._memory)
        return {
            "enabled": self.enabled,
            "backend": "redis+memory" if self.redis is not None else "memory",
            "memory_entries": memory_entries,
            "tiers": tiers,
        }

    async def clear(self, tier: Tier | None = None) -> dict[str, int]:
        """Delete every cached entry of `tier` (all tiers if None) from Redis and memory, and
        reset those tiers' stats. Returns {tier: entries deleted}. Redis errors propagate."""
        tiers: tuple[Tier, ...] = (tier,) if tier else TIERS
        cleared: dict[str, int] = {}
        redis = self.redis
        for name in tiers:
            deleted = 0
            if redis is not None:
                tracker = f"cachekeys:{name}"
                keys = [str(k) for k in (await _with_timeout(redis.smembers(tracker)) or [])]
                for i in range(0, len(keys), _DELETE_BATCH):
                    deleted += int(await _with_timeout(redis.delete(*keys[i : i + _DELETE_BATCH])))
                await _with_timeout(redis.delete(tracker))
            prefix = f"cache:{name}:"
            with self._lock:
                stale = [key for key in self._memory if key.startswith(prefix)]
                for key in stale:
                    del self._memory[key]
                self._stats[name] = {"hits": 0, "misses": 0, "sets": 0}
            cleared[name] = max(deleted, len(stale))
        logger.info("Cache cleared: {}", cleared)
        return cleared


query_cache = QueryCacheService()
