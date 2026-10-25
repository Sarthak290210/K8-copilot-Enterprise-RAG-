"""The only place that talks to the chat LLM API (Cerebras, OpenAI-compatible)."""

import asyncio
import json
import time
from functools import lru_cache
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import settings

# The SDK retries 429s/5xx itself with exponential backoff and honours Retry-After.
# Generous because the Cerebras free trial allows only 5 requests per minute.
LLM_MAX_RETRIES = 6
LLM_TIMEOUT_SECONDS = 60


@lru_cache
def get_client() -> AsyncOpenAI:
    """Shared Cerebras client; main.py's lifespan closes it and clears this cache."""
    return AsyncOpenAI(
        api_key=settings.cerebras_api_key.get_secret_value(),
        base_url=settings.cerebras_base_url,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def is_reasoning_model(model: str) -> bool:
    """gpt-oss takes `reasoning_effort`; other models may reject it or start reasoning."""
    return "gpt-oss" in model


_pace_lock: asyncio.Lock | None = None
_pace_loop: asyncio.AbstractEventLoop | None = None
_last_call_at = 0.0


async def _pace() -> None:
    """Space calls at least `llm_min_interval_seconds` apart (no-op when it's 0).

    Rate limits are per account, so pacing is process-wide; the lock makes concurrent callers
    queue instead of all firing at once.
    """
    global _pace_lock, _pace_loop, _last_call_at
    interval = settings.llm_min_interval_seconds
    if interval <= 0:
        return
    loop = asyncio.get_running_loop()
    if _pace_lock is None or _pace_loop is not loop:
        # An asyncio.Lock belongs to one event loop: a new loop (another asyncio.run, a test)
        # gets a new lock instead of 'bound to a different event loop'.
        _pace_lock, _pace_loop = asyncio.Lock(), loop
    async with _pace_lock:
        wait = _last_call_at + interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_at = time.monotonic()


async def generate(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> dict[str, Any]:
    """One chat completion -> {"content", "model", "usage": {prompt/completion/total tokens}}.

    gpt-oss's reasoning comes back in a separate field on Cerebras, never in `content`.
    """
    model = model or settings.llm_model_answer
    # Optional params are only sent when used: the SDK would send None as JSON null.
    extra: dict[str, Any] = {}
    if is_reasoning_model(model):
        extra["reasoning_effort"] = settings.llm_reasoning_effort
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    await _pace()
    response = await get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        **extra,
    )
    usage = response.usage
    return {
        "content": (response.choices[0].message.content or "").strip(),
        "model": response.model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
    }


class LLMJSONError(ValueError):
    """The model was asked for JSON but returned something else.

    `usage` holds the tokens the failed call still spent, so callers can count them.
    """

    def __init__(self, message: str, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}


def tokens_spent(exc: BaseException) -> int:
    """Tokens a failed call still used: an LLMJSONError's usage, otherwise 0 (the call never
    returned, e.g. a network error or a rate limit the SDK gave up on)."""
    return exc.usage.get("total_tokens", 0) if isinstance(exc, LLMJSONError) else 0


async def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Same as generate, in JSON mode; returns {"data": parsed JSON, "model", "usage"}."""
    result = await generate(system_prompt, user_message, model, temperature, json_mode=True)
    try:
        data = json.loads(result["content"])
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON ({} chars)", len(result["content"]))
        raise LLMJSONError(f"Model returned invalid JSON: {exc}", result["usage"]) from exc
    if not isinstance(data, dict):
        raise LLMJSONError("Model returned JSON that is not an object", result["usage"])
    return {"data": data, "model": result["model"], "usage": result["usage"]}
