"""The only place that talks to the chat LLM API (Cerebras, OpenAI-compatible)."""

import json
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
    """The model was asked for JSON but returned something else."""


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
        raise LLMJSONError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMJSONError("Model returned JSON that is not an object")
    return {"data": data, "model": result["model"], "usage": result["usage"]}
