"""L9 output validator (Phase 6): guarantee the shape of what the LLM returns.

- `clean_answer`: free-text answers. Strips a ```fence wrapped around the whole answer and
  rewrites gpt-oss's own citation styles (`【3】`, `【3†L1-L7】`, `[3†source]`) to plain `[3]`,
  the format the UI, the grounding check and the eval read.
- `validate_with_retry`: structured (JSON) output. Parses the reply into a pydantic model; if
  it doesn't fit, asks the LLM again with the validation error, up to MAX_VALIDATION_RETRIES
  (fix vs reference, which never wired its validator in). Used by the grounding check (L10);
  the older JSON callers (router, CRAG, SQL, HyDE, Self-RAG) keep their own fail-soft fallbacks.
"""

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import settings

_FENCE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_ODD_CITATION = re.compile(r"[【\[]\s*(\d{1,3})\s*(?:†[^】\]]*)?[】\]]")


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def normalize_citations(text: str) -> str:
    return _ODD_CITATION.sub(r"[\1]", text)


def clean_answer(text: str) -> str:
    return normalize_citations(strip_markdown_fences(text))


class OutputValidationError(ValueError):
    def __init__(self, message: str, tokens_used: int = 0) -> None:
        super().__init__(message)
        self.tokens_used = tokens_used


async def validate_with_retry[M: BaseModel](
    raw: str,
    model_cls: type[M],
    ask_again: Callable[[str], Awaitable[dict[str, Any]]],
    max_retries: int | None = None,
) -> tuple[M, int]:
    """(parsed object, tokens spent on retries). `ask_again(error)` returns an
    llm_service.generate result for a corrected reply. Raises OutputValidationError (with the
    retry tokens) if no reply fits."""
    retries = settings.max_validation_retries if max_retries is None else max_retries
    tokens = 0
    for attempt in range(retries + 1):
        try:
            return model_cls.model_validate(json.loads(strip_markdown_fences(raw))), tokens
        except (json.JSONDecodeError, ValidationError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        if attempt == retries:
            break
        result = await ask_again(error)
        tokens += result["usage"]["total_tokens"]
        raw = result["content"]
    raise OutputValidationError(f"no valid {model_cls.__name__} after {retries} retries", tokens)
