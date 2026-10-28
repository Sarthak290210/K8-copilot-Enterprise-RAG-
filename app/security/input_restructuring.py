"""L5 input restructuring (Phase 6): keep the user's text within the token limit.

Counts tokens with tiktoken (`o200k_base`, the gpt-oss tokenizer family; a ~4 chars/token
estimate if the encoding can't load) and cuts anything over `MAX_INPUT_TOKENS`. The plan's LLM
summary for very long inputs isn't built: a question is at most 2,000 characters (L1), well
under 3,000 tokens, so it could never trigger, and an untestable paid path isn't worth having.
"""

from functools import lru_cache
from typing import Any, Literal

from loguru import logger

from app.config import settings

Action = Literal["unchanged", "truncated"]


@lru_cache
def _encoding() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # first use downloads the encoding; DNS here is flaky
        logger.warning("tiktoken unavailable ({}); estimating tokens", type(exc).__name__)
        return None


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return (len(text) + 3) // 4
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int | None = None) -> tuple[str, Action]:
    limit = max_tokens if max_tokens is not None else settings.max_input_tokens
    if count_tokens(text) <= limit:
        return text, "unchanged"
    encoding = _encoding()
    cut = encoding.decode(encoding.encode(text)[:limit]) if encoding else text[: limit * 4]
    return cut, "truncated"


def restructure_input(text: str) -> tuple[str, Action]:
    """Within the limit -> unchanged; over -> cut to MAX_INPUT_TOKENS."""
    restructured, action = truncate_text(text)
    if action != "unchanged":
        logger.info("Input {} to {} tokens", action, settings.max_input_tokens)
    return restructured, action
