"""L9: answer clean-up and validate-with-retry for structured output."""

import pytest
from pydantic import BaseModel

from app.security.output_validator import (
    OutputValidationError,
    clean_answer,
    validate_with_retry,
)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("A Pod 【1】 runs containers【2†L1-L7】.", "A Pod [1] runs containers[2]."),
        ("See [3†source].", "See [3]."),
        ("```markdown\nA Pod [1]\n```", "A Pod [1]"),
        ("Use `kubectl get pods` [1]", "Use `kubectl get pods` [1]"),
    ],
)
def test_clean_answer(raw, clean):
    assert clean_answer(raw) == clean


class Verdict(BaseModel):
    ok: bool


async def test_valid_reply_needs_no_retry():
    async def never(error):
        raise AssertionError("no retry expected")

    assert await validate_with_retry('```json\n{"ok": true}\n```', Verdict, never) == (
        Verdict(ok=True), 0,
    )  # fmt: skip


async def test_bad_reply_is_retried_with_the_error():
    errors = []

    async def ask_again(error):
        errors.append(error)
        return {"content": '{"ok": false}', "usage": {"total_tokens": 7}}

    assert await validate_with_retry('{"ok": "maybe"}', Verdict, ask_again) == (
        Verdict(ok=False), 7,
    )  # fmt: skip
    assert errors and errors[0].startswith("ValidationError")


async def test_gives_up_after_max_retries():
    async def still_bad(error):
        return {"content": "not json", "usage": {"total_tokens": 5}}

    with pytest.raises(OutputValidationError) as exc:
        await validate_with_retry("nope", Verdict, still_bad, max_retries=2)
    assert exc.value.tokens_used == 10
