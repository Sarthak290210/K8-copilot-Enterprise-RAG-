import time
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import llm_service


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


@pytest.fixture
def fake_llm(monkeypatch):
    def install(content):
        completions = FakeCompletions(content)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm_service, "get_client", lambda: client)
        return completions

    return install


async def test_generate(fake_llm):
    completions = fake_llm("  A Pod is ...  ")
    result = await llm_service.generate("system", "question")
    assert result == {
        "content": "A Pod is ...",
        "model": settings.llm_model_answer,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    sent = completions.kwargs
    assert sent["messages"][0] == {"role": "system", "content": "system"}
    assert sent["temperature"] == 0.0
    assert sent["reasoning_effort"] == settings.llm_reasoning_effort  # gpt-oss answer model
    assert "response_format" not in sent  # never sent as JSON null


async def test_non_reasoning_model_gets_no_reasoning_effort(fake_llm):
    completions = fake_llm("ok")
    await llm_service.generate("system", "q", model="qwen-3.8-27b")
    assert "reasoning_effort" not in completions.kwargs


async def test_generate_with_json(fake_llm):
    completions = fake_llm('{"intent": "rag"}')
    result = await llm_service.generate_with_json("system", "q", model="gpt-oss-120b")
    assert result["data"] == {"intent": "rag"}
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["model"] == "gpt-oss-120b"


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
async def test_generate_with_json_rejects_bad_output(fake_llm, content):
    fake_llm(content)
    with pytest.raises(llm_service.LLMJSONError) as caught:
        await llm_service.generate_with_json("system", "q")
    # the failed call was still billed: callers can count it
    assert caught.value.usage["total_tokens"] == 15
    assert llm_service.tokens_spent(caught.value) == 15


def test_tokens_spent_is_zero_for_other_errors():
    assert llm_service.tokens_spent(RuntimeError("network")) == 0


async def test_pacing_spaces_calls(fake_llm, monkeypatch):
    fake_llm("ok")
    monkeypatch.setattr(llm_service.settings, "llm_min_interval_seconds", 0.2)
    monkeypatch.setattr(llm_service, "_pace_lock", None)
    monkeypatch.setattr(llm_service, "_last_call_at", 0.0)
    start = time.monotonic()
    for _ in range(3):
        await llm_service.generate("s", "q")
    assert time.monotonic() - start >= 0.4  # 3 calls, 2 gaps of 0.2 s


async def test_pacing_off_by_default(fake_llm):
    fake_llm("ok")
    assert llm_service.settings.llm_min_interval_seconds == 0
    start = time.monotonic()
    for _ in range(3):
        await llm_service.generate("s", "q")
    assert time.monotonic() - start < 0.2


def test_pacing_works_across_event_loops(fake_llm, monkeypatch):
    """An asyncio.Lock is bound to one loop: a second asyncio.run must get a fresh lock."""
    import asyncio

    fake_llm("ok")
    monkeypatch.setattr(llm_service.settings, "llm_min_interval_seconds", 0.01)
    monkeypatch.setattr(llm_service, "_pace_lock", None)
    monkeypatch.setattr(llm_service, "_pace_loop", None)

    async def two_concurrent_calls():  # contention makes the lock bind to the loop
        await asyncio.gather(llm_service.generate("s", "q"), llm_service.generate("s", "q"))

    asyncio.run(two_concurrent_calls())
    asyncio.run(two_concurrent_calls())  # raised "bound to a different event loop" before
