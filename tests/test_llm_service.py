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
    with pytest.raises(llm_service.LLMJSONError):
        await llm_service.generate_with_json("system", "q")
