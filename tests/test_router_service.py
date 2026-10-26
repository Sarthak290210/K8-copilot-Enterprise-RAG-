"""Intent routing with a mocked LLM."""

import pytest

from app.services import router_service
from app.services.llm_service import LLMJSONError


@pytest.fixture
def llm(monkeypatch):
    state = {"data": None, "error": None}

    async def generate_with_json(system, user, model=None, temperature=0.0):
        if state["error"]:
            raise state["error"]
        return {"data": state["data"], "model": model, "usage": {"total_tokens": 42}}

    monkeypatch.setattr(router_service.llm_service, "generate_with_json", generate_with_json)
    return state


@pytest.mark.parametrize("intent", ["sql", "rag", "hybrid"])
async def test_valid_intents(llm, intent):
    llm["data"] = {"intent": f" {intent.upper()} "}
    assert await router_service.classify_intent("q") == (intent, 42)


@pytest.mark.parametrize("data", [{"intent": "database"}, {}, {"intent": None}])
async def test_unknown_values_default_to_rag(llm, data):
    llm["data"] = data
    assert await router_service.classify_intent("q") == ("rag", 42)


async def test_invalid_json_defaults_to_rag_and_counts_tokens(llm):
    llm["error"] = LLMJSONError("bad", {"total_tokens": 9})
    assert await router_service.classify_intent("q") == ("rag", 9)


async def test_errors_default_to_rag(llm):
    llm["error"] = RuntimeError("down")
    assert await router_service.classify_intent("q") == ("rag", 0)


async def test_intent_is_cached_per_question(llm, monkeypatch, cache_on):
    calls = []
    real = router_service.llm_service.generate_with_json

    async def counting(*args, **kwargs):
        calls.append(1)
        return await real(*args, **kwargs)

    monkeypatch.setattr(router_service.llm_service, "generate_with_json", counting)
    llm["data"] = {"intent": "sql"}
    assert await router_service.classify_intent("How many pods?") == ("sql", 42)
    assert await router_service.classify_intent("how many  pods?") == ("sql", 0)  # 0 tokens
    assert len(calls) == 1


async def test_fallbacks_are_not_cached(llm, cache_on):
    llm["data"] = {"intent": "weather"}
    assert (await router_service.classify_intent("q"))[0] == "rag"
    llm["data"] = {"intent": "sql"}
    assert await router_service.classify_intent("q") == ("sql", 42)
