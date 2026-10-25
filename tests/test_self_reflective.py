"""Self-RAG reflection with a mocked grader."""

import pytest

from app.config import settings
from app.models import ReflectionResult, RetrievedChunk
from app.services import self_reflective
from app.services.llm_service import LLMJSONError

CHUNKS = [RetrievedChunk(text="Pods are...", source="pods.html", score=0.9)]


@pytest.fixture
def grader(monkeypatch):
    state = {"data": None, "error": None, "user": None, "system": None}

    async def generate_with_json(system, user, model=None, temperature=0.0):
        state["system"], state["user"] = system, user
        if state["error"]:
            raise state["error"]
        return {
            "data": state["data"],
            "model": model,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 300},
        }

    monkeypatch.setattr(self_reflective.llm_service, "generate_with_json", generate_with_json)
    return state


async def test_scores_four_criteria_and_averages_them(grader):
    grader["data"] = {
        "relevance": 0.2,
        "accuracy": 1.0,
        "completeness": 0.4,
        "clarity": 0.6,
        "issues": ["does not answer"],
        "refined_question": " What is a Kubernetes Pod? ",
    }
    result = await self_reflective.reflect_on_answer("pods?", "I don't know.", CHUNKS)
    assert (result.relevance, result.accuracy, result.completeness, result.clarity) == (
        0.2,
        1.0,
        0.4,
        0.6,
    )
    assert result.score == 0.55  # the mean of the four
    assert result.issues == ["does not answer"]
    assert result.refined_question == "What is a Kubernetes Pod?"
    assert result.tokens_used == 300
    assert "User's question: pods?" in grader["user"] and "[1] Pods are..." in grader["user"]
    assert str(settings.reflection_min_score) in grader["system"]  # the bar is in the prompt


async def test_grader_sees_whole_chunks(grader):
    """The answer was written from whole chunks; cutting them would hide facts it cites."""
    grader["data"] = {}
    long = RetrievedChunk(text="x" * 5000, source="a", score=0.5)
    await self_reflective.reflect_on_answer("q", "a", [long])
    assert "x" * 5000 in grader["user"]


async def test_refined_question_is_shown_to_the_grader(grader):
    grader["data"] = {}
    await self_reflective.reflect_on_answer("explain", "a", CHUNKS, "What is a Pod?")
    assert "User's question: explain" in grader["user"]
    assert "Clarified question (used for this answer): What is a Pod?" in grader["user"]


async def test_grader_error_fails_closed(grader):
    grader["error"] = RuntimeError("down")
    result = await self_reflective.reflect_on_answer("q", "a", CHUNKS)
    assert result.score == 0.0 and result.issues == ["reflection unavailable"]
    assert result.tokens_used == 0


async def test_bad_json_fails_closed_but_counts_tokens(grader):
    grader["error"] = LLMJSONError("bad", {"total_tokens": 120})
    result = await self_reflective.reflect_on_answer("q", "a", CHUNKS)
    assert result.score == 0.0 and result.tokens_used == 120


@pytest.mark.parametrize(
    "data", [{"relevance": "bad"}, {"relevance": 7, "clarity": -2}, {}, {"refined_question": " "}]
)
async def test_odd_values_are_clamped(grader, data):
    grader["data"] = data
    result = await self_reflective.reflect_on_answer("q", "a", CHUNKS)
    assert 0.0 <= result.score <= 1.0
    assert all(0.0 <= v <= 1.0 for v in (result.relevance, result.clarity))
    assert result.refined_question is None


async def test_refine_question_fallback(grader):
    grader["data"] = {"refined_question": "How do I debug a crashing Pod?"}
    reflection = ReflectionResult(score=0.3, issues=["vague"])
    refined, tokens = await self_reflective.refine_question("what is wrong", "?", reflection)
    assert refined == "How do I debug a crashing Pod?" and tokens == 300
    assert "Problems: vague" in grader["user"]


async def test_refine_question_failure_means_no_retry(grader):
    grader["error"] = LLMJSONError("bad", {"total_tokens": 40})
    refined, tokens = await self_reflective.refine_question("q", "a", ReflectionResult())
    assert refined is None and tokens == 40


def test_retry_below_the_bar_until_max_retries():
    low = ReflectionResult(score=0.79)
    good = ReflectionResult(score=0.8)
    assert settings.reflection_min_score == 0.8 and settings.max_reflection_retries == 2
    assert self_reflective.should_regenerate(low, 0)
    assert self_reflective.should_regenerate(low, 1)
    assert not self_reflective.should_regenerate(low, 2)  # 2 retries already made
    assert not self_reflective.should_regenerate(good, 0)  # reaches the bar: show it
