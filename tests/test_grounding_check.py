"""L10: claims checked against the chunks, policy sets a real confidence (fake grader)."""

import json

import pytest

from app.config import settings
from app.models import ChatResponse, GroundingResult, RetrievedChunk
from app.security import grounding_check as gc

CHUNKS = [
    RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=1),
    RetrievedChunk(text="Pods share a network namespace.", source="pods.html", score=1),
]


@pytest.fixture
def grader(monkeypatch):
    replies = []
    calls = []

    async def generate(system, user, model=None, temperature=0.0, json_mode=False):
        calls.append(user)
        content = replies.pop(0)
        return {"content": content, "model": model, "usage": {"total_tokens": 40}}

    monkeypatch.setattr(gc.llm_service, "generate", generate)
    return replies, calls


def verdict(*supported):
    return json.dumps({"claims": [{"claim": f"c{i}", "supported": s}
                                  for i, s in enumerate(supported)]})  # fmt: skip


async def test_score_is_the_supported_fraction(grader):
    replies, calls = grader
    replies.append(verdict(True, True, False, True))
    result = await gc.check_grounding("A Pod is the smallest unit [1] and [7].", CHUNKS)
    assert result.score == 0.75 and result.claims == 4 and result.unsupported_claims == ["c2"]
    assert result.invalid_citations == [7] and result.tokens_used == 40
    # the grader sees spotlighted documents with their sources, as the answer model does, so
    # "it's documented in pods.html" can be checked (q-008)
    assert "<<DOCUMENT id=1 source=pods.html>>" in calls[0]
    assert "id and source" in gc.GROUNDING_PROMPT


def test_grounding_is_strict():
    """User's rule: only stated claims count; implied / well-known / extra advice doesn't."""
    prompt = gc.GROUNDING_PROMPT
    assert "supported only if the documents actually state it" in prompt
    assert "only implied or inferred" in prompt and "general" in prompt
    assert "even when it is true" in prompt
    assert "directly imply" not in prompt  # the old lenient rule is gone


def test_grounding_checks_summaries_and_strong_words():
    """User's rule (2026-10-28, q-014): a concluding "this guarantees ..." is a claim too."""
    prompt = gc.GROUNDING_PROMPT
    assert "summary and concluding sentences" in prompt
    assert "guarantee" in prompt and "claims of their own" in prompt


async def test_bad_reply_is_retried_then_unavailable(grader):
    replies, _ = grader
    replies.extend(["oops", "still oops", "nope"])
    result = await gc.check_grounding("A Pod [1]", CHUNKS)
    assert result.action == "unavailable" and result.score is None
    assert result.tokens_used == 120  # the failed replies were billed too


async def test_no_claims_is_skipped(grader):
    grader[0].append('{"claims": []}')
    assert (await gc.check_grounding("Hello!", CHUNKS)).action == "skipped"


@pytest.mark.parametrize(
    ("score", "action", "answer"),
    # warn never touches the answer text (E6: Ragas scored an appended note as a claim)
    [(0.9, "pass", "A Pod [1]"), (0.6, "warn", "A Pod [1]"),
     (0.2, "replaced", gc.UNVERIFIED_ANSWER)],
)  # fmt: skip
def test_policy(score, action, answer):
    response = ChatResponse(answer="A Pod [1]", confidence=0.99)
    gc.apply_grounding_policy(response, GroundingResult(score=score, claims=5))
    assert response.answer == answer and response.confidence == score
    assert response.metadata.grounding is not None
    assert response.metadata.grounding.action == action


def test_unavailable_keeps_answer_and_confidence():
    response = ChatResponse(answer="A Pod [1]", confidence=0.7)
    gc.apply_grounding_policy(response, GroundingResult(action="unavailable"))
    assert (response.answer, response.confidence) == ("A Pod [1]", 0.7)


def test_what_needs_a_check(monkeypatch):
    monkeypatch.setattr(settings, "grounding_enabled", True)
    assert gc.needs_check("A Pod is ... [1]", CHUNKS, ())
    assert not gc.needs_check("I don’t know.", CHUNKS, ())  # a decline has no claims
    assert not gc.needs_check("Not found.", CHUNKS, ("Not found.",))  # fixed reply
    assert not gc.needs_check("A Pod [1]", [], ())
    monkeypatch.setattr(settings, "grounding_enabled", False)
    assert not gc.needs_check("A Pod is ... [1]", CHUNKS, ())


# ---- one rewrite below the pass score (after E6) ----


@pytest.fixture
def scores(monkeypatch):
    results = []

    async def check(answer, chunks, cited_documents=None):
        score, unsupported = results.pop(0)
        return GroundingResult(score=score, claims=4, unsupported_claims=unsupported,
                               tokens_used=40)  # fmt: skip

    monkeypatch.setattr(gc, "check_grounding", check)
    return results


async def test_well_grounded_answer_is_not_rewritten(scores):
    scores.append((0.9, []))

    async def rewrite(unsupported):
        raise AssertionError("no rewrite above the pass score")

    answer, result = await gc.ground_with_retry("A Pod [1]", CHUNKS, rewrite)
    assert answer == "A Pod [1]" and not result.regenerated and result.tokens_used == 40


async def test_weak_answer_is_rewritten_with_its_unsupported_claims(scores):
    scores.extend([(0.5, ["made up"]), (1.0, [])])
    asked = []

    async def rewrite(unsupported):
        asked.append(unsupported)
        return "A Pod is the smallest deployable unit [1].", 100

    answer, result = await gc.ground_with_retry("A Pod was invented in 1999 [1]", CHUNKS, rewrite)
    assert asked == [["made up"]] and answer == "A Pod is the smallest deployable unit [1]."
    assert result.regenerated and result.score == 1.0 and result.first_score == 0.5
    assert result.tokens_used == 180  # check + rewrite + check


def test_pass_score_default_is_0_9():
    """User's choice (2026-10-28): was 0.8, so one unsupported claim in six passed (q-014)."""
    assert type(settings).model_fields["grounding_pass_score"].default == 0.9


async def test_one_unsupported_claim_in_six_is_rewritten(scores, monkeypatch):
    monkeypatch.setattr(settings, "grounding_pass_score", 0.9)
    scores.extend([(5 / 6, ["a quota guarantees reserved CPU"]), (1.0, [])])

    async def rewrite(unsupported):
        return "A ResourceQuota caps usage [1].", 100

    answer, result = await gc.ground_with_retry("... guarantees CPU [1]", CHUNKS, rewrite)
    assert result.regenerated and answer == "A ResourceQuota caps usage [1]."


async def test_a_worse_rewrite_is_discarded(scores):
    scores.extend([(0.6, ["x"]), (0.4, ["x", "y"])])

    async def rewrite(unsupported):
        return "worse", 100

    answer, result = await gc.ground_with_retry("first [1]", CHUNKS, rewrite)
    assert answer == "first [1]" and not result.regenerated and result.score == 0.6
    assert result.tokens_used == 180 and result.first_score == 0.6


def test_feedback_lists_the_unsupported_statements():
    text = gc.regeneration_feedback(["Pods were added in 1999"])
    assert "- Pods were added in 1999" in text and "NOT supported" in text
