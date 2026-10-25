"""CRAG with a mocked grader and web search (no Cerebras, no Tavily calls)."""

import pytest

from app.models import CRAGEvaluation, RetrievedChunk
from app.services import crag
from app.services.llm_service import LLMJSONError


def chunk(source, index=0):
    return RetrievedChunk(
        text=f"text {source}", source=source, score=0.5, metadata={"chunk_index": index}
    )


WEB = [
    RetrievedChunk(
        text="web page", source="https://example.com", score=0.9, metadata={"origin": "web"}
    )
]


@pytest.fixture
def fakes(monkeypatch):
    state = {
        "scores": None,
        "related": True,  # the grader's kubernetes_related verdict (None = left out)
        "vague": False,  # the grader's too_vague verdict (None = left out)
        "grader_error": None,
        "web": WEB,
        "web_calls": 0,
        "user": None,
    }

    async def generate_with_json(system, user, model=None, temperature=0.0):
        state["user"] = user
        if state["grader_error"]:
            raise state["grader_error"]
        data = {"scores": state["scores"], "reason": "because"}
        if state["related"] is not None:
            data["kubernetes_related"] = state["related"]
        if state["vague"] is not None:
            data["too_vague"] = state["vague"]
        return {
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 200},
        }

    async def search_web(query, max_results=5):
        state["web_calls"] += 1
        return state["web"]

    monkeypatch.setattr(crag.llm_service, "generate_with_json", generate_with_json)
    monkeypatch.setattr(crag, "search_web", search_web)
    return state


@pytest.mark.parametrize(
    ("max_score", "action"),
    [
        (0.95, "correct"),
        (0.7, "correct"),
        (0.69, "ambiguous"),
        (0.5, "ambiguous"),
        (0.49, "incorrect"),
    ],
)
def test_threshold_mapping(max_score, action):
    assert crag._action(max_score) == action


def test_should_trigger_web_search_only_for_in_scope_questions():
    assert not crag.should_trigger_web_search(CRAGEvaluation(action="correct", in_scope=True))
    assert crag.should_trigger_web_search(CRAGEvaluation(action="ambiguous", in_scope=True))
    assert crag.should_trigger_web_search(CRAGEvaluation(action="incorrect", in_scope=True))
    assert not crag.should_trigger_web_search(CRAGEvaluation(action="incorrect", in_scope=False))
    assert not crag.should_trigger_web_search(CRAGEvaluation(action="incorrect", in_scope=None))
    assert not crag.should_trigger_web_search(CRAGEvaluation(action="ungraded", in_scope=True))


def test_clean_scores_clamps_and_pads():
    assert crag._clean_scores([1.5, "0.3", -1], 4) == [1.0, 0.3, 0.0, 0.0]
    assert crag._clean_scores("bad", 2) == [0.0, 0.0]


async def test_correct_keeps_only_relevant_chunks_no_web(fakes):
    fakes["scores"] = [0.9, 0.2, 0.6]
    chunks, evaluation = await crag.crag_pipeline("q", [chunk("a"), chunk("b"), chunk("c")])
    assert evaluation.action == "correct" and evaluation.max_score == 0.9
    assert [c.source for c in chunks] == ["a", "c"]
    assert chunks[0].metadata["crag_score"] == 0.9
    assert fakes["web_calls"] == 0 and not evaluation.web_used
    assert evaluation.tokens_used == 200 and evaluation.in_scope is True


async def test_correct_answers_from_docs_even_if_off_topic(fakes):
    """If our docs cover it, scope doesn't matter: nothing goes to the web anyway."""
    fakes["scores"], fakes["related"] = [0.9], False
    chunks, evaluation = await crag.crag_pipeline("q", [chunk("a")])
    assert [c.source for c in chunks] == ["a"] and fakes["web_calls"] == 0


async def test_ambiguous_adds_web_results(fakes):
    fakes["scores"] = [0.6, 0.1]
    chunks, evaluation = await crag.crag_pipeline("q", [chunk("a"), chunk("b")])
    assert evaluation.action == "ambiguous" and evaluation.web_used
    assert [c.source for c in chunks] == ["a", "https://example.com"]


async def test_incorrect_uses_web_only(fakes):
    fakes["scores"] = [0.1, 0.0]
    chunks, evaluation = await crag.crag_pipeline("new k8s release?", [chunk("a"), chunk("b")])
    assert evaluation.action == "incorrect"
    assert [c.source for c in chunks] == ["https://example.com"]


async def test_off_topic_incorrect_skips_the_web_and_returns_nothing(fakes):
    """User's rule: only Kubernetes / IT-ops questions may go to the web."""
    fakes["scores"], fakes["related"] = [0.1], False
    chunks, evaluation = await crag.crag_pipeline("write me a poem", [chunk("noise.pdf")])
    assert chunks == [] and fakes["web_calls"] == 0
    assert evaluation.in_scope is False and not evaluation.web_used
    assert "off-topic" in evaluation.reason


async def test_off_topic_ambiguous_keeps_relevant_chunks_without_web(fakes):
    fakes["scores"], fakes["related"] = [0.6, 0.1], False
    chunks, evaluation = await crag.crag_pipeline("q", [chunk("a"), chunk("b")])
    assert [c.source for c in chunks] == ["a"] and fakes["web_calls"] == 0
    assert "web search skipped" in evaluation.reason


async def test_unknown_scope_keeps_original_chunks_without_web(fakes):
    fakes["scores"], fakes["related"] = [0.1], None  # grader left kubernetes_related out
    original = [chunk("a")]
    chunks, evaluation = await crag.crag_pipeline("q", original)
    assert chunks == original and fakes["web_calls"] == 0
    assert evaluation.in_scope is None and "scope unknown" in evaluation.reason


async def test_no_web_results_keeps_original_chunks(fakes):
    """Fix vs reference: never answer from an empty context because Tavily is off."""
    fakes["scores"] = [0.1]
    fakes["web"] = []
    original = [chunk("a")]
    chunks, evaluation = await crag.crag_pipeline("q", original)
    assert chunks == original
    assert evaluation.action == "incorrect" and not evaluation.web_used
    assert "web search unavailable" in evaluation.reason


async def test_grader_failure_keeps_chunks_ungraded(fakes):
    fakes["grader_error"] = RuntimeError("down")
    original = [chunk("a")]
    chunks, evaluation = await crag.crag_pipeline("q", original)
    assert chunks == original
    assert evaluation.action == "ungraded" and evaluation.max_score is None
    assert fakes["web_calls"] == 0 and evaluation.tokens_used == 0


async def test_bad_json_still_counts_its_tokens(fakes):
    fakes["grader_error"] = LLMJSONError("bad", {"total_tokens": 150})
    _, evaluation = await crag.crag_pipeline("q", [chunk("a")])
    assert evaluation.action == "ungraded" and evaluation.tokens_used == 150


async def test_no_chunks_still_asks_the_grader_for_the_scope(fakes):
    fakes["scores"] = []
    evaluation = await crag.grade_chunks("k8s 1.40 features?", [])
    assert evaluation.action == "incorrect" and evaluation.max_score == 0.0
    assert evaluation.in_scope is True and "(no passages found)" in fakes["user"]


async def test_grader_sees_cut_passages(fakes):
    fakes["scores"] = [0.9]
    long = RetrievedChunk(text="x" * 5000, source="a", score=0.5)
    await crag.grade_chunks("q", [long])
    assert "x" * crag.MAX_CHARS_PER_CHUNK in fakes["user"]
    assert "x" * (crag.MAX_CHARS_PER_CHUNK + 1) not in fakes["user"]


async def test_too_vague_incorrect_asks_for_detail_without_web(fakes):
    """E3: a web search for "explain" returned dictionary definitions."""
    fakes["scores"], fakes["vague"] = [0.1], True
    chunks, evaluation = await crag.crag_pipeline("what is wrong", [chunk("noise.pdf")])
    assert chunks == [] and fakes["web_calls"] == 0
    assert evaluation.too_vague is True and "too vague" in evaluation.reason


async def test_too_vague_ambiguous_keeps_relevant_chunks_without_web(fakes):
    fakes["scores"], fakes["vague"] = [0.6, 0.1], True
    chunks, _ = await crag.crag_pipeline("explain", [chunk("a"), chunk("b")])
    assert [c.source for c in chunks] == ["a"] and fakes["web_calls"] == 0


async def test_too_vague_but_docs_answer_it_is_kept(fakes):
    fakes["scores"], fakes["vague"] = [0.9], True
    chunks, _ = await crag.crag_pipeline("explain", [chunk("a")])
    assert [c.source for c in chunks] == ["a"]


def test_vague_questions_never_trigger_web_search():
    vague = CRAGEvaluation(action="incorrect", in_scope=True, too_vague=True)
    assert not crag.should_trigger_web_search(vague)
