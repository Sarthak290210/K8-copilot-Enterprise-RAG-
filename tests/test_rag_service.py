import pytest

from app.models import CRAGEvaluation, ReflectionResult, RetrievedChunk
from app.services import rag_service
from eval.profiles import get_profile


@pytest.fixture
def fakes(monkeypatch):
    """Fake embedding, vector store and LLM; records what the pipeline sent."""
    state = {"chunks": [], "llm_calls": [], "search_top_k": None, "calls": [], "rerank": None}

    async def embed_query(question):
        state["calls"].append("embed")
        return [0.1, 0.2]

    async def search(vector, top_k=5):
        state["search_top_k"] = top_k
        state["calls"].append(("dense", top_k))
        return state["chunks"]

    async def sparse_search(question, top_k=20):
        state["calls"].append(("sparse", top_k))
        return state["chunks"]

    async def hybrid_search(question, vector, top_k=20, dense_search=None):
        state["calls"].append(("hybrid", top_k))
        if dense_search is not None:  # HyDE plugged in as the dense half
            return [*(await dense_search(20)), *state["chunks"]]
        return state["chunks"]

    class FakeReranker:
        def rerank(self, question, chunks, top_k=5):
            state["rerank"] = (question, len(chunks), top_k)
            return list(reversed(chunks))[:top_k]

    async def generate(system_prompt, user_message, **kwargs):
        state["llm_calls"].append((system_prompt, user_message))
        return {
            "content": "A Pod is ... [1]",
            "model": "test-model",
            "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
        }

    monkeypatch.setattr(rag_service.embedding_service, "embed_query", embed_query)
    monkeypatch.setattr(rag_service.vector_store, "search", search)
    monkeypatch.setattr(rag_service.vector_store, "sparse_search", sparse_search)
    monkeypatch.setattr(rag_service.vector_store, "hybrid_search", hybrid_search)
    monkeypatch.setattr(rag_service, "get_reranker", FakeReranker)
    monkeypatch.setattr(rag_service.llm_service, "generate", generate)
    # Pipeline-mechanics tests start from the plain dense path; the shipped defaults
    # (hybrid + CRAG) are checked in test_default_flags_*.
    monkeypatch.setattr(rag_service, "DEFAULT_FLAGS", get_profile("naive"))
    return state


def chunk(source, text, score=0.8):
    return RetrievedChunk(text=text, source=source, score=score, metadata={"chunk_index": 0})


async def test_no_chunks_means_no_llm_call(fakes):
    response = await rag_service.run_rag("What is a Pod?")
    assert response.answer == rag_service.NOT_FOUND_ANSWER
    assert fakes["llm_calls"] == []
    assert response.sources == [] and response.metadata.tokens_used == 0


async def test_answer_with_sources_and_trace(fakes):
    fakes["chunks"] = [
        chunk("pods.html", "Pods are the smallest units.", 0.9),
        chunk("pod-v1.docx", "PodSpec fields.", 0.8),
        chunk("pods.html", "Pods share a network namespace.", 0.7),
    ]
    response, traced = await rag_service.run_rag_with_trace("What is a Pod?")
    assert response.answer == "A Pod is ... [1]"
    assert response.sources == ["pods.html", "pod-v1.docx"]  # deduped, retrieval order
    assert [p.source for p in response.chunks] == ["pods.html", "pod-v1.docx", "pods.html"]
    assert traced == fakes["chunks"]  # eval gets the full chunks
    assert response.metadata.tokens_used == 100
    assert response.metadata.flags_used == get_profile("naive")
    assert response.confidence is None  # no grader ran
    assert response.metadata.reflection_iterations == 0 and response.metadata.crag is None

    system_prompt, user_message = fakes["llm_calls"][0]
    assert "Answer ONLY from the provided context" in system_prompt
    assert "[1] (source: pods.html)\nPods are the smallest units." in user_message
    assert user_message.endswith("Question: What is a Pod?")


async def test_top_k_flag_is_used(fakes):
    await rag_service.run_rag("q", {"top_k": 3})
    assert fakes["search_top_k"] == 3


def test_default_flags_are_hybrid_plus_crag():
    """User's choice after E3: hybrid + CRAG on; HyDE, rerank and Self-RAG off."""
    assert rag_service._default_flags() == {
        "search_mode": "hybrid",
        "top_k": 5,
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": True,
        "enable_self_reflective": False,
    }


def test_default_flags_follow_settings(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(rag_service.settings, "crag_enabled_by_default", False)
    monkeypatch.setattr(rag_service.settings, "hyde_enabled_by_default", True)
    defaults = rag_service._default_flags()
    assert defaults["search_mode"] == "dense"
    assert not defaults["enable_crag"] and defaults["enable_hyde"]


async def test_request_can_turn_crag_off(fakes, monkeypatch):
    async def fail_crag(question, chunks):
        raise AssertionError("CRAG should not run")

    monkeypatch.setattr(rag_service, "DEFAULT_FLAGS", rag_service._default_flags())
    monkeypatch.setattr(rag_service, "crag_pipeline", fail_crag)
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    response = await rag_service.run_rag("q", {"enable_crag": False})
    assert fakes["calls"] == ["embed", ("hybrid", 5)]  # still the hybrid default
    assert response.metadata.flags_used["enable_crag"] is False


@pytest.mark.parametrize("flags", [None, "not-a-dict", {}])
def test_flag_falls_back_to_default(flags):
    assert rag_service._flag(flags, "top_k") == 5  # type: ignore[arg-type]


async def test_unknown_search_mode(fakes):
    with pytest.raises(ValueError, match="Unknown search_mode 'semantic'"):
        await rag_service.run_rag("q", {"search_mode": "semantic"})


async def test_naive_profile_is_exactly_the_phase1_path(fakes):
    """E1 measured dense top-5 with no rerank: the naive flags must still do only that."""
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    await rag_service.run_rag("q", get_profile("naive"))
    assert fakes["calls"] == ["embed", ("dense", 5)]
    assert fakes["rerank"] is None


@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        ("sparse", [("sparse", 5)]),  # BM25 needs no embedding
        ("hybrid", ["embed", ("hybrid", 5)]),
    ],
)
async def test_search_modes(fakes, mode, expected_calls):
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    response = await rag_service.run_rag("q", {"search_mode": mode})
    assert fakes["calls"] == expected_calls
    assert response.metadata.flags_used["search_mode"] == mode


async def test_rerank_fetches_candidates_then_keeps_top_k(fakes):
    fakes["chunks"] = [chunk(f"doc{i}.txt", f"text {i}") for i in range(20)]
    response, traced = await rag_service.run_rag_with_trace(
        "q", {"search_mode": "hybrid", "enable_rerank": True, "top_k": 3}
    )
    assert fakes["calls"] == ["embed", ("hybrid", 20)]  # reranker_initial_top_k candidates
    assert fakes["rerank"] == ("q", 20, 3)
    assert [c.source for c in traced] == ["doc19.txt", "doc18.txt", "doc17.txt"]
    assert response.metadata.flags_used["enable_rerank"] is True


# ---- flag validation (code review: top_k bounds, real booleans) ----


@pytest.mark.parametrize(
    "flags",
    [{"top_k": 0}, {"top_k": -1}, {"top_k": 100000}, {"enable_crag": "maybe"}, {"unknown": 1}],
)
async def test_bad_flags_are_rejected(fakes, flags):
    with pytest.raises(ValueError, match="Invalid flags"):
        await rag_service.run_rag("q", flags)
    assert fakes["llm_calls"] == []


async def test_string_false_means_off(fakes, monkeypatch):
    async def fail_crag(question, chunks):
        raise AssertionError("CRAG should not run")

    monkeypatch.setattr(rag_service, "crag_pipeline", fail_crag)
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    response = await rag_service.run_rag("q", {"enable_crag": "false", "top_k": "3"})
    assert response.metadata.flags_used["enable_crag"] is False
    assert fakes["search_top_k"] == 3


def test_flags_must_be_an_object():
    with pytest.raises(ValueError, match="JSON object"):
        rag_service.validate_flags(["top_k", 5])


# ---- Phase 3: HyDE, CRAG, Self-RAG wiring ----


async def test_hyde_is_the_dense_half_of_hybrid_search(fakes, monkeypatch):
    calls = []

    class FakeHyDE:
        def __init__(self):
            self.hypotheses, self.tokens_used = ["fake passage"], 120

        async def retrieve(self, question, top_k=5):
            calls.append(("hyde", top_k))
            return [chunk("hyde.txt", "from hyde")]

    monkeypatch.setattr(rag_service, "HyDERetriever", FakeHyDE)
    fakes["chunks"] = [chunk("bm25.txt", "from bm25")]
    response, traced = await rag_service.run_rag_with_trace(
        "q", {"search_mode": "hybrid", "enable_hyde": True}
    )
    assert calls == [("hyde", 20)]  # hybrid_search's depth
    assert fakes["calls"] == [("hybrid", 5)]  # one hybrid_search; no plain embedding
    assert {c.source for c in traced} == {"hyde.txt", "bm25.txt"}
    assert response.metadata.hyde_hypotheses == ["fake passage"]
    assert response.metadata.tokens_used == 120 + 100  # HyDE call + answer


async def test_crag_corrects_the_context_and_sets_confidence(fakes, monkeypatch):
    web = RetrievedChunk(text="web", source="https://x.io", score=0.9, metadata={"origin": "web"})

    async def fake_crag(question, chunks):
        return [web], CRAGEvaluation(
            scores=[0.1],
            max_score=0.1,
            action="incorrect",
            web_used=True,
            in_scope=True,
            tokens_used=50,
        )

    monkeypatch.setattr(rag_service, "crag_pipeline", fake_crag)
    fakes["chunks"] = [chunk("noise.pdf", "unrelated")]
    response = await rag_service.run_rag(
        "k8s 1.40?", {"search_mode": "hybrid", "enable_crag": True}
    )
    assert response.sources == ["https://x.io"]
    assert "[1] (source: https://x.io)" in fakes["llm_calls"][0][1]
    assert response.metadata.crag is not None and response.metadata.crag.action == "incorrect"
    assert response.confidence is None  # web results replaced the graded chunks
    assert response.metadata.tokens_used == 150


async def test_off_topic_question_is_declined_without_an_answer_call(fakes, monkeypatch):
    """User's rule: CRAG answers only Kubernetes / IT-ops questions from the web."""

    async def fake_crag(question, chunks):
        return [], CRAGEvaluation(
            scores=[0.0], max_score=0.0, action="incorrect", in_scope=False, tokens_used=50
        )

    monkeypatch.setattr(rag_service, "crag_pipeline", fake_crag)
    fakes["chunks"] = [chunk("noise.pdf", "unrelated")]
    response = await rag_service.run_rag("write me a poem", {"enable_crag": True})
    assert response.answer == rag_service.OUT_OF_SCOPE_ANSWER
    assert fakes["llm_calls"] == [] and response.sources == []
    assert response.confidence is None and response.metadata.tokens_used == 50


async def test_vague_question_gets_a_request_for_detail(fakes, monkeypatch):
    async def fake_crag(question, chunks):
        return [], CRAGEvaluation(
            scores=[0.0], max_score=0.0, action="incorrect", in_scope=True, too_vague=True
        )

    monkeypatch.setattr(rag_service, "crag_pipeline", fake_crag)
    fakes["chunks"] = [chunk("noise.pdf", "unrelated")]
    response = await rag_service.run_rag("what is wrong", {"enable_crag": True})
    assert response.answer == rag_service.CLARIFY_ANSWER and fakes["llm_calls"] == []


async def test_crag_without_web_uses_its_score_as_confidence(fakes, monkeypatch):
    async def fake_crag(question, chunks):
        return chunks, CRAGEvaluation(scores=[0.8], max_score=0.8, action="correct")

    monkeypatch.setattr(rag_service, "crag_pipeline", fake_crag)
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    response = await rag_service.run_rag("q", {"enable_crag": True})
    assert response.confidence == 0.8


# ---- Self-RAG: 4 criteria, < 0.8 -> refine the question, answer again (user's design) ----


@pytest.fixture
def reflections(fakes, monkeypatch):
    """Scripted grader verdicts; records what each reflection and retrieval saw."""
    state = {"verdicts": [], "reflected": [], "retrieved": [], "refines": []}

    async def reflect(question, answer, chunks, refined_question=None):
        state["reflected"].append((question, refined_question))
        return state["verdicts"].pop(0)

    async def refine(question, answer, reflection):
        state["refines"].append(question)
        return state.get("fallback"), 7

    real_retrieve = rag_service._retrieve

    async def spy_retrieve(question, flags, trace=None):
        state["retrieved"].append(question)
        return await real_retrieve(question, flags, trace)

    monkeypatch.setattr(rag_service, "reflect_on_answer", reflect)
    monkeypatch.setattr(rag_service, "refine_question", refine)
    monkeypatch.setattr(rag_service, "_retrieve", spy_retrieve)
    fakes["chunks"] = [chunk("pods.html", "Pods.")]
    return state


SELF_RAG = {"enable_self_reflective": True}


async def test_good_first_answer_is_shown_without_retry(fakes, reflections):
    reflections["verdicts"] = [ReflectionResult(score=0.8, tokens_used=10)]
    response = await rag_service.run_rag("What is a Pod?", SELF_RAG)
    assert len(fakes["llm_calls"]) == 1 and response.metadata.reflection_iterations == 0
    assert response.confidence == 0.8 and response.metadata.refined_question is None


async def test_low_score_refines_and_answers_the_refined_question(fakes, reflections):
    reflections["verdicts"] = [
        ReflectionResult(score=0.4, refined_question="What is a Kubernetes Pod?", tokens_used=10),
        ReflectionResult(score=0.9, tokens_used=10),
    ]
    response = await rag_service.run_rag("tell me about pods", SELF_RAG)
    assert reflections["retrieved"] == ["tell me about pods", "What is a Kubernetes Pod?"]
    # the retry answers the refined question, and is graded against the user's question
    assert fakes["llm_calls"][1][1].endswith("Question: What is a Kubernetes Pod?")
    assert reflections["reflected"] == [
        ("tell me about pods", None),
        ("tell me about pods", "What is a Kubernetes Pod?"),
    ]
    assert response.metadata.reflection_iterations == 1
    assert response.confidence == 0.9
    assert response.metadata.refined_question == "What is a Kubernetes Pod?"
    assert response.metadata.tokens_used == 2 * 100 + 2 * 10


async def test_stops_after_max_retries_and_keeps_the_best(fakes, reflections):
    reflections["verdicts"] = [
        ReflectionResult(score=0.3, refined_question="q2"),
        ReflectionResult(score=0.6, refined_question="q3"),
        ReflectionResult(score=0.5, refined_question="q4"),
    ]
    response = await rag_service.run_rag("q1", SELF_RAG)
    assert reflections["retrieved"] == ["q1", "q2", "q3"]  # 1 answer + 2 retries
    assert response.metadata.reflection_iterations == 2
    assert response.confidence == 0.6 and response.metadata.refined_question == "q2"


async def test_no_refined_question_uses_the_rewrite_fallback(fakes, reflections):
    reflections["fallback"] = "How do I find why a Pod restarts?"
    reflections["verdicts"] = [
        ReflectionResult(score=0.2, issues=["vague"]),
        ReflectionResult(score=0.85),
    ]
    response = await rag_service.run_rag("what is wrong", SELF_RAG)
    assert reflections["refines"] == ["what is wrong"]
    assert reflections["retrieved"][1] == "How do I find why a Pod restarts?"
    assert response.metadata.reflection_iterations == 1 and response.confidence == 0.85
    assert response.metadata.tokens_used == 2 * 100 + 7  # answers + the rewrite call


async def test_stops_when_no_new_question_can_be_made(fakes, reflections):
    reflections["fallback"] = None
    reflections["verdicts"] = [ReflectionResult(score=0.2)]
    response = await rag_service.run_rag("q", SELF_RAG)
    assert len(fakes["llm_calls"]) == 1 and response.metadata.reflection_iterations == 0
    assert response.confidence == 0.2


async def test_stops_when_the_refined_question_is_the_same(fakes, reflections):
    reflections["verdicts"] = [ReflectionResult(score=0.2, refined_question="What Is A POD?")]
    response = await rag_service.run_rag("what is a pod?", SELF_RAG)
    assert len(fakes["llm_calls"]) == 1 and response.metadata.reflection_iterations == 0


async def test_self_rag_empty_retry_keeps_the_earlier_answer(fakes, monkeypatch):
    """A retry that retrieves nothing must not replace a real answer with NOT_FOUND."""
    results = iter([[chunk("pods.html", "Pods.")], []])

    async def retrieve(question, flags, trace=None):
        return next(results)

    async def reflect(question, answer, chunks, refined_question=None):
        return ReflectionResult(score=0.6, refined_question="a clearer question")

    monkeypatch.setattr(rag_service, "_retrieve", retrieve)
    monkeypatch.setattr(rag_service, "reflect_on_answer", reflect)
    response = await rag_service.run_rag("q", SELF_RAG)
    assert response.answer == "A Pod is ... [1]" and response.sources == ["pods.html"]
    assert response.confidence == 0.6 and response.metadata.reflection_iterations == 1
    assert response.metadata.refined_question is None  # attempt 1 used the user's question


async def test_self_rag_metadata_describes_the_returned_attempt(fakes, monkeypatch):
    """CRAG/HyDE metadata must come from the best attempt, not from the last retry."""
    good = CRAGEvaluation(scores=[0.9], max_score=0.9, action="correct")
    web = CRAGEvaluation(scores=[0.1], max_score=0.1, action="incorrect", web_used=True)
    attempts = iter(
        [
            (chunk("pods.html", "Pods."), good, ["passage 1"]),
            (chunk("https://x.io", "web"), web, ["passage 2"]),
        ]
    )
    verdicts = iter(
        [
            ReflectionResult(score=0.7, refined_question="clearer"),
            ReflectionResult(score=0.3, refined_question="clearer still"),
            ReflectionResult(score=0.3),
        ]
    )

    async def retrieve(question, flags, trace):
        found, trace.crag, trace.hypotheses = next(attempts, attempts_done)
        return [found]

    attempts_done = (chunk("https://x.io", "web"), web, ["passage 3"])

    async def reflect(question, answer, chunks, refined_question=None):
        return next(verdicts)

    monkeypatch.setattr(rag_service, "_retrieve", retrieve)
    monkeypatch.setattr(rag_service, "reflect_on_answer", reflect)
    response = await rag_service.run_rag("q", SELF_RAG)
    assert response.sources == ["pods.html"]  # attempt 1 scored best
    assert response.metadata.crag == good and response.metadata.hyde_hypotheses == ["passage 1"]
    assert response.confidence == 0.7 and response.metadata.reflection_iterations == 2


# ---------------- Phase 5: rag_answer cache ----------------


async def test_repeat_question_is_served_from_the_cache(fakes, cache_on):
    fakes["chunks"] = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    first, _ = await rag_service.run_rag_with_trace("What is a Pod?")
    assert first.metadata.cache_hits == {"rag_answer": False}
    second, traced = await rag_service.run_rag_with_trace("  what is a pod? ")
    assert len(fakes["llm_calls"]) == 1  # no second LLM call
    assert second.answer == first.answer and second.sources == first.sources
    assert traced == fakes["chunks"]  # full chunks come back too (eval contexts)
    meta = second.metadata
    assert meta.cache_hits == {"rag_answer": True}
    assert meta.tokens_used == 0 and meta.cost_saved_usd == pytest.approx(100 * 0.5 / 1e6)


async def test_changing_a_flag_misses_the_cache(fakes, cache_on):
    fakes["chunks"] = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    await rag_service.run_rag_with_trace("What is a Pod?")
    response, _ = await rag_service.run_rag_with_trace("What is a Pod?", {"top_k": 3})
    assert response.metadata.cache_hits == {"rag_answer": False}
    assert len(fakes["llm_calls"]) == 2


async def test_answers_without_context_are_not_cached(fakes, cache_on):
    await rag_service.run_rag_with_trace("What is a Pod?")  # no chunks -> NOT_FOUND
    fakes["chunks"] = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    response, _ = await rag_service.run_rag_with_trace("What is a Pod?")
    assert response.answer == "A Pod is ... [1]"  # retried for real, not the cached refusal


def test_model_written_declines_are_not_cached():
    """'I don't know' with chunks (q-009/q-036-style) would otherwise stick for CACHE_TTL_RAG."""
    chunks = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    assert rag_service._cacheable("A Pod is ... [1]", chunks)
    assert not rag_service._cacheable("I don't know.", chunks)
    assert not rag_service._cacheable("I don’t know based on the context.", chunks)
    assert not rag_service._cacheable(rag_service.NOT_FOUND_ANSWER, chunks)
    assert not rag_service._cacheable("A Pod is ... [1]", [])
    assert rag_service._cacheable("If you don't know the pod name, run kubectl get pods.", chunks)


async def test_changing_a_pipeline_setting_misses_the_cache(fakes, cache_on, monkeypatch):
    """CRAG thresholds, RRF_K, reranker... change answers, so they're in the key too."""
    fakes["chunks"] = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    await rag_service.run_rag_with_trace("What is a Pod?")
    monkeypatch.setattr(rag_service.settings, "rrf_k", 30)
    response, _ = await rag_service.run_rag_with_trace("What is a Pod?")
    assert response.metadata.cache_hits == {"rag_answer": False}
    assert len(fakes["llm_calls"]) == 2


async def test_cache_disabled_always_runs_the_pipeline(fakes):
    fakes["chunks"] = [chunk("pods.html", "Pods are the smallest units.", 0.9)]
    await rag_service.run_rag_with_trace("What is a Pod?")
    response, _ = await rag_service.run_rag_with_trace("What is a Pod?")
    assert len(fakes["llm_calls"]) == 2 and response.metadata.cache_hits == {}
