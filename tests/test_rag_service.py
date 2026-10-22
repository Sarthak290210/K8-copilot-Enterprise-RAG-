import pytest

from app.models import RetrievedChunk
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

    async def hybrid_search(question, vector, top_k=20):
        state["calls"].append(("hybrid", top_k))
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
    assert response.metadata.flags_used == {
        "search_mode": "dense",
        "top_k": 5,
        "enable_rerank": False,
    }

    system_prompt, user_message = fakes["llm_calls"][0]
    assert "Answer ONLY from the provided context" in system_prompt
    assert "[1] (source: pods.html)\nPods are the smallest units." in user_message
    assert user_message.endswith("Question: What is a Pod?")


async def test_top_k_flag_is_used(fakes):
    await rag_service.run_rag("q", {"top_k": 3})
    assert fakes["search_top_k"] == 3


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
