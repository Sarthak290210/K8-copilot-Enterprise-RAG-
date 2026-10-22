import pytest

from app.models import RetrievedChunk
from app.services import rag_service


@pytest.fixture
def fakes(monkeypatch):
    """Fake embedding, vector store and LLM; records what the pipeline sent."""
    state = {"chunks": [], "llm_calls": [], "search_top_k": None}

    async def embed_query(question):
        return [0.1, 0.2]

    async def search(vector, top_k=5):
        state["search_top_k"] = top_k
        return state["chunks"]

    async def generate(system_prompt, user_message, **kwargs):
        state["llm_calls"].append((system_prompt, user_message))
        return {
            "content": "A Pod is ... [1]",
            "model": "test-model",
            "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
        }

    monkeypatch.setattr(rag_service.embedding_service, "embed_query", embed_query)
    monkeypatch.setattr(rag_service.vector_store, "search", search)
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
    assert response.metadata.flags_used == {"search_mode": "dense", "top_k": 5}

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


async def test_unsupported_search_mode(fakes):
    with pytest.raises(ValueError, match="not available yet"):
        await rag_service.run_rag("q", {"search_mode": "hybrid"})
