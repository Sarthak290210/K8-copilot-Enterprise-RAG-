"""HyDE with a mocked LLM, embeddings and vector store."""

import pytest

from app.models import RetrievedChunk
from app.services import hyde


def chunk(source, text, index=0):
    return RetrievedChunk(text=text, source=source, score=0.5, metadata={"chunk_index": index})


@pytest.fixture
def fakes(monkeypatch):
    state = {"searches": [], "embedded_docs": None, "llm_response": None, "llm_error": None}

    async def generate_with_json(system, user, model=None, temperature=0.0):
        if state["llm_error"]:
            raise state["llm_error"]
        return {
            "data": state["llm_response"],
            "model": model,
            "usage": {"prompt_tokens": 50, "completion_tokens": 70, "total_tokens": 120},
        }

    async def embed_query(question):
        return ["q"]

    async def embed_texts(texts, cache=False):
        state["embedded_docs"] = list(texts)
        return [[f"h{i}"] for i in range(len(texts))]

    results = {
        ("q",): [chunk("a.txt", "Alpha"), chunk("b.txt", "Beta")],
        ("h0",): [chunk("c.txt", "Gamma"), chunk("a.txt", "Alpha")],
        ("h1",): [chunk("a.txt", "Alpha"), chunk("d.txt", "  BETA ", index=1)],
        ("h2",): [chunk("c.txt", "Gamma")],
    }

    async def search(vector, top_k=5):
        state["searches"].append((tuple(vector), top_k))
        return results[tuple(vector)][:top_k]

    monkeypatch.setattr(hyde.llm_service, "generate_with_json", generate_with_json)
    monkeypatch.setattr(hyde.embedding_service, "embed_query", embed_query)
    monkeypatch.setattr(hyde.embedding_service, "embed_texts", embed_texts)
    monkeypatch.setattr(hyde.vector_store, "search", search)
    return state


async def test_three_hypotheses_mean_four_searches_merged_and_deduped(fakes):
    fakes["llm_response"] = {"hypotheses": ["p1", "p2", "p3"]}
    retriever = hyde.HyDERetriever(num_hypotheses=3)
    chunks = await retriever.retrieve("vague question", top_k=5)

    assert len(fakes["searches"]) == 4  # the question + 3 hypotheses
    assert fakes["embedded_docs"] == ["p1", "p2", "p3"]  # embedded as documents
    assert retriever.hypotheses == ["p1", "p2", "p3"]
    assert retriever.tokens_used == 120
    # a.txt is found by 3 of the 4 searches -> first. "  BETA " is the same text as "Beta"
    # (whitespace/case), so only one of b.txt / d.txt survives.
    assert chunks[0].source == "a.txt"
    sources = [c.source for c in chunks]
    assert len(sources) == len(set(sources)) == 3
    assert {"a.txt", "c.txt"} <= set(sources)


async def test_generation_failure_falls_back_to_the_question(fakes):
    fakes["llm_error"] = RuntimeError("LLM down")
    retriever = hyde.HyDERetriever()
    chunks = await retriever.retrieve("q", top_k=2)
    assert retriever.hypotheses == [] and retriever.tokens_used == 0
    assert len(fakes["searches"]) == 1
    assert [c.source for c in chunks] == ["a.txt", "b.txt"]


@pytest.mark.parametrize(
    "response",
    [{"hypotheses": "not a list"}, {"other": []}, {"hypotheses": [1, "", None, " ok "]}],
)
async def test_bad_hypotheses_are_filtered(fakes, response):
    fakes["llm_response"] = response
    hypotheses = await hyde.HyDERetriever(num_hypotheses=3).generate_hypotheses("q")
    assert hypotheses in ([], ["ok"])


def test_normalize_text():
    assert hyde._normalize_text("  Hello \n  WORLD ") == "hello world"
