"""Reranker with mocked models (no model download, no Voyage calls)."""

from types import SimpleNamespace

from app.models import RetrievedChunk
from app.services import reranking


def chunks(*sources):
    return [
        RetrievedChunk(text=f"text of {s}", source=s, score=0.1 * i, metadata={"chunk_index": i})
        for i, s in enumerate(sources)
    ]


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def test_local_rerank_orders_by_model_score_and_keeps_old_score():
    reranker = reranking.Reranker(backend="local")
    model = FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker._model = model  # type: ignore[assignment]
    result = reranker.rerank("q", chunks("a", "b", "c"), top_k=2)
    assert [c.source for c in result] == ["b", "c"]
    assert result[0].score == 0.9
    assert result[0].metadata == {"chunk_index": 1, "retrieval_score": 0.1}
    assert model.pairs == [("q", "text of a"), ("q", "text of b"), ("q", "text of c")]


def test_voyage_rerank_maps_results_back():
    reranker = reranking.Reranker(backend="voyage")
    calls = {}

    class FakeVoyage:
        def rerank(self, query, documents, model, top_k):
            calls.update(query=query, documents=documents, top_k=top_k)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=2, relevance_score=0.8),
                    SimpleNamespace(index=0, relevance_score=0.3),
                ]
            )

    reranker._voyage = FakeVoyage()
    result = reranker.rerank("q", chunks("a", "b", "c"), top_k=2)
    assert [c.source for c in result] == ["c", "a"]
    assert result[0].score == 0.8
    assert calls == {"query": "q", "documents": ["text of a", "text of b", "text of c"], "top_k": 2}


def test_rerank_failure_falls_back_to_retrieval_order():
    reranker = reranking.Reranker(backend="local")

    class Broken:
        def predict(self, pairs):
            raise RuntimeError("model crashed")

    reranker._model = Broken()  # type: ignore[assignment]
    original = chunks("a", "b", "c")
    assert reranker.rerank("q", original, top_k=2) == original[:2]


def test_empty_input_and_singleton():
    assert reranking.Reranker(backend="local").rerank("q", [], top_k=5) == []
    assert reranking.get_reranker() is reranking.get_reranker()  # model loads once
