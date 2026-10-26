import numpy as np

from app.services import embedding_service


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.append((list(texts), kwargs))
        return np.array([[float(len(t)), 0.0] for t in texts])


def install(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda name=None: model)
    return model


async def test_embed_texts_keeps_order_and_normalises(monkeypatch):
    model = install(monkeypatch)
    vectors = await embedding_service.embed_texts(["a", "bbb"])
    assert vectors == [[1.0, 0.0], [3.0, 0.0]]
    assert model.seen[0][1]["normalize_embeddings"] is True


async def test_empty_input_skips_the_model(monkeypatch):
    model = install(monkeypatch)
    assert await embedding_service.embed_texts([]) == []
    assert model.seen == []


async def test_query_gets_bge_instruction(monkeypatch):
    model = install(monkeypatch)
    await embedding_service.embed_query("What is a Pod?")
    assert model.seen[0][0] == [embedding_service.BGE_QUERY_INSTRUCTION + "What is a Pod?"]


async def test_cached_texts_run_the_model_only_for_misses(monkeypatch, cache_on):
    model = install(monkeypatch)
    assert await embedding_service.embed_texts(["a", "bb"], cache=True) == [[1.0, 0.0], [2.0, 0.0]]
    assert await embedding_service.embed_texts(["bb", "ccc", "a"], cache=True) == [
        [2.0, 0.0],
        [3.0, 0.0],
        [1.0, 0.0],
    ]
    assert [seen for seen, _ in model.seen] == [["a", "bb"], ["ccc"]]


async def test_documents_are_never_cached(monkeypatch, cache_on):
    model = install(monkeypatch)
    await embedding_service.embed_texts(["chunk"])
    await embedding_service.embed_texts(["chunk"])
    assert len(model.seen) == 2 and cache_on.stats()["tiers"]["embedding"]["sets"] == 0


async def test_query_embedding_is_cached(monkeypatch, cache_on):
    model = install(monkeypatch)
    await embedding_service.embed_query("What is a Pod?")
    await embedding_service.embed_query("What is a Pod?")
    assert len(model.seen) == 1
