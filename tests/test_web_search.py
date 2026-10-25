"""Web search with a mocked Tavily client."""

from app.models import RetrievedChunk
from app.services import web_search


class FakeTavily:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_no_key_means_no_client(monkeypatch):
    monkeypatch.setattr(web_search.settings, "tavily_api_key", None)
    web_search.get_client.cache_clear()
    try:
        assert web_search.get_client() is None
    finally:
        web_search.get_client.cache_clear()


async def test_no_client_returns_empty(monkeypatch):
    monkeypatch.setattr(web_search, "get_client", lambda: None)
    assert await web_search.search_web("q") == []


async def test_results_become_web_chunks(monkeypatch):
    client = FakeTavily(
        {
            "results": [
                {"url": "https://k8s.io/a", "title": "A", "content": "About A", "score": 0.7},
                {"url": "https://x.io", "title": "", "content": "  "},  # empty -> skipped
                {"title": "no url", "content": "x"},  # no url -> skipped
            ]
        }
    )
    monkeypatch.setattr(web_search, "get_client", lambda: client)
    chunks = await web_search.search_web("what is k8s 1.34?", max_results=3)
    assert chunks == [
        RetrievedChunk(
            text="A\nAbout A",
            source="https://k8s.io/a",
            score=0.7,
            metadata={"origin": "web", "title": "A"},
        )
    ]
    assert web_search.is_web_chunk(chunks[0])
    query, kwargs = client.calls[0]
    assert query == "what is k8s 1.34?"
    assert kwargs["max_results"] == 3 and kwargs["search_depth"] == "basic"


async def test_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(web_search, "get_client", lambda: FakeTavily(error=TimeoutError()))
    assert await web_search.search_web("q") == []
