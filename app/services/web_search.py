"""Web search (Tavily): the fallback knowledge source CRAG uses when the docs don't cover a
question. General web, not restricted to Kubernetes sites (user's choice, 2026-10-23)."""

from functools import lru_cache

from loguru import logger
from tavily import AsyncTavilyClient

from app.config import settings
from app.models import RetrievedChunk

WEB_ORIGIN = "web"  # metadata["origin"] of every web result
WEB_TIMEOUT_SECONDS = 15


@lru_cache
def get_client() -> AsyncTavilyClient | None:
    """Shared Tavily client, or None when TAVILY_API_KEY isn't set."""
    if settings.tavily_api_key is None:
        return None
    return AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())


def is_web_chunk(chunk: RetrievedChunk) -> bool:
    return chunk.metadata.get("origin") == WEB_ORIGIN


async def search_web(query: str, max_results: int = 5) -> list[RetrievedChunk]:
    """Top web results as chunks (`source` = the URL). Never raises: no key or a failed call
    gives an empty list, and the caller decides what to do without web results.

    Each call costs 1 Tavily credit ("basic" depth; the free plan has 1,000 a month).
    """
    client = get_client()
    if client is None:
        logger.info("Web search skipped: TAVILY_API_KEY is not set")
        return []
    try:
        response = await client.search(
            query, max_results=max_results, search_depth="basic", timeout=WEB_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning("Web search failed ({})", type(exc).__name__)
        return []
    chunks = []
    for result in response.get("results", []):
        url, content = result.get("url"), (result.get("content") or "").strip()
        if not url or not content:
            continue
        title = (result.get("title") or "").strip()
        chunks.append(
            RetrievedChunk(
                text=f"{title}\n{content}" if title else content,
                source=url,
                score=float(result.get("score") or 0.0),
                metadata={"origin": WEB_ORIGIN, "title": title},
            )
        )
    return chunks
