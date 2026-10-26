"""Test settings: never use real secrets or cloud services.

Environment variables take precedence over .env in pydantic-settings, so setting them here
(before any `app` import) isolates the tests from the developer's real .env.
"""

import os
import tempfile

os.environ["CEREBRAS_API_KEY"] = "test-cerebras-key"
os.environ["DATABASE_URL"] = "postgresql://test:test@127.0.0.1:1/test"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-characters-long"
# Empty Upstash settings -> get_redis_client() returns None -> in-memory rate limiting.
os.environ["UPSTASH_REDIS_URL"] = ""
os.environ["UPSTASH_REDIS_TOKEN"] = ""
# Models (tokenizer, bge) load from the local Hugging Face cache; tests never hit the network.
os.environ["HF_HUB_OFFLINE"] = "1"
# Phase 5: caching is off unless a test turns it on (the `cache_on` fixture), so a cached
# answer from one test never leaks into another; doc-dedup records go to a temp folder.
os.environ["CACHE_ENABLED"] = "false"
os.environ["LOCAL_STORAGE_DIR"] = tempfile.mkdtemp(prefix="k8-copilot-test-storage-")

import pytest  # noqa: E402

from app.middleware import rate_limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_limit_state():
    """Each test starts with empty in-memory rate-limit windows and no cached client."""
    rate_limiter._memory_windows.clear()
    rate_limiter.get_redis_client.cache_clear()
    yield
    rate_limiter._memory_windows.clear()


@pytest.fixture(autouse=True)
def _fresh_doc_storage(tmp_path, monkeypatch):
    """Each test gets an empty doc-dedup store (same bytes in two tests aren't duplicates)."""
    from app.api import documents
    from app.config import settings
    from app.storage import storage_backend

    monkeypatch.setattr(settings, "local_storage_dir", str(tmp_path / "storage"))
    storage_backend.get_storage_backend.cache_clear()
    documents.get_doc_cache.cache_clear()
    yield
    storage_backend.get_storage_backend.cache_clear()
    documents.get_doc_cache.cache_clear()


@pytest.fixture
def cache_on(monkeypatch):
    """Caching enabled, memory only, empty; returns the shared query_cache."""
    from app.config import settings
    from app.services import query_cache_service

    cache = query_cache_service.QueryCacheService(use_shared_redis=False)
    monkeypatch.setattr(settings, "cache_enabled", True)
    for module in (
        "app.services.query_cache_service",
        "app.services.embedding_service",
        "app.services.router_service",
        "app.services.rag_service",
        "app.services.sql_service",
        "app.api.documents",
        "app.api.admin",
    ):
        monkeypatch.setattr(f"{module}.query_cache", cache)
    return cache
