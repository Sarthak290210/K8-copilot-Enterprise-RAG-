"""One way to ask the system a question during eval."""

from abc import ABC, abstractmethod
from typing import Any

from app.models import ChatResponse, RetrievedChunk
from app.services import rag_service

# Text2SQL and the RAG+SQL agent arrive in Phase 4.
NOT_YET_SUPPORTED_INTENTS = {
    "sql": "Text2SQL arrives in Phase 4",
    "hybrid": "agent arrives in Phase 4",
}


class SkippedIntent(Exception):
    """The question can't be evaluated yet; the row is recorded as skipped, not failed."""


class Invoker(ABC):
    @abstractmethod
    async def invoke(
        self, question: str, flags: dict[str, Any], intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        """Return the answer and the full retrieved chunks (Ragas needs them as contexts)."""


class ServiceInvoker(Invoker):
    """Calls the RAG service in-process (no HTTP, no auth, no rate limit)."""

    async def invoke(
        self, question: str, flags: dict[str, Any], intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        if intent in NOT_YET_SUPPORTED_INTENTS:
            raise SkippedIntent(NOT_YET_SUPPORTED_INTENTS[intent])
        # web_fallback questions run through RAG too: until CRAG (Phase 3) they *should* fail.
        return await rag_service.run_rag_with_trace(question, flags)
