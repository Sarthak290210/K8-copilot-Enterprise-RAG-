"""One way to ask the system a question during eval."""

import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.models import ChatResponse, RetrievedChunk
from app.services import rag_service

# ServiceInvoker calls the RAG pipeline only, so it can't answer database questions.
RAG_ONLY_SKIPS = {
    "sql": "needs Text2SQL (use AgentInvoker)",
    "hybrid": "needs the agent (use AgentInvoker)",
}


class SkippedIntent(Exception):
    """The question can't be evaluated by this invoker; recorded as skipped, not failed."""


class Invoker(ABC):
    @abstractmethod
    async def invoke(
        self, question: str, flags: dict[str, Any], intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        """Return the answer and the full retrieved chunks (Ragas needs them as contexts)."""


class ServiceInvoker(Invoker):
    """Calls the RAG service in-process (no routing, no SQL): the Phase 1-3 path."""

    async def invoke(
        self, question: str, flags: dict[str, Any], intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        if intent in RAG_ONLY_SKIPS:
            raise SkippedIntent(RAG_ONLY_SKIPS[intent])
        return await rag_service.run_rag_with_trace(question, flags)


class AgentInvoker(Invoker):
    """The Phase 4 agent graph in-process (routing, Text2SQL, hybrid), with an in-memory
    checkpointer and **auto-approved** SQL: the eval measures answers, not the human step.
    The golden's intent is not passed in: the router has to find it (intent_accuracy)."""

    def __init__(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from app.core.graph import build_graph

        self.graph = build_graph(InMemorySaver())

    async def invoke(
        self, question: str, flags: dict[str, Any], intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        from app.core.graph import resume_agent, run_agent

        query_id = uuid.uuid4()
        result = await run_agent(self.graph, question, flags, query_id)
        if result.pending:
            before = result.response.metadata
            result = await resume_agent(self.graph, query_id, approved=True)
            # One question, two calls: report the cache hits/savings of both.
            meta = result.response.metadata
            for tier, hit in before.cache_hits.items():
                meta.cache_hits[tier] = meta.cache_hits.get(tier, False) or hit
            meta.cost_saved_usd = round(meta.cost_saved_usd + before.cost_saved_usd, 6)
        return result.response, result.chunks
