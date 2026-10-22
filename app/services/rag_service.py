"""The RAG pipeline: retrieve -> build prompt -> answer with sources."""

import asyncio
import time
from typing import Any

from app.config import settings
from app.models import ChatResponse, ResponseMetadata, RetrievedChunk, RetrievedChunkPreview
from app.security.system_prompt import build_system_prompt
from app.services import embedding_service, llm_service, vector_store
from app.services.reranking import get_reranker

NOT_FOUND_ANSWER = "I couldn't find this in the knowledge base."

# Defaults when a flag isn't given: the `naive` profile (E1 baseline) = exactly these.
DEFAULT_FLAGS: dict[str, Any] = {"search_mode": "dense", "top_k": 5, "enable_rerank": False}
SUPPORTED_SEARCH_MODES = {"dense", "sparse", "hybrid"}


def _flag(flags: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Safe read from the flags dict (None or a non-dict means 'use the default')."""
    if isinstance(flags, dict) and key in flags:
        return flags[key]
    return DEFAULT_FLAGS.get(key, default)


async def _retrieve(question: str, flags: dict[str, Any] | None) -> list[RetrievedChunk]:
    """dense | sparse | hybrid search; with `enable_rerank`, two-stage retrieval.

    Without rerank, `top_k` chunks are fetched directly, so dense without rerank is exactly
    the Phase 1 path that E1 measured. With rerank, `reranker_initial_top_k` (20) candidates
    are fetched and the cross-encoder keeps the best `top_k`.
    """
    search_mode = _flag(flags, "search_mode")
    if search_mode not in SUPPORTED_SEARCH_MODES:
        raise ValueError(
            f"Unknown search_mode '{search_mode}'. Use one of: "
            + ", ".join(sorted(SUPPORTED_SEARCH_MODES))
        )
    top_k = int(_flag(flags, "top_k"))
    rerank = bool(_flag(flags, "enable_rerank"))
    fetch_k = max(top_k, settings.reranker_initial_top_k) if rerank else top_k

    if search_mode == "sparse":
        chunks = await vector_store.sparse_search(question, top_k=fetch_k)
    else:
        query_vector = await embedding_service.embed_query(question)
        if search_mode == "hybrid":
            chunks = await vector_store.hybrid_search(question, query_vector, top_k=fetch_k)
        else:
            chunks = await vector_store.search(query_vector, top_k=fetch_k)

    if rerank:
        # The cross-encoder is CPU-bound: keep it off the event loop.
        chunks = await asyncio.to_thread(get_reranker().rerank, question, chunks, top_k)
    return chunks


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered blocks the model cites as [n]. Replaced by spotlighting in Phase 6."""
    return "\n\n".join(
        f"[{i}] (source: {chunk.source})\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )


async def _generate(
    question: str, chunks: list[RetrievedChunk], flags: dict[str, Any] | None
) -> dict[str, Any]:
    user_message = f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}"
    result = await llm_service.generate(build_system_prompt(), user_message)
    return {"answer": result["content"], "usage": result["usage"], "model": result["model"]}


def _dedupe(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))  # keeps retrieval order


async def run_rag_with_trace(
    question: str, flags: dict[str, Any] | None = None
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    """run_rag plus the full retrieved chunks (eval needs them as Ragas contexts)."""
    start = time.perf_counter()
    flags_used = {key: _flag(flags, key) for key in DEFAULT_FLAGS}
    chunks = await _retrieve(question, flags)

    if not chunks:
        # Nothing to ground an answer in: don't spend LLM tokens.
        answer, tokens, model = NOT_FOUND_ANSWER, 0, None
    else:
        generated = await _generate(question, chunks, flags)
        answer, tokens, model = (
            generated["answer"],
            generated["usage"]["total_tokens"],
            generated["model"],
        )

    response = ChatResponse(
        answer=answer,
        sources=_dedupe([chunk.source for chunk in chunks]),
        chunks=[RetrievedChunkPreview.from_chunk(chunk) for chunk in chunks],
        metadata=ResponseMetadata(
            intent="rag",
            flags_used=flags_used,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            tokens_used=tokens,
            model=model or settings.llm_model_answer,
        ),
    )
    return response, chunks


async def run_rag(question: str, flags: dict[str, Any] | None = None) -> ChatResponse:
    """Public entry point: answer a question from the knowledge base."""
    response, _ = await run_rag_with_trace(question, flags)
    return response
