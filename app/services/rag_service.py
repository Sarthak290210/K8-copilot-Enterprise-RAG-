"""The RAG pipeline: retrieve -> (correct) -> build prompt -> answer -> (reflect) with sources.

Every feature is a flag (see DEFAULT_FLAGS), so each can be measured on and off:
- search_mode dense | sparse | hybrid, enable_rerank (Phase 2);
- enable_hyde: HyDE for the dense part of retrieval; enable_crag: grade the chunks and fall
  back to the web (Kubernetes questions only); enable_self_reflective: grade the answer and
  retry with a refined question (Phase 3).
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.models import (
    ChatResponse,
    CRAGEvaluation,
    QueryFlags,
    ResponseMetadata,
    RetrievedChunk,
    RetrievedChunkPreview,
)
from app.security.grounding_check import (
    UNVERIFIED_ANSWER,
    apply_grounding_policy,
    ground_with_retry,
    needs_check,
    regeneration_feedback,
)
from app.security.output_validator import clean_answer
from app.security.system_prompt import build_system_prompt
from app.services import embedding_service, llm_service, vector_store
from app.services.crag import crag_pipeline
from app.services.hyde import HyDERetriever
from app.services.passages import format_passages
from app.services.query_cache_service import cache_scope, note_tokens_saved, query_cache
from app.services.reranking import get_reranker
from app.services.self_reflective import refine_question, reflect_on_answer, should_regenerate

NOT_FOUND_ANSWER = "I couldn't find this in the knowledge base."
OUT_OF_SCOPE_ANSWER = (
    "I can only help with Kubernetes and IT-operations questions (clusters, workloads, "
    "networking, storage, monitoring, troubleshooting, incidents). Please ask something in "
    "that area."
)
CLARIFY_ANSWER = (
    "Your question is too vague for me to answer well. Please add more detail: which resource "
    "it's about (Pod, Deployment, Service, node...), what you're trying to do, and any error "
    "message or command output you see."
)
# Fixed replies written in code, never by the model: nothing to ground-check or cache.
_FIXED_ANSWERS = (NOT_FOUND_ANSWER, OUT_OF_SCOPE_ANSWER, CLARIFY_ANSWER)


def _default_flags() -> dict[str, Any]:
    """Defaults when a request doesn't set a flag, from the `*_ENABLED_BY_DEFAULT` settings.

    User's choice after E3 (2026-10-23): hybrid + CRAG on; HyDE, rerank, Self-RAG off.
    Any request can still override any flag (e.g. {"enable_crag": false}); eval profiles
    always pass every flag, so they don't depend on these defaults.
    """
    return {
        "search_mode": "hybrid" if settings.hybrid_search_enabled else "dense",
        "top_k": 5,
        "enable_rerank": settings.reranking_enabled_by_default,
        "enable_hyde": settings.hyde_enabled_by_default,
        "enable_crag": settings.crag_enabled_by_default,
        "enable_self_reflective": settings.self_reflective_enabled_by_default,
    }


DEFAULT_FLAGS: dict[str, Any] = _default_flags()


def validate_flags(flags: Any) -> dict[str, Any]:
    """The flags a caller set, checked by `QueryFlags` (top_k 1-20, real booleans, a known
    search_mode, no unknown keys). Raises ValueError (the API maps it to 400)."""
    if flags is None:
        return {}
    if not isinstance(flags, dict):
        raise ValueError("flags must be a JSON object")
    try:
        return QueryFlags.model_validate(flags).model_dump(exclude_none=True)
    except ValidationError as exc:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        raise ValueError("Invalid flags: " + "; ".join(problems)) from None


@dataclass
class _Trace:
    """What one request did, for the response metadata."""

    tokens: int = 0
    hypotheses: list[str] = field(default_factory=list)
    crag: CRAGEvaluation | None = None


@dataclass
class _Attempt:
    """One retrieve -> answer pass (Self-RAG may make several; the best one is returned)."""

    answer: str
    chunks: list[RetrievedChunk]
    model: str | None
    crag: CRAGEvaluation | None
    hypotheses: list[str]
    refined_question: str | None = None  # None = the user's own question was used
    score: float | None = None  # reflection score, if Self-RAG ran


def _flag(flags: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Safe read from the flags dict (None or a non-dict means 'use the default')."""
    if isinstance(flags, dict) and key in flags:
        return flags[key]
    return DEFAULT_FLAGS.get(key, default)


async def _retrieve(
    question: str, flags: dict[str, Any] | None, trace: _Trace | None = None
) -> list[RetrievedChunk]:
    """dense | sparse | hybrid search, optionally with HyDE, rerank and CRAG.

    Without rerank/HyDE/CRAG, `top_k` chunks are fetched directly, so the `naive` flags run
    exactly the Phase 1 path that E1 measured. With rerank, `reranker_initial_top_k` (20)
    candidates are fetched and the cross-encoder keeps the best `top_k`. HyDE replaces the
    dense part, and in hybrid mode BM25 searches the question plus HyDE's passages (HyDE is
    ignored for `sparse`); a HyDE request answers from at least `hyde_top_k` (8) chunks.
    Hybrid search keeps at most `max_chunks_per_source` (2) chunks per document, except for
    rerank candidates (the reranker makes its own choice).
    """
    flags = validate_flags(flags)  # cheap; callers other than run_rag (Phase 4 graph) too
    trace = trace or _Trace()
    search_mode = _flag(flags, "search_mode")
    top_k = int(_flag(flags, "top_k"))
    hyde = bool(_flag(flags, "enable_hyde")) and search_mode != "sparse"
    if hyde:
        top_k = max(top_k, settings.hyde_top_k)
    rerank = bool(_flag(flags, "enable_rerank"))
    fetch_k = max(top_k, settings.reranker_initial_top_k) if rerank else top_k
    per_source = 0 if rerank else settings.max_chunks_per_source

    if search_mode == "sparse":
        chunks = await vector_store.sparse_search(question, top_k=fetch_k)
    elif hyde:
        retriever = HyDERetriever()
        if search_mode == "hybrid":
            # HyDE is the dense half of hybrid_search (same depth/RRF); BM25 then searches the
            # question plus the passages it wrote.
            chunks = await vector_store.hybrid_search(
                question,
                None,
                top_k=fetch_k,
                dense_search=lambda depth: retriever.retrieve(question, top_k=depth),
                sparse_query=lambda: retriever.sparse_query(question),
                max_per_source=per_source,
            )
        else:
            chunks = await retriever.retrieve(question, top_k=fetch_k)
        trace.hypotheses = retriever.hypotheses
        trace.tokens += retriever.tokens_used
    else:
        query_vector = await embedding_service.embed_query(question)
        if search_mode == "hybrid":
            chunks = await vector_store.hybrid_search(
                question, query_vector, top_k=fetch_k, max_per_source=per_source
            )
        else:
            chunks = await vector_store.search(query_vector, top_k=fetch_k)

    if rerank:
        # The cross-encoder is CPU-bound: keep it off the event loop.
        chunks = await asyncio.to_thread(get_reranker().rerank, question, chunks, top_k)
    if _flag(flags, "enable_crag"):
        chunks, trace.crag = await crag_pipeline(question, chunks)
        trace.tokens += trace.crag.tokens_used
    return chunks


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Spotlighted documents (L8) the model cites as [n]."""
    return format_passages(chunks, with_sources=True)


async def _generate(
    question: str,
    chunks: list[RetrievedChunk],
    flags: dict[str, Any] | None,
    feedback: str = "",
) -> dict[str, Any]:
    """feedback: the grounding check's list of unsupported statements, for the one rewrite."""
    user_message = f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}{feedback}"
    system_prompt = build_system_prompt(
        reranked=bool(_flag(flags, "enable_rerank")),
        # HyDE is ignored for sparse search (_retrieve), so the prompt mustn't mention it there
        hyde=bool(_flag(flags, "enable_hyde")) and _flag(flags, "search_mode") != "sparse",
    )
    result = await llm_service.generate(system_prompt, user_message)
    answer = clean_answer(result["content"])  # L9: plain [n] citations, no wrapping fence
    return {"answer": answer, "usage": result["usage"], "model": result["model"]}


def _no_context_answer(crag: CRAGEvaluation | None) -> str:
    """What to say when there's nothing to answer from (no LLM call is made)."""
    if crag is not None and crag.too_vague:
        return CLARIFY_ANSWER  # vague and the docs don't help: ask instead of guessing
    if crag is not None and crag.in_scope is False:
        return OUT_OF_SCOPE_ANSWER  # off-topic and not in the docs: CRAG skipped the web
    return NOT_FOUND_ANSWER


# Bump when a prompt or the pipeline changes in a way that changes answers: old cached answers
# then stop matching (they expire within CACHE_TTL_RAG anyway).
# 2: the key gained pipeline_settings(); model-written declines skipped.
# 3: Phase 6 (spotlighted context, hardened prompt, grounding check).
# 4: after E6: one grounded rewrite, no warning text in the answer, "name the document" rule.
# 5: HyDE + hybrid: BM25 searches the question plus HyDE's passages.
# 6: answer prompt: no assumptions / nothing from the model's own knowledge.
# 7: answer prompt: web results count as documents (YAML examples from them allowed).
# 8: the grounding check sees each document's source (it can rewrite / replace answers).
# 9: answer prompt says how the documents were chosen when rerank / HyDE is on.
# 10: strict grounding check (only stated claims count; more answers get the one rewrite).
# 11: HyDE answer prompt: documents' terms, keep qualifiers, no goal-as-fact, say what's missing.
# 12: hybrid search max 2 chunks per document; HyDE answers from 8 chunks.
# 13: grounding check: summary sentences and strong words checked as claims; pass score 0.9.
RAG_CACHE_VERSION = 13


def pipeline_settings() -> dict[str, Any]:
    """Settings that change the answers (beyond the request flags and the answer model).

    Part of the rag_answer cache key, so changing one in .env never serves answers built under
    the old value; the eval also puts it in its checkpoint key and result file.
    """
    return {
        "grader_model": settings.llm_model_grader,
        "reasoning_effort": settings.llm_reasoning_effort,
        "rrf_k": settings.rrf_k,
        "reranker_model": settings.reranker_model,
        "reranker_max_length": settings.reranker_max_length,
        "reranker_initial_top_k": settings.reranker_initial_top_k,
        "hyde_num_hypotheses": settings.hyde_num_hypotheses,
        "hyde_sparse_query": "question+passages",  # BM25 text with HyDE on (2026-10-28)
        "hyde_top_k": settings.hyde_top_k,
        "max_chunks_per_source": settings.max_chunks_per_source,
        "crag_relevance_threshold": settings.crag_relevance_threshold,
        "crag_ambiguous_threshold": settings.crag_ambiguous_threshold,
        "reflection_min_score": settings.reflection_min_score,
        "max_reflection_retries": settings.max_reflection_retries,
        # Phase 6: the grounding check can rewrite an answer; the guard thresholds decide what
        # gets blocked (recorded in eval results; also part of the eval checkpoint key).
        "grounding_enabled": settings.grounding_enabled,
        "grounding_pass_score": settings.grounding_pass_score,
        "grounding_warn_score": settings.grounding_warn_score,
        # 2026-10-28: "stated or implied" -> "stated_only" -> + summaries / strong words
        "grounding_strictness": "stated_only+conclusions",
        "security_models_enabled": settings.security_models_enabled,
        "prompt_injection_threshold": settings.prompt_injection_threshold,
        "toxicity_threshold": settings.toxicity_threshold,
        "output_toxicity_threshold": settings.output_toxicity_threshold,
    }


def _cache_context(flags_used: dict[str, Any]) -> dict[str, Any]:
    """Everything besides the question that changes the answer: every flag (search_mode,
    rerank, HyDE, CRAG, Self-RAG, top_k), the models and the pipeline settings. A different
    profile or setting never shares an answer."""
    return {
        "flags": flags_used,
        "answer_model": settings.llm_model_answer,
        "embedding_model": settings.embedding_model,
        "pipeline": pipeline_settings(),
        "version": RAG_CACHE_VERSION,
    }


# The system prompt tells the model to "say you don't know" (straight or curly apostrophe).
_MODEL_DECLINE = re.compile(r"\bI don['’]t know\b", re.IGNORECASE)


def _cacheable(answer: str, chunks: list[RetrievedChunk]) -> bool:
    """Only real answers: never one made without context (not found / off-topic / too vague)
    or one where the model said it doesn't know — cheap to redo, and the docs, the web or a
    retry may have it next time."""
    if not chunks or answer in (*_FIXED_ANSWERS, UNVERIFIED_ANSWER):
        return False
    return _MODEL_DECLINE.search(answer) is None


def _dedupe(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))  # keeps retrieval order


async def run_rag_with_trace(
    question: str, flags: dict[str, Any] | None = None
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    """run_rag plus the full retrieved chunks (eval needs them as Ragas contexts).

    With `enable_self_reflective` (user's design, 2026-10-26): answer → grade relevance,
    accuracy, completeness and clarity (score = their mean) → if it's below
    `reflection_min_score` (0.8), refine the question (the grader's suggestion, or one small
    rewrite call), retrieve and answer the refined question, and grade again (still against the
    user's question) — at most `max_reflection_retries` (2) times. The first answer that
    reaches the bar is returned, otherwise the best-scoring one.
    """
    with cache_scope():
        return await _run_rag_with_trace(question, flags)


async def _run_rag_with_trace(
    question: str, flags: dict[str, Any] | None
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    start = time.perf_counter()
    flags = validate_flags(flags)
    flags_used = {key: _flag(flags, key) for key in DEFAULT_FLAGS}
    cache_context = _cache_context(flags_used)
    if (hit := await query_cache.get_rag_answer(question, cache_context)) is not None:
        response = ChatResponse.model_validate(hit["response"])
        note_tokens_saved(response.metadata.tokens_used)
        response.metadata.tokens_used = 0
        _stamp_cache(response, start)
        return response, [RetrievedChunk.model_validate(c) for c in hit.get("chunks", [])]
    reflective = bool(_flag(flags, "enable_self_reflective"))
    trace = _Trace()

    current_question = question  # what this attempt searches for and answers
    refined: str | None = None
    retries = 0
    best: _Attempt | None = None
    while True:
        # Per-attempt trace: the response describes the attempt it returns, not the last one.
        trace.crag, trace.hypotheses = None, []
        chunks = await _retrieve(current_question, flags, trace)
        if not chunks:
            # Nothing to ground an answer in: don't spend LLM tokens.
            answer, model = _no_context_answer(trace.crag), None
        else:
            generated = await _generate(current_question, chunks, flags)
            answer, model = generated["answer"], generated["model"]
            trace.tokens += generated["usage"]["total_tokens"]
        attempt = _Attempt(answer, chunks, model, trace.crag, trace.hypotheses, refined)
        if not reflective or not chunks:
            # A retry that finds nothing never replaces an earlier real answer.
            best = best or attempt
            break
        reflection = await reflect_on_answer(question, answer, chunks, refined)
        trace.tokens += reflection.tokens_used
        attempt.score = reflection.score
        if best is None or best.score is None or reflection.score > best.score:
            best = attempt
        if not should_regenerate(reflection, retries):
            break  # good enough (>= the bar), or out of retries
        next_question = reflection.refined_question
        if next_question is None:
            next_question, used = await refine_question(question, answer, reflection)
            trace.tokens += used
        if not next_question or next_question.casefold() == current_question.casefold():
            break  # the same question would retrieve the same chunks
        retries += 1
        refined = current_question = next_question

    crag = best.crag
    if best.score is not None:
        confidence: float | None = best.score
    elif best.chunks and crag is not None and crag.action != "ungraded" and not crag.web_used:
        confidence = crag.max_score
    else:
        # No grader judged the context actually used (none ran, CRAG swapped in web results it
        # never graded, or there was no context at all): never invent a confidence.
        confidence = None

    response = ChatResponse(
        answer=best.answer,
        sources=_dedupe([chunk.source for chunk in best.chunks]),
        confidence=confidence,
        chunks=[RetrievedChunkPreview.from_chunk(chunk) for chunk in best.chunks],
        metadata=ResponseMetadata(
            intent="rag",
            flags_used=flags_used,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            tokens_used=trace.tokens,
            model=best.model or settings.llm_model_answer,
            hyde_hypotheses=best.hypotheses,
            crag=crag,
            reflection_iterations=retries,
            reflection_score=best.score,
            refined_question=best.refined_question,
        ),
    )
    if needs_check(response.answer, best.chunks, _FIXED_ANSWERS):
        # L10 (Phase 6): checked before caching, so a cached answer carries its grounding.
        # Below the pass score the answer is rewritten once with the unsupported statements.
        attempt = best

        async def rewrite(unsupported: list[str]) -> tuple[str, int]:
            question_asked = attempt.refined_question or question
            generated = await _generate(
                question_asked, attempt.chunks, flags, regeneration_feedback(unsupported)
            )
            return generated["answer"], generated["usage"]["total_tokens"]

        response.answer, grounding = await ground_with_retry(response.answer, best.chunks, rewrite)
        response.metadata.tokens_used += grounding.tokens_used
        apply_grounding_policy(response, grounding)
    _stamp_cache(response, start)
    if _cacheable(response.answer, best.chunks):
        await query_cache.set_rag_answer(
            question,
            cache_context,
            {
                "response": response.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in best.chunks],
            },
        )
    return response, best.chunks


def _stamp_cache(response: ChatResponse, start: float) -> None:
    """This request's cache hits, savings and latency (a cached answer keeps none of the
    original request's)."""
    with cache_scope() as trace:
        response.metadata.cache_hits = dict(trace.hits)
        response.metadata.cost_saved_usd = trace.cost_saved_usd
    response.metadata.latency_ms = round((time.perf_counter() - start) * 1000, 1)


async def run_rag(question: str, flags: dict[str, Any] | None = None) -> ChatResponse:
    """Public entry point: answer a question from the knowledge base."""
    response, _ = await run_rag_with_trace(question, flags)
    return response
