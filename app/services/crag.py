"""CRAG (Corrective RAG): grade the retrieved chunks, and correct the context if it's bad.

A grader LLM scores how relevant each chunk is (0-1). The best score decides the action:
- correct   (>= CRAG_RELEVANCE_THRESHOLD, 0.7): keep the relevant chunks;
- ambiguous (>= CRAG_AMBIGUOUS_THRESHOLD, 0.5): relevant chunks + a web search;
- incorrect (below):                            web results only.
Like a librarian who checks the books before handing them over.

The web is searched **only for Kubernetes / IT-operations questions** (user's rule,
2026-10-26: this is a Kubernetes ops assistant). The same grader call says whether the question
is in scope, so the check costs no extra call. An off-topic question the docs can't answer
gets no web search and no context (the pipeline then declines). Phase 6 adds the full input
guard (injection, toxicity) on top.
"""

from typing import Any

from loguru import logger

from app.config import settings
from app.models import CRAGEvaluation, RetrievedChunk
from app.services import llm_service
from app.services.passages import format_passages
from app.services.web_search import search_web

GRADER_SYSTEM_PROMPT = (
    "You grade search results for a Kubernetes IT-operations assistant.\n"
    "1. For each numbered passage, score from 0 to 1 how useful it is for answering the "
    "question (1 = directly answers it, 0.5 = related but incomplete, 0 = unrelated).\n"
    "2. Decide whether the question itself is in scope: true if it is about Kubernetes "
    "(any version, including new releases), containers, the cloud-native tools around them "
    "(e.g. Docker, Helm, Argo CD, Prometheus, Istio, kubectl), or the IT/DevOps/SRE operations "
    "of running them (deploying, scaling, networking, storage, security, monitoring, "
    "troubleshooting, incidents). False for anything else (general knowledge, weather, "
    "creative writing, personal messages, other subjects).\n"
    "3. Decide whether the question is too vague to answer: it has no clear subject or goal "
    "(e.g. 'explain', 'what is wrong', 'help me', 'why?'). A vague question is not off-topic: "
    "set kubernetes_related true for it.\n"
    'Return JSON: {"scores": [<one number per passage, in order>], '
    '"kubernetes_related": true|false, "too_vague": true|false, "reason": "<one sentence>"}.'
)
MAX_CHARS_PER_CHUNK = 1_200  # relevance grading doesn't need whole chunks; keeps the call cheap


def _action(max_score: float) -> str:
    if max_score >= settings.crag_relevance_threshold:
        return "correct"
    if max_score >= settings.crag_ambiguous_threshold:
        return "ambiguous"
    return "incorrect"


def _clean_scores(raw: Any, expected: int) -> list[float]:
    """Numbers clamped to 0-1, exactly one per chunk (missing ones count as 0)."""
    values = raw if isinstance(raw, list) else []
    scores = []
    for i in range(expected):
        try:
            scores.append(min(1.0, max(0.0, float(values[i]))))
        except (IndexError, TypeError, ValueError):
            scores.append(0.0)
    return scores


async def grade_chunks(question: str, chunks: list[RetrievedChunk]) -> CRAGEvaluation:
    """One JSON-mode call to the grader model: a score per chunk + whether the question is in
    scope. A failed call gives action "ungraded" (the caller keeps the chunks as they are).

    With no chunks there is nothing to score, but the call still runs to decide the scope: a
    Kubernetes question with no hits should go to the web, an off-topic one shouldn't.
    """
    passages = format_passages(chunks, max_chars=MAX_CHARS_PER_CHUNK) or "(no passages found)"
    try:
        result = await llm_service.generate_with_json(
            GRADER_SYSTEM_PROMPT,
            f"Question: {question}\n\nPassages:\n{passages}",
            model=settings.llm_model_grader,
        )
    except Exception as exc:
        logger.warning("CRAG grading failed ({}); keeping the chunks ungraded", type(exc).__name__)
        return CRAGEvaluation(
            action="ungraded",
            reason="grader unavailable",
            tokens_used=llm_service.tokens_spent(exc),  # a bad-JSON reply is still billed
        )
    data = result["data"]
    scores = _clean_scores(data.get("scores"), len(chunks))
    max_score = max(scores, default=0.0)
    related, vague = data.get("kubernetes_related"), data.get("too_vague")
    return CRAGEvaluation(
        scores=scores,
        max_score=max_score,
        action=_action(max_score),  # type: ignore[arg-type]
        reason=str(data.get("reason", ""))[:300],
        in_scope=related if isinstance(related, bool) else None,
        too_vague=vague if isinstance(vague, bool) else None,
        tokens_used=result["usage"]["total_tokens"],
    )


def should_trigger_web_search(evaluation: CRAGEvaluation) -> bool:
    """The docs fell short **and** the question is a clear Kubernetes / IT-ops one."""
    return (
        evaluation.action in ("ambiguous", "incorrect")
        and evaluation.in_scope is True
        and evaluation.too_vague is not True
    )


async def crag_pipeline(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], CRAGEvaluation]:
    """Grade, then correct: the context to answer from, plus the evaluation.

    If a web search is needed but gives nothing (no Tavily key, or it failed), the original
    chunks are kept rather than answering from an empty context; the evaluation's reason
    says so and its low max_score shows as low confidence.

    Too-vague questions never reach the web either: ambiguous keeps its relevant chunks,
    incorrect returns no chunks so the caller asks for more detail.

    Off-topic questions (in_scope False) never reach the web: ambiguous keeps its relevant
    chunks, incorrect returns no chunks so the caller declines. If the grader didn't say
    (in_scope None), incorrect keeps the original chunks, as when the web is unavailable.
    """
    evaluation = await grade_chunks(question, chunks)
    if evaluation.action == "ungraded":
        return chunks, evaluation

    relevant = [
        chunk.model_copy(update={"metadata": {**chunk.metadata, "crag_score": score}})
        for chunk, score in zip(chunks, evaluation.scores, strict=True)
        if score >= settings.crag_ambiguous_threshold
    ]
    if evaluation.action == "correct":
        return relevant, evaluation
    if evaluation.too_vague:
        # A web search for "explain" returns dictionary definitions (E3 q-023/q-024).
        if evaluation.action == "ambiguous":
            evaluation.reason = f"{evaluation.reason} (too vague: web search skipped)"
            return relevant, evaluation
        evaluation.reason = f"{evaluation.reason} (too vague: asking the user for detail)"
        return [], evaluation
    if not should_trigger_web_search(evaluation):
        if evaluation.action == "ambiguous":
            evaluation.reason = (
                f"{evaluation.reason} (web search skipped: not a confirmed Kubernetes question)"
            )
            return relevant, evaluation
        if evaluation.in_scope is False:
            evaluation.reason = f"{evaluation.reason} (off-topic: not a Kubernetes question)"
            return [], evaluation
        evaluation.reason = f"{evaluation.reason} (scope unknown: original chunks kept)"
        return chunks, evaluation

    web = await search_web(question)
    if not web:
        evaluation.reason = f"{evaluation.reason} (web search unavailable: original chunks kept)"
        return chunks, evaluation
    evaluation.web_used = True
    corrected = relevant + web if evaluation.action == "ambiguous" else web
    return corrected, evaluation
