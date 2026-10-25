"""Self-RAG (self-reflection): check an answer before the user sees it.

A grader LLM reads the user's question, the context and the answer, and scores four things
from 0 to 1 (user's design, 2026-10-26):
- relevance:    does it address what the user asked?
- accuracy:     is every claim supported by the context?
- completeness: does it cover what the question needs (as far as the context allows)?
- clarity:      is it clear and well organised?
The overall score is their mean. Below REFLECTION_MIN_SCORE (0.8) the pipeline refines the
question, retrieves and answers again, and grades again — at most MAX_REFLECTION_RETRIES (2)
times — and returns the first answer that reaches the bar (or the best one).
"""

from loguru import logger

from app.config import settings
from app.models import ReflectionResult, RetrievedChunk
from app.services import llm_service
from app.services.passages import format_passages

CRITERIA = ("relevance", "accuracy", "completeness", "clarity")

REFLECTION_SYSTEM_PROMPT = (
    "You review answers from a Kubernetes IT-operations assistant. Given the user's question, "
    "the numbered context passages and the answer, score each criterion from 0 to 1:\n"
    "- relevance: does the answer address what the user asked? (an 'I don't know' scores 0)\n"
    "- accuracy: is every claim supported by the context passages?\n"
    "- completeness: does it cover everything the question needs, as far as the context "
    "allows?\n"
    "- clarity: is it clear, specific and well organised?\n"
    "If the user's question is too vague to know what they want (e.g. 'explain', 'what is "
    "wrong', 'help'), the answer can only guess: relevance and completeness are each at most "
    "0.5. If a clarified question is given, judge relevance and completeness against it "
    "instead, as long as it is a reasonable reading of the user's question.\n"
    "If the average of the four is below {threshold}, you MUST also write refined_question: a "
    "clearer, more specific, self-contained Kubernetes question that states what the user most "
    "likely meant (use the answer and passages for hints) and would retrieve better passages. "
    "Otherwise refined_question is null.\n"
    'Return JSON: {{"relevance": <0-1>, "accuracy": <0-1>, "completeness": <0-1>, '
    '"clarity": <0-1>, "issues": [<short strings>], "refined_question": "<string or null>"}}.'
)

REFINE_SYSTEM_PROMPT = (
    "You rewrite unclear questions for a Kubernetes IT-operations assistant. Given the user's "
    "question, a weak answer and its problems, write one clearer, more specific, self-contained "
    "Kubernetes question that states what the user most likely meant. "
    'Return JSON: {"refined_question": "<the question>"}.'
)


def _criterion(data: dict, name: str) -> float:
    try:
        return min(1.0, max(0.0, float(data.get(name, 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _clean_question(raw: object) -> str | None:
    return raw.strip()[:500] if isinstance(raw, str) and raw.strip() else None


async def reflect_on_answer(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    refined_question: str | None = None,
) -> ReflectionResult:
    """One JSON-mode grader call over the **whole** chunks (the answer was written from them;
    cut chunks would hide facts it cites). `refined_question` is the clarified question this
    answer was written for, if any; relevance is still judged against the user's question.

    **Fails closed**: if the call or its JSON fails, the answer gets score 0 and the issue
    "reflection unavailable" — never a silent pass.
    """
    asked = f"User's question: {question}"
    if refined_question:
        asked += f"\nClarified question (used for this answer): {refined_question}"
    try:
        result = await llm_service.generate_with_json(
            REFLECTION_SYSTEM_PROMPT.format(threshold=settings.reflection_min_score),
            f"{asked}\n\nContext:\n{format_passages(chunks)}\n\nAnswer:\n{answer}",
            model=settings.llm_model_grader,
        )
    except Exception as exc:
        logger.warning("Self-reflection failed ({}); failing closed", type(exc).__name__)
        return ReflectionResult(
            score=0.0,
            issues=["reflection unavailable"],
            tokens_used=llm_service.tokens_spent(exc),
        )
    data = result["data"]
    criteria = {name: _criterion(data, name) for name in CRITERIA}
    issues = data.get("issues")
    return ReflectionResult(
        **criteria,
        score=round(sum(criteria.values()) / len(CRITERIA), 4),
        issues=[str(i)[:200] for i in issues] if isinstance(issues, list) else [],
        refined_question=_clean_question(data.get("refined_question")),
        tokens_used=result["usage"]["total_tokens"],
    )


async def refine_question(
    question: str, answer: str, reflection: ReflectionResult
) -> tuple[str | None, int]:
    """Fallback when the grader scored low but gave no refined question: one small JSON call.
    Returns (refined question or None, tokens used). Fail-soft: an error means no retry."""
    issues = "; ".join(reflection.issues) or "low score"
    try:
        result = await llm_service.generate_with_json(
            REFINE_SYSTEM_PROMPT,
            f"User's question: {question}\n\nWeak answer:\n{answer}\n\nProblems: {issues}",
            model=settings.llm_model_grader,
        )
    except Exception as exc:
        logger.warning("Question refinement failed ({}); not retrying", type(exc).__name__)
        return None, llm_service.tokens_spent(exc)
    return _clean_question(result["data"].get("refined_question")), result["usage"]["total_tokens"]


def should_regenerate(reflection: ReflectionResult, iteration: int) -> bool:
    """Retry while the answer scores below the bar and retries are left (iteration = retries
    already made)."""
    return (
        reflection.score < settings.reflection_min_score
        and iteration < settings.max_reflection_retries
    )
