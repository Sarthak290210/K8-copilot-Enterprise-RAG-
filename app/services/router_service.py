"""Intent routing: decide whether a question needs the database (sql), the docs (rag), or both.

One JSON-mode call per request, made once in the graph (never again inside the RAG service).
Fix vs reference: no `_DOCUMENT_HINTS` keyword list (it belonged to an unrelated corpus).
"""

from typing import Literal

from loguru import logger

from app.config import settings
from app.services import llm_service

Intent = Literal["sql", "rag", "hybrid"]
INTENTS: tuple[Intent, ...] = ("sql", "rag", "hybrid")

_INTENT_SYSTEM_PROMPT = """You route questions for a Kubernetes IT-operations assistant.
Two knowledge sources exist:
- DATABASE (live fleet data): clusters, nodes, deployments, pods (with current status such as
  Running / Pending / CrashLoopBackOff / Failed), incidents (severity P1-P4, MTTR, root causes),
  alerts, and on-call paging logs. Use it for counts, lists, numbers, "which cluster", "how
  many", "show details for <cluster>", trends and statistics about OUR fleet.
- DOCS (Kubernetes documentation): concepts, how-to steps, commands, configuration, best
  practices, troubleshooting procedures, and general/new Kubernetes knowledge.

Choose one intent:
- "sql": answerable from the database alone. e.g. "How many P1 incidents happened last month?",
  "Which cluster has the most Pending pods?"
- "rag": needs the docs (or general Kubernetes knowledge) and no fleet data. e.g. "What is a
  StatefulSet?", "How do I roll back a Deployment?", and anything unrelated to our data.
- "hybrid": needs BOTH a number/fact from the database AND an explanation or procedure from the
  docs. e.g. "How many ImagePullBackOff pods do we have, and how do I fix them?"
Return JSON: {"intent": "sql" | "rag" | "hybrid"}."""


async def classify_intent(question: str) -> tuple[Intent, int]:
    """(intent, tokens used). Anything unexpected — an error, bad JSON, an unknown value —
    falls back to "rag", the path that never touches the database."""
    try:
        result = await llm_service.generate_with_json(
            _INTENT_SYSTEM_PROMPT, f"Question: {question}", model=settings.llm_model_grader
        )
    except Exception as exc:
        logger.warning("Intent routing failed ({}); defaulting to rag", type(exc).__name__)
        return "rag", llm_service.tokens_spent(exc)
    raw = str(result["data"].get("intent", "")).strip().lower()
    intent: Intent = raw if raw in INTENTS else "rag"  # type: ignore[assignment]
    if raw not in INTENTS:
        logger.info("Unknown intent {!r}; defaulting to rag", raw[:40])
    return intent, result["usage"]["total_tokens"]
