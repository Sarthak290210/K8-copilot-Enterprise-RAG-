"""HyDE (Hypothetical Document Embeddings): better dense retrieval for vague questions.

A short, vague question ("how do I stop a tenant hogging CPU?") embeds poorly. So the LLM first
writes a few short passages that *would* answer it, in the style of the docs; those passages
are embedded and searched, and land near the real documentation. The original question is
searched too, and all the lists are merged with RRF. In hybrid search the BM25 half searches
the question plus the passages (`sparse_query`), so keyword search gets the docs' terms too.
"""

import asyncio
import re
from typing import Any

from loguru import logger

from app.config import settings
from app.models import RetrievedChunk
from app.services import embedding_service, llm_service, vector_store
from app.services.sparse_vector_service import fuse_many

HYDE_SYSTEM_PROMPT = (
    "You write short passages in the style of the official Kubernetes documentation. "
    'Return JSON: {{"hypotheses": [<{n} strings>]}}. Each passage is 2-4 sentences that '
    "would answer the user's question, using the Kubernetes terms the docs would use "
    "(object kinds, fields, kubectl commands). Write only the JSON."
)
MAX_HYPOTHESIS_CHARS = 1_000


def _normalize_text(text: str) -> str:
    """Lowercase, whitespace collapsed: used to spot the same chunk text twice."""
    return re.sub(r"\s+", " ", text).strip().lower()


class HyDERetriever:
    """One per request: it keeps the hypotheses and token usage of its last `retrieve`."""

    def __init__(self, num_hypotheses: int | None = None) -> None:
        self.num_hypotheses = num_hypotheses or settings.hyde_num_hypotheses
        self.hypotheses: list[str] = []
        self.tokens_used = 0

    async def generate_hypotheses(self, question: str) -> list[str]:
        """One JSON-mode call. Fail-soft: on any error, no hypotheses (the question alone is
        still searched), so HyDE can only add recall, never break a request."""
        try:
            result = await llm_service.generate_with_json(
                HYDE_SYSTEM_PROMPT.format(n=self.num_hypotheses),
                question,
                model=settings.llm_model_grader,
            )
        except Exception as exc:
            self.tokens_used += llm_service.tokens_spent(exc)  # a bad-JSON reply is still billed
            logger.warning(
                "HyDE generation failed ({}); searching the question only", type(exc).__name__
            )
            return []
        self.tokens_used += result["usage"]["total_tokens"]
        raw: Any = result["data"].get("hypotheses", [])
        if not isinstance(raw, list):
            return []
        return [h.strip()[:MAX_HYPOTHESIS_CHARS] for h in raw if isinstance(h, str) and h.strip()][
            : self.num_hypotheses
        ]

    def sparse_query(self, question: str) -> str:
        """BM25 text for hybrid search: the question joined with this request's passages.

        The user's own words ("tenant", "shared computer") matched kernel/networking noise PDFs
        and pushed out the docs HyDE had found; the passages carry the docs' terms
        (ResourceQuota, LimitRange). Free replay over E3's passages (28 questions, ID-based):
        precision 0.731 -> 0.808, recall 0.750 -> 0.804. Without passages: the question only.
        """
        return " ".join([question, *self.hypotheses])

    async def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Dense search with the question + each hypothesis, merged by RRF, best `top_k`.

        Hypotheses are embedded as documents (no bge query instruction): they're meant to look
        like doc passages. Duplicate chunk texts (the same passage stored twice) are dropped.
        """
        # The LLM call and the question's embedding don't depend on each other: run together.
        self.hypotheses, query_vector = await asyncio.gather(
            self.generate_hypotheses(question), embedding_service.embed_query(question)
        )
        hypothesis_vectors = (
            await embedding_service.embed_texts(self.hypotheses, cache=True)
            if self.hypotheses
            else []
        )
        ranked_lists = list(
            await asyncio.gather(
                *(
                    vector_store.search(vector, top_k=top_k)
                    for vector in [query_vector, *hypothesis_vectors]
                )
            )
        )
        fused = fuse_many(ranked_lists, k=settings.rrf_k, top_k=len(ranked_lists) * top_k)
        unique: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in fused:
            key = _normalize_text(chunk.text)
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique[:top_k]
