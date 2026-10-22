"""Re-order candidate chunks with a cross-encoder (two-stage retrieval).

Embeddings encode the question and a chunk separately (fast, rough). A cross-encoder reads them
together and scores how well the chunk answers the question (slow, precise). So: fetch ~20
candidates cheaply, rerank them here, keep the best `top_k`.
"""

from functools import lru_cache
from typing import Any

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from app.config import settings
from app.models import RetrievedChunk


class Reranker:
    """`local` (sentence-transformers CrossEncoder on this machine) or `voyage` (Voyage API)."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or settings.reranker_backend
        self._model: CrossEncoder | None = None
        self._voyage: Any = None

    def _load_local_model(self) -> CrossEncoder:
        """Loaded once, on first use. Local Hugging Face cache first (flaky DNS here)."""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs: dict[str, Any] = {
                "device": device,
                "max_length": settings.reranker_max_length,  # longer chunks are truncated
            }
            try:
                self._model = CrossEncoder(settings.reranker_model, local_files_only=True, **kwargs)
            except OSError:
                self._model = CrossEncoder(settings.reranker_model, **kwargs)
            logger.info("Loaded reranker {} on {}", settings.reranker_model, device)
        return self._model

    def _load_voyage_client(self) -> Any:
        if self._voyage is None:
            from voyageai.client import Client  # only needed for the voyage backend

            if settings.voyage_api_key is None:
                raise RuntimeError("RERANKER_BACKEND=voyage but VOYAGE_API_KEY is not set")
            self._voyage = Client(api_key=settings.voyage_api_key.get_secret_value())
        return self._voyage

    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """The best `top_k` chunks by rerank score; the old score is kept in metadata.

        Fail-soft: if the reranker breaks, log it and return the first `top_k` unchanged, so a
        reranker problem degrades quality instead of failing the request.
        """
        if not chunks:
            return []
        try:
            if self.backend == "voyage":
                scored = self._rerank_voyage(question, chunks, top_k)
            else:
                scored = self._rerank_local(question, chunks, top_k)
        except Exception as exc:
            logger.warning("Reranking failed ({}); keeping retrieval order", type(exc).__name__)
            return chunks[:top_k]
        return [
            chunk.model_copy(
                update={
                    "score": round(score, 6),
                    "metadata": {**chunk.metadata, "retrieval_score": chunk.score},
                }
            )
            for chunk, score in scored
        ]

    def _rerank_local(
        self, question: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[tuple[RetrievedChunk, float]]:
        scores = self._load_local_model().predict([(question, c.text) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]

    def _rerank_voyage(
        self, question: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[tuple[RetrievedChunk, float]]:
        result = self._load_voyage_client().rerank(
            question, [c.text for c in chunks], model=settings.voyage_model, top_k=top_k
        )
        return [(chunks[r.index], float(r.relevance_score)) for r in result.results]


@lru_cache
def get_reranker() -> Reranker:
    """One reranker per process, so the model loads once."""
    return Reranker()
