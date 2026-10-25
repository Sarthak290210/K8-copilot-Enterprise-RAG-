"""Keyword (BM25) search and Reciprocal Rank Fusion.

BM25 scores a chunk by how many query words it contains, weighting rare words (like
`CrashLoopBackOff`) far above common ones and not over-rewarding long chunks. Dense search
matches meaning; this matches exact tokens. `fuse_rrf` merges the two ranked lists by rank.
"""

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import RetrievedChunk

# Words that appear everywhere and carry no meaning for search.
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "me my of on or so that the their them then there these this to was we what when where "
    "which who why will with you your".split()
)
# Runs of letters/digits joined by - _ . so K8s names stay whole: kube-proxy, v1.34, pod_spec.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokens, K8s names kept whole (`kube-proxy`, `crashloopbackoff`), no stopwords.

    A joined name also yields its parts, so a question saying "proxy" still matches a chunk
    that only says `kube-proxy`.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token not in STOPWORDS:
            tokens.append(token)
        parts = re.split(r"[-_.]", token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part and part not in STOPWORDS)
    return tokens


def chunk_id(chunk: RetrievedChunk) -> str:
    """Same identity as the Qdrant point ID: source + chunk_index."""
    return f"{chunk.source}::{chunk.metadata.get('chunk_index')}"


class SparseVectorIndex:
    """BM25 over every chunk, held in memory (built once, see vector_store)."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._docs)

    def fit(self, documents: list[dict[str, Any]]) -> None:
        """documents: [{text, source, metadata}], the same shape as Qdrant payloads."""
        self._docs = list(documents)
        # rank-bm25 divides by the corpus size: an empty corpus stays unfitted.
        self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in self._docs]) if self._docs else None

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Best BM25 matches first; chunks sharing no word with the query are left out."""
        tokens = _tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        best = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self._docs[i]["text"],
                source=self._docs[i]["source"],
                score=float(scores[i]),
                metadata=self._docs[i].get("metadata", {}),
            )
            for i in best
            if scores[i] > 0
        ]


def fuse_rrf(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int | None = None,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank + 1) over the lists a chunk appears in.

    Ranks, not raw scores, are merged (cosine and BM25 scores aren't comparable). A chunk found
    by both searches beats one found by only one. The fused score replaces `score`; the dense
    and sparse ranks are kept in metadata.
    """
    return fuse_many([dense, sparse], k=k, top_k=top_k, names=["dense", "sparse"])


def fuse_many(
    ranked_lists: list[list[RetrievedChunk]],
    k: int | None = None,
    top_k: int = 20,
    names: list[str] | None = None,
) -> list[RetrievedChunk]:
    """RRF over any number of ranked lists (HyDE merges the question's and each hypothesis'
    results). With `names`, each chunk's 1-based rank per list is kept as `<name>_rank`."""
    k = settings.rrf_k if k is None else k
    fused: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    for i, ranked in enumerate(ranked_lists):
        for rank, chunk in enumerate(ranked):
            key = chunk_id(chunk)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(key, chunk)
            ranks.setdefault(key, {})
            if names is not None:
                ranks[key][f"{names[i]}_rank"] = rank + 1
    ordered = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]
    return [
        first_seen[key].model_copy(
            update={
                "score": round(fused[key], 6),
                "metadata": {**first_seen[key].metadata, **ranks[key]},
            }
        )
        for key in ordered
    ]
