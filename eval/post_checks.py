"""Cheap, deterministic checks that run beside Ragas."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


def forbidden_keywords_check(answer: str, forbidden: list[str]) -> dict[str, Any]:
    """Fail if the answer contains any forbidden phrase (case-insensitive)."""
    lowered = answer.lower()
    hits = [word for word in forbidden if word.lower() in lowered]
    return {"passed": not hits, "hits": hits}


def _norm(source: str) -> str:
    """Lowercase file name without its directory, so 'seed/docs/x/Pods.HTML' == 'pods.html'."""
    name = PureWindowsPath(source).name if "\\" in source else PurePosixPath(source).name
    return name.strip().lower()


ID_METRICS = ("context_precision", "context_recall")


def id_context_scores(context_sources: list[str], golden: list[str]) -> dict[str, float | None]:
    """context_precision / context_recall from file names instead of an LLM judge (0 tokens).

    A retrieved chunk is relevant when its source file is a golden source.
    - precision: rank-aware like Ragas' context_precision (mean of precision@k at each relevant
      rank), so moving relevant chunks up (reranking) raises it.
    - recall: fraction of golden sources that appear among the retrieved chunks.
    Both are None when the golden has no sources (security/decline questions) or nothing was
    retrieved, so they don't drag the averages down.
    """
    if not golden or not context_sources:
        return dict.fromkeys(ID_METRICS)
    golden_norm = {_norm(s) for s in golden}
    relevant = [_norm(s) in golden_norm for s in context_sources]
    hits = 0
    precision_sum = 0.0
    for k, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / k
    precision = precision_sum / hits if hits else 0.0
    found = golden_norm & {_norm(s) for s in context_sources}
    return {
        "context_precision": round(precision, 4),
        "context_recall": round(len(found) / len(golden_norm), 4),
    }


def source_overlap(actual: list[str], golden: list[str]) -> dict[str, Any]:
    """Fraction of golden sources found in the answer's sources (None when there are none)."""
    if not golden:
        return {"overlap": None, "matched": []}
    actual_norm = {_norm(s) for s in actual}
    matched = [s for s in golden if _norm(s) in actual_norm]
    return {"overlap": round(len(matched) / len(golden), 3), "matched": matched}
