"""Summarise an eval run: aggregate numbers and a terminal table."""

from statistics import mean
from typing import Any

METRIC_KEYS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(mean(present), 3) if present else None


def aggregate(
    rows: list[dict[str, Any]], skipped: list[dict[str, Any]], errors: list[dict[str, Any]]
) -> dict[str, Any]:
    """Means of each Ragas metric, forbidden violations, mean source overlap and latency."""
    summary: dict[str, Any] = {
        "n_evaluated": len(rows),
        "n_skipped": len(skipped),
        "n_errors": len(errors),
    }
    for key in METRIC_KEYS:
        summary[key] = _mean([row["metrics"].get(key) for row in rows])
    summary["source_overlap"] = _mean([row["source_overlap"]["overlap"] for row in rows])
    summary["forbidden_violations"] = sum(not row["forbidden"]["passed"] for row in rows)
    summary["latency_ms"] = _mean([row["latency_ms"] for row in rows])
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_table(payload: dict[str, Any]) -> None:
    """Markdown table: one line per evaluated question, then the aggregate."""
    print(f"\n## Eval: profile `{payload['profile']}` ({payload['timestamp']})\n")
    header = ["id", "feature", "faith", "ctx_prec", "ctx_rec", "ans_rel", "overlap", "forbid", "ms"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in payload["rows"]:
        m = row["metrics"]
        cells = [
            row["id"],
            row["demonstrates_feature"],
            *(_fmt(m.get(k)) for k in METRIC_KEYS),
            _fmt(row["source_overlap"]["overlap"]),
            "ok" if row["forbidden"]["passed"] else "HIT",
            _fmt(row["latency_ms"]),
        ]
        print("| " + " | ".join(cells) + " |")

    agg = payload["aggregate"]
    print(
        f"\n**Aggregate** — evaluated {agg['n_evaluated']}, skipped {agg['n_skipped']}, "
        f"errors {agg['n_errors']} | "
        + " | ".join(f"{k} {_fmt(agg[k])}" for k in METRIC_KEYS)
        + f" | source_overlap {_fmt(agg['source_overlap'])}"
        + f" | forbidden {agg['forbidden_violations']} | latency_ms {_fmt(agg['latency_ms'])}"
    )
    if payload["skipped"]:
        reasons = sorted({s["reason"] for s in payload["skipped"]})
        print(f"Skipped reasons: {'; '.join(reasons)}")
