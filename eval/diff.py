"""Compare two eval runs.

uv run python -m eval.diff eval/results/A.json eval/results/B.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

COMPARED_KEYS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "source_overlap",
    "forbidden_violations",
    "latency_ms",
]
PASS_FAITHFULNESS = 0.7


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_passed(row: dict[str, Any]) -> bool:
    """Pass = faithfulness >= 0.7 and at least one golden source cited (if it has any)."""
    faith = row["metrics"].get("faithfulness")
    overlap = row["source_overlap"]["overlap"]
    return faith is not None and faith >= PASS_FAITHFULNESS and (overlap is None or overlap > 0)


def diff_results(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for key in COMPARED_KEYS:
        va, vb = a["aggregate"].get(key), b["aggregate"].get(key)
        delta = round(vb - va, 3) if va is not None and vb is not None else None
        metrics[key] = {"a": va, "b": vb, "delta": delta}

    passed_a = {row["id"]: row_passed(row) for row in a["rows"]}
    passed_b = {row["id"]: row_passed(row) for row in b["rows"]}
    common = sorted(passed_a.keys() & passed_b.keys())
    return {
        "metrics": metrics,
        "fail_to_pass": [q for q in common if not passed_a[q] and passed_b[q]],
        "pass_to_fail": [q for q in common if passed_a[q] and not passed_b[q]],
    }


def _arrow(key: str, delta: float | None) -> str:
    if not delta:
        return ""
    # Lower is better for these two; higher is better for everything else.
    better = delta < 0 if key in ("forbidden_violations", "latency_ms") else delta > 0
    return "▲" if better else "▼"


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval result files.")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()

    a, b = load_result(args.a), load_result(args.b)
    result = diff_results(a, b)
    print(f"\n## `{a['profile']}` ({args.a.name}) → `{b['profile']}` ({args.b.name})\n")
    print("| metric | A | B | delta |")
    print("|---|---|---|---|")
    for key, m in result["metrics"].items():
        print(
            f"| {key} | {_fmt(m['a'])} | {_fmt(m['b'])} | {_fmt(m['delta'])} {_arrow(key, m['delta'])} |"
        )
    print(f"\nFlipped fail → pass: {', '.join(result['fail_to_pass']) or 'none'}")
    print(f"Flipped pass → fail: {', '.join(result['pass_to_fail']) or 'none'}")


if __name__ == "__main__":
    main()
