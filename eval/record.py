"""Append an eval result to the Eval Results Log in CLAUDE.md.

uv run python -m eval.record eval/results/<file>.json --phase 1 --note "first baseline"
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.diff import load_result

START_MARKER = "<!-- EVAL-LOG:START -->"
END_MARKER = "<!-- EVAL-LOG:END -->"
DEFAULT_CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    text = f"{value:.3f}" if isinstance(value, float) else str(value)
    return text.replace("|", "\\|")  # a pipe would break the markdown table


def build_log_row(payload: dict[str, Any], phase: int, note: str, result_file: str) -> str:
    agg = payload["aggregate"]
    run_date = datetime.strptime(payload["timestamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    cells = [
        run_date.strftime("%Y-%m-%d %H:%M"),
        phase,
        payload["profile"]
        + (f" (filter: {payload['filter']})" if payload.get("filter") else "")
        + (
            f" [judge: {payload['judge_model']}]" if "20b" in payload.get("judge_model", "") else ""
        ),
        f"{agg['n_evaluated']}/{agg['n_skipped']}",
        agg["faithfulness"],
        agg["context_precision"],
        agg["context_recall"],
        agg["answer_relevancy"],
        agg["source_overlap"],
        agg["forbidden_violations"],
        agg["latency_ms"],
        f"`{result_file}`",
        note,
    ]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def append_to_claude_md(row: str, path: Path = DEFAULT_CLAUDE_MD) -> None:
    """Insert `row` just above the END marker. Never touches existing rows."""
    text = path.read_text(encoding="utf-8")
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{path} is missing the EVAL-LOG markers (or they're out of order)")
    if text.count(END_MARKER) != 1:
        raise ValueError(f"{path} must contain exactly one {END_MARKER}")
    path.write_text(text[:end] + row + "\n" + text[end:], encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an eval result in CLAUDE.md.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--claude-md", type=Path, default=DEFAULT_CLAUDE_MD)
    args = parser.parse_args()

    row = build_log_row(load_result(args.result), args.phase, args.note, _display_path(args.result))
    append_to_claude_md(row, args.claude_md)
    print(f"Recorded in {args.claude_md.name}:\n{row}")


if __name__ == "__main__":
    main()
