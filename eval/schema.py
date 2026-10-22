"""Load and validate the golden questions file.

Run `uv run python -m eval.schema` to validate eval/golden_questions.yaml and print counts.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Literal, get_args

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_GOLDENS_PATH = Path(__file__).resolve().parent / "golden_questions.yaml"

Intent = Literal["rag", "sql", "hybrid", "web_fallback"]
Feature = Literal[
    "baseline",
    "sparse",
    "dense",
    "hybrid",
    "rerank",
    "hyde",
    "crag",
    "self_rag",
    "sql",
    "hybrid_rag_sql",
    "security",
    "wild",
]
Outcome = Literal["pass", "fail"]

# Non-file sources: SQL rows and Tavily web results.
SPECIAL_SOURCES = frozenset({"query_results", "tavily_web"})


class Golden(BaseModel):
    """One golden question with its known-good sources and answer keywords."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q-\d{3}$")
    question: str = Field(min_length=1)
    intent: Intent
    golden_sources: list[str]
    golden_answer_keywords: list[str]
    demonstrates_feature: Feature
    expected_baseline: Outcome
    expected_with_feature: Outcome
    notes: str = ""
    forbidden_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_expected_with_feature(self) -> "Golden":
        # A golden exists to show a feature working, so it must pass with that feature on.
        if self.expected_with_feature != "pass":
            raise ValueError("expected_with_feature must be 'pass'")
        # Only security goldens (refusals) have no doc to cite and no answer keywords.
        if self.demonstrates_feature != "security" and not (
            self.golden_sources and self.golden_answer_keywords
        ):
            raise ValueError("only security goldens may have empty sources or keywords")
        return self


def load_goldens(path: Path = DEFAULT_GOLDENS_PATH) -> list[Golden]:
    """Read and validate the YAML. Raises ValueError on a bad entry or duplicate IDs."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a YAML list of goldens")

    goldens: list[Golden] = []
    for index, entry in enumerate(raw):
        try:
            goldens.append(Golden.model_validate(entry))
        except ValidationError as exc:
            label = entry.get("id", f"entry #{index}") if isinstance(entry, dict) else index
            raise ValueError(f"Invalid golden {label}: {exc}") from exc

    duplicates = [gid for gid, n in Counter(g.id for g in goldens).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate golden IDs: {sorted(duplicates)}")

    covered = {g.demonstrates_feature for g in goldens}
    for feature in get_args(Feature):
        if feature not in covered:
            logger.warning("No golden questions for feature '{}'", feature)
    return goldens


def main() -> None:
    try:
        goldens = load_goldens()
    except (OSError, ValueError) as exc:
        print(f"INVALID: {exc}")
        sys.exit(1)

    print(f"OK: {len(goldens)} goldens in {DEFAULT_GOLDENS_PATH.name}\n")
    print("By feature:")
    for feature, n in sorted(Counter(g.demonstrates_feature for g in goldens).items()):
        print(f"  {feature:<16} {n}")
    print("By intent:")
    for intent, n in sorted(Counter(g.intent for g in goldens).items()):
        print(f"  {intent:<16} {n}")


if __name__ == "__main__":
    main()
