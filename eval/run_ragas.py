"""The eval command.

    uv run python -m eval.run_ragas --profile naive [--filter hyde] [--questions path] [--output path]

Writes eval/results/<UTC timestamp>_<profile>.json and prints a summary table.

Progress is checkpointed to eval/results/checkpoints/ after every answered and every scored
question, so a run stopped by the LLM provider's daily token limit (or anything else) loses nothing:
run the same command again and it resumes. The checkpoint is deleted once the result is written.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import settings
from app.services.rag_service import pipeline_settings
from app.services.web_search import is_web_chunk
from eval.invokers import AgentInvoker, Invoker, ServiceInvoker, SkippedIntent
from eval.post_checks import (
    ID_METRICS,
    forbidden_keywords_check,
    id_context_scores,
    intent_match,
    source_overlap,
    sql_keyword_match,
)
from eval.profiles import get_profile
from eval.reporting import aggregate, print_table
from eval.schema import DEFAULT_GOLDENS_PATH, Golden, load_goldens

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
WEB_SOURCE = "tavily_web"  # golden_sources marker for "answered from the web"


class JudgeQuotaExhausted(RuntimeError):
    """Every metric of a question came back empty: almost always the provider's daily token limit."""


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def select_goldens(goldens: list[Golden], feature: str | None) -> list[Golden]:
    """--filter keeps that feature's goldens plus the baseline ones."""
    if feature is None:
        return goldens
    return [g for g in goldens if g.demonstrates_feature in (feature, "baseline")]


def checkpoint_path(profile: str, feature: str | None, cache_mode: str = "on") -> Path:
    """One file per profile/filter, and per cache mode: E5's cold run (cache cleared first) and
    a --no-cache run never resume, or get resumed by, a normal (warm) run."""
    name = profile if feature is None else f"{profile}_{feature}"
    if cache_mode != "on":
        name = f"{name}_{cache_mode}"
    return CHECKPOINT_DIR / f"{name.replace('+', '-')}.json"


def cache_mode(clear_cache: bool, no_cache: bool) -> str:
    return "off" if no_cache else "cold" if clear_cache else "on"


def load_checkpoint(path: Path, key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Answered rows by id, if the checkpoint belongs to the same run (same key)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != key:
        logger.warning("Ignoring checkpoint {}: it was made with other settings", path.name)
        return {}
    logger.info("Resuming from checkpoint {} ({} answered)", path.name, len(data["rows"]))
    return {row["id"]: row for row in data["rows"]}


def save_checkpoint(path: Path, key: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "rows": rows}, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a kill mid-write never leaves a broken checkpoint


async def collect_rows(
    goldens: list[Golden],
    flags: dict[str, Any],
    invoker: Invoker,
    done: dict[str, dict[str, Any]] | None = None,
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask every golden. Returns (rows to score, skipped, errors).

    Goldens already in `done` are reused, not asked again; `on_row` gets the rows so far
    after each new answer (for checkpointing). Errors are not checkpointed: they're retried.
    """
    done = done or {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for golden in goldens:
        if golden.id in done:
            rows.append(done[golden.id])
            continue
        start = time.perf_counter()
        try:
            response, chunks = await invoker.invoke(golden.question, flags, golden.intent)
        except SkippedIntent as exc:
            skipped.append({"id": golden.id, "reason": str(exc)})
            continue
        except Exception as exc:
            # Recorded, never hidden: a crash on one question shouldn't stop the run.
            logger.exception("Question {} failed", golden.id)
            errors.append({"id": golden.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # Web results (CRAG) count as the golden source "tavily_web", whatever their URL.
        eval_sources = [WEB_SOURCE if is_web_chunk(c) else c.source for c in chunks]
        rows.append(
            {
                "id": golden.id,
                "question": golden.question,
                "intent": golden.intent,
                "demonstrates_feature": golden.demonstrates_feature,
                "golden_sources": golden.golden_sources,
                "golden_answer_keywords": golden.golden_answer_keywords,
                "answer": response.answer,
                "sources": list(response.sources),
                "contexts": [chunk.text for chunk in chunks],
                "context_sources": eval_sources,  # in rank order
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "forbidden": forbidden_keywords_check(response.answer, golden.forbidden_keywords),
                "source_overlap": source_overlap(eval_sources, golden.golden_sources),
                # Phase 3 trace: cost and what the advanced features did.
                "tokens_used": response.metadata.tokens_used,
                "confidence": response.confidence,
                "crag_action": response.metadata.crag.action if response.metadata.crag else None,
                "reflection_iterations": response.metadata.reflection_iterations,
                "hyde_hypotheses": response.metadata.hyde_hypotheses,
                "refined_question": response.metadata.refined_question,
                # Phase 4: routing and Text2SQL
                "predicted_intent": response.metadata.intent,
                "intent_match": intent_match(golden.intent, response.metadata.intent),
                "generated_sql": response.metadata.generated_sql,
                "sql_rows_count": response.metadata.sql_row_count,
                "sql_keyword_match": sql_keyword_match(
                    golden.intent,
                    golden.golden_answer_keywords,
                    response.answer,
                    response.metadata.generated_sql,
                ),
                # Phase 5: which cache tiers answered, and the LLM spend they skipped.
                "cache_hits": dict(response.metadata.cache_hits),
                "cost_saved_usd": response.metadata.cost_saved_usd,
            }
        )
        if on_row is not None:
            on_row(rows)
    return rows, skipped, errors


async def _collect_and_close(
    goldens: list[Golden],
    flags: dict[str, Any],
    done: dict[str, dict[str, Any]],
    on_row: Callable[[list[dict[str, Any]]], None],
    rag_only: bool = False,
    clear_cache: bool = False,
    cache_report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask all goldens in one event loop, then close the shared clients (they're loop-bound).

    Default: the Phase 4 agent (routing + Text2SQL, SQL auto-approved); rag_only: the RAG
    pipeline alone (sql/hybrid goldens are skipped), as in E1-E3. clear_cache empties every
    cache tier (Redis + memory) first: E5's cold run. `cache_report` receives this process's
    per-tier cache stats (the API's /admin/cache/stats can't see an in-process eval)."""
    from app.middleware import rate_limiter
    from app.services import llm_service, vector_store, web_search
    from app.services.query_cache_service import query_cache

    try:
        if clear_cache:
            logger.info("Cache cleared before the run: {}", await query_cache.clear())
        invoker = ServiceInvoker() if rag_only else AgentInvoker()
        return await collect_rows(goldens, flags, invoker, done, on_row)
    finally:
        if cache_report is not None:
            cache_report.update(query_cache.stats())
        await llm_service.get_client().close()
        await vector_store.get_client().close()
        tavily = web_search.get_client()
        if tavily is not None:
            await tavily.close()
        redis = rate_limiter.get_redis_client()
        if redis is not None:
            await redis.close()
        llm_service.get_client.cache_clear()
        vector_store.get_client.cache_clear()
        web_search.get_client.cache_clear()
        rate_limiter.get_redis_client.cache_clear()


def needs_scoring(row: dict[str, Any], judge_model: str) -> bool:
    """Unscored, scored by another judge, or with a failed (None) LLM metric -> score (again).

    The ID-based metrics are None by design for questions without golden sources: ignored.
    """
    metrics = row.get("metrics")
    if not metrics or row.get("judge_model") != judge_model:
        return True
    return any(value is None for key, value in metrics.items() if key not in ID_METRICS)


def score_rows(
    rows: list[dict[str, Any]],
    judge_model: str,
    on_scored: Callable[[], None] | None = None,
    run_judge: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]] | None = None,
) -> None:
    """Attach Ragas metrics to each row (in place), one question at a time.

    `on_scored` runs after each question (for checkpointing). If every metric of a question
    fails, the judge is out of quota: stop with JudgeQuotaExhausted instead of grinding
    through the rest with nothing but failures.
    """
    if run_judge is None:
        from eval import ragas_adapter  # lazy: ragas is slow to import

        run_judge = ragas_adapter.run
    todo = [row for row in rows if needs_scoring(row, judge_model)]
    logger.info("Scoring {} of {} rows with judge {}", len(todo), len(rows), judge_model)
    for n, row in enumerate(todo, start=1):
        (metrics,) = run_judge([row], judge_model)
        if all(value is None for value in metrics.values()):
            raise JudgeQuotaExhausted(row["id"])
        row["metrics"] = {
            **metrics,
            **id_context_scores(row.get("context_sources", []), row["golden_sources"]),
        }
        row["judge_model"] = judge_model
        logger.info("Scored {} ({}/{})", row["id"], n, len(todo))
        if on_scored is not None:
            on_scored()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ragas eval for one profile.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_GOLDENS_PATH)
    parser.add_argument("--filter", dest="feature", default=None)
    parser.add_argument("--output", type=Path, default=None)
    # Default judge: settings.eval_judge_model (EVAL_JUDGE_MODEL). Keep it the same across all
    # checkpoints so scores stay comparable; recorded in the result file.
    parser.add_argument("--judge-model", default=None)

    parser.add_argument(
        "--fresh", action="store_true", help="ignore any checkpoint and start from scratch"
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="RAG pipeline only, no agent/routing (the E1-E3 path; sql/hybrid goldens skipped)",
    )
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--clear-cache",
        action="store_true",
        help="empty every cache tier (Redis + memory) before asking: a cold run (E5)",
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        help="skip the cache entirely (to measure retrieval/answer changes, not cached answers)",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="write the result even if some questions errored (they're listed in it)",
    )
    args = parser.parse_args()

    if args.no_cache:
        settings.cache_enabled = False
    flags = get_profile(args.profile)
    goldens = select_goldens(load_goldens(args.questions), args.feature)
    judge_model = args.judge_model or settings.eval_judge_model
    logger.info("Profile {} | {} goldens | flags {}", args.profile, len(goldens), flags)

    # Answers depend on these; the judge is stored per row, so changing it only rescores.
    key = {
        "profile": args.profile,
        "flags": flags,
        "questions_file": args.questions.name,
        "filter": args.feature,
        "answer_model": settings.llm_model_answer,
        "pipeline": pipeline_settings(),
        "invoker": "rag_only" if args.rag_only else "agent",
        "cache": cache_mode(args.clear_cache, args.no_cache),
    }
    ckpt = checkpoint_path(args.profile, args.feature, key["cache"])
    done = {} if args.fresh else load_checkpoint(ckpt, key)
    if args.clear_cache and done:
        # Resuming a cold run: the cache was cleared when it started, and clearing it again
        # would delete the answers the rows so far cached (E5's warm run needs them).
        logger.info("Resuming a cold run: cache not cleared again")

    cache_stats: dict[str, Any] = {}
    rows, skipped, errors = asyncio.run(
        _collect_and_close(
            goldens,
            flags,
            done,
            lambda r: save_checkpoint(ckpt, key, r),
            args.rag_only,
            clear_cache=args.clear_cache and not done,
            cache_report=cache_stats,
        ),
        # async psycopg (Text2SQL) can't use Windows' default Proactor loop
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
    save_checkpoint(ckpt, key, rows)
    try:
        score_rows(rows, judge_model, on_scored=lambda: save_checkpoint(ckpt, key, rows))
    except JudgeQuotaExhausted as exc:
        scored = sum(not needs_scoring(row, judge_model) for row in rows)
        logger.error(
            "Judge {} returned no scores for {} (daily token limit?). {}/{} rows scored; "
            "progress saved to {}. Run the same command again later to resume.",
            judge_model,
            exc,
            scored,
            len(rows),
            ckpt.as_posix(),
        )
        raise SystemExit(2) from None
    if errors and not args.allow_errors:
        # A result missing questions isn't comparable with other runs: keep the checkpoint
        # and let a re-run retry only the errored ones.
        logger.error(
            "{} question(s) errored ({}); no result written. Progress saved to {}. "
            "Run the same command again to retry them (or pass --allow-errors).",
            len(errors),
            ", ".join(e["id"] for e in errors),
            ckpt.as_posix(),
        )
        raise SystemExit(3)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "profile": args.profile,
        "flags": flags,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "questions_file": args.questions.name,
        "filter": args.feature,
        "judge_model": judge_model,
        "answer_relevancy_strictness": settings.eval_relevancy_strictness,
        "answer_model": settings.llm_model_answer,
        "pipeline": pipeline_settings(),
        "invoker": "rag_only" if args.rag_only else "agent",
        # Phase 5: cache state of this run (stats count only the questions asked this time;
        # rows reused from a checkpoint keep their own cache_hits).
        "cache": {
            "enabled": settings.cache_enabled,
            "cleared_before_run": args.clear_cache,
            "stats": cache_stats,
        },
        "rows": rows,
        "skipped": skipped,
        "errors": errors,
        "aggregate": aggregate(rows, skipped, errors),
    }
    output = args.output or RESULTS_DIR / f"{timestamp}_{args.profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ckpt.unlink(missing_ok=True)

    print_table(payload)
    print(f"\nWrote {output.as_posix()}")


if __name__ == "__main__":
    main()
