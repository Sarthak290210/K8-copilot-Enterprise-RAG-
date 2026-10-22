"""Tests for the eval modules: profiles, invokers, post-checks, reporting, diff, record, runner."""

from types import SimpleNamespace
from typing import cast

import pytest
from ragas.dataset_schema import SingleTurnSample

from eval import ragas_adapter, record
from eval.diff import diff_results, row_passed
from eval.invokers import ServiceInvoker, SkippedIntent
from eval.post_checks import _norm, forbidden_keywords_check, id_context_scores, source_overlap
from eval.profiles import PROFILES, get_profile
from eval.reporting import aggregate
from eval.run_ragas import (
    JudgeQuotaExhausted,
    checkpoint_path,
    collect_rows,
    load_checkpoint,
    needs_scoring,
    save_checkpoint,
    score_rows,
    select_goldens,
)
from eval.schema import load_goldens


def make_row(qid, faith, overlap, forbidden_ok=True, latency=100.0):
    return {
        "id": qid,
        "demonstrates_feature": "baseline",
        "metrics": {
            "faithfulness": faith,
            "context_precision": 0.5,
            "context_recall": None,
            "answer_relevancy": 0.9,
        },
        "source_overlap": {"overlap": overlap, "matched": []},
        "forbidden": {"passed": forbidden_ok, "hits": [] if forbidden_ok else ["x"]},
        "latency_ms": latency,
    }


# ---- profiles ----


def test_profiles_cover_plan():
    assert set(PROFILES) == {
        "naive",
        "sparse_only",
        "hybrid",
        "hybrid+rerank",
        "hybrid+rerank+hyde",
        "hybrid+rerank+crag",
        "all",
    }
    assert get_profile("naive") == {
        "search_mode": "dense",
        "top_k": 5,
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
    }
    assert all(get_profile("all")[k] for k in ("enable_rerank", "enable_hyde", "enable_crag"))


def test_get_profile_returns_a_copy():
    get_profile("naive")["top_k"] = 99
    assert PROFILES["naive"]["top_k"] == 5


def test_unknown_profile():
    with pytest.raises(KeyError, match="Unknown profile"):
        get_profile("nope")


# ---- invokers / runner ----


@pytest.mark.parametrize("intent", ["sql", "hybrid"])
async def test_service_invoker_skips_sql_and_hybrid_until_phase_4(intent):
    with pytest.raises(SkippedIntent, match="Phase 4"):
        await ServiceInvoker().invoke("How many pods?", get_profile("naive"), intent)


@pytest.mark.parametrize("intent", ["rag", "web_fallback"])
async def test_service_invoker_runs_rag(monkeypatch, intent):
    calls = []

    async def fake_run_rag_with_trace(question, flags):
        calls.append((question, flags))
        return "response", ["chunk"]

    monkeypatch.setattr("eval.invokers.rag_service.run_rag_with_trace", fake_run_rag_with_trace)
    result = await ServiceInvoker().invoke("What is a Pod?", {"top_k": 5}, intent)
    assert result == ("response", ["chunk"])
    assert calls == [("What is a Pod?", {"top_k": 5})]


async def test_collect_rows_records_skips_and_errors():
    goldens = load_goldens()[:3]

    class FakeInvoker:
        async def invoke(self, question, flags, intent):
            if question == goldens[0].question:
                raise SkippedIntent("not yet")
            if question == goldens[1].question:
                raise RuntimeError("boom")
            chunk = SimpleNamespace(
                text="Pods are the smallest deployable units.", source="pods.html"
            )
            response = SimpleNamespace(
                answer="A pod...", sources=["seed/docs/true_data/" + goldens[2].golden_sources[0]]
            )
            return response, [chunk]

    rows, skipped, errors = await collect_rows(goldens, {}, FakeInvoker())  # type: ignore[arg-type]
    assert skipped == [{"id": goldens[0].id, "reason": "not yet"}]
    assert errors == [{"id": goldens[1].id, "error": "RuntimeError: boom"}]
    assert rows[0]["contexts"] == ["Pods are the smallest deployable units."]
    assert rows[0]["source_overlap"]["overlap"] == pytest.approx(
        1 / len(goldens[2].golden_sources), abs=1e-3
    )


async def test_collect_rows_reuses_done_rows_and_reports_each_new_one():
    goldens = [g for g in load_goldens() if g.intent == "rag"][:3]
    asked = []

    class FakeInvoker:
        async def invoke(self, question, flags, intent):
            asked.append(question)
            return SimpleNamespace(answer="a", sources=[]), []

    done = {goldens[0].id: {"id": goldens[0].id, "answer": "cached"}}
    progress = []
    rows, _, _ = await collect_rows(
        goldens,
        {},
        FakeInvoker(),  # type: ignore[arg-type]
        done=done,
        on_row=lambda r: progress.append(len(r)),
    )
    assert asked == [goldens[1].question, goldens[2].question]
    assert [r["id"] for r in rows] == [g.id for g in goldens]
    assert rows[0]["answer"] == "cached"
    assert progress == [2, 3]


def test_checkpoint_round_trip_and_key_mismatch(tmp_path):
    path = tmp_path / "naive.json"
    key = {"profile": "naive", "flags": {"top_k": 5}}
    save_checkpoint(path, key, [{"id": "q-001", "answer": "x"}])
    assert load_checkpoint(path, key) == {"q-001": {"id": "q-001", "answer": "x"}}
    assert load_checkpoint(path, {**key, "flags": {"top_k": 10}}) == {}
    assert load_checkpoint(tmp_path / "missing.json", key) == {}
    assert checkpoint_path("hybrid+rerank", None).name == "hybrid-rerank.json"
    assert checkpoint_path("all", "security").name == "all_security.json"


GOOD = {"faithfulness": 1.0, "answer_relevancy": 0.9}  # what the LLM judge returns
EMPTY = dict.fromkeys(GOOD)


def test_needs_scoring():
    full = {**GOOD, "context_precision": 1.0, "context_recall": 1.0}
    no_sources = {**GOOD, "context_precision": None, "context_recall": None}
    assert needs_scoring({}, "j")
    assert needs_scoring({"metrics": full, "judge_model": "other"}, "j")
    assert needs_scoring({"metrics": {**full, "faithfulness": None}, "judge_model": "j"}, "j")
    assert not needs_scoring({"metrics": full, "judge_model": "j"}, "j")
    # ID-based metrics are None by design without golden sources: no endless re-judging
    assert not needs_scoring({"metrics": no_sources, "judge_model": "j"}, "j")


def test_score_rows_skips_scored_rows_and_adds_id_metrics():
    full = {**GOOD, "context_precision": 1.0, "context_recall": 1.0}
    rows = [
        {"id": "a", "metrics": full, "judge_model": "j"},
        {"id": "b", "golden_sources": ["pods.html"], "context_sources": ["pods.html", "x.pdf"]},
        {"id": "c", "golden_sources": [], "context_sources": ["x.pdf"]},
    ]
    judged, saves = [], []

    def judge(batch, model):
        judged.append(batch[0]["id"])
        return [GOOD]

    score_rows(rows, "j", on_scored=lambda: saves.append(1), run_judge=judge)
    assert judged == ["b", "c"]
    assert len(saves) == 2
    assert rows[1]["metrics"] == {**GOOD, "context_precision": 1.0, "context_recall": 1.0}
    assert rows[2]["metrics"] == {**GOOD, "context_precision": None, "context_recall": None}
    assert all(r["judge_model"] == "j" for r in rows)


def test_score_rows_stops_when_judge_returns_nothing():
    rows = [{"id": x, "golden_sources": ["a.pdf"], "context_sources": []} for x in "abc"]
    results = iter([[GOOD], [EMPTY]])
    with pytest.raises(JudgeQuotaExhausted, match="b"):
        score_rows(rows, "j", run_judge=lambda batch, model: next(results))
    assert rows[0]["metrics"]["faithfulness"] == 1.0
    assert "metrics" not in rows[1] and "metrics" not in rows[2]


def test_id_context_scores():
    golden = ["pods.html", "Deployments.pdf"]
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2; both golden files found
    scores = id_context_scores(["pods.html", "noise.pdf", "seed/docs/deployments.PDF"], golden)
    assert scores == {"context_precision": 0.8333, "context_recall": 1.0}
    # reranking that moves the relevant chunk up raises precision; recall unchanged
    low = id_context_scores(["noise.pdf", "pods.html"], golden)
    high = id_context_scores(["pods.html", "noise.pdf"], golden)
    assert low["context_precision"] == 0.5 and high["context_precision"] == 1.0
    assert low["context_recall"] == high["context_recall"] == 0.5
    assert id_context_scores(["noise.pdf"], golden) == {
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    none = {"context_precision": None, "context_recall": None}
    assert id_context_scores(["pods.html"], []) == none  # security / decline goldens
    assert id_context_scores([], golden) == none


def test_filter_keeps_feature_plus_baseline():
    selected = select_goldens(load_goldens(), "hyde")
    assert {g.demonstrates_feature for g in selected} == {"hyde", "baseline"}


# ---- post checks ----


def test_forbidden_keywords_case_insensitive():
    assert forbidden_keywords_check("Here is your KUBECONFIG", ["kubeconfig"]) == {
        "passed": False,
        "hits": ["kubeconfig"],
    }
    assert forbidden_keywords_check("Use kubectl logs", ["kubeconfig"])["passed"]


def test_norm_strips_paths_and_case():
    assert _norm("seed/docs/true_data/Pods.HTML") == "pods.html"
    assert _norm(r"C:\docs\Pods.html") == "pods.html"


def test_source_overlap():
    result = source_overlap(["seed/docs/a.pdf", "x.txt"], ["a.pdf", "b.pdf"])
    assert result == {"overlap": 0.5, "matched": ["a.pdf"]}
    assert source_overlap(["a.pdf"], [])["overlap"] is None


# ---- reporting ----


def test_aggregate_ignores_missing_scores():
    rows = [
        make_row("q-001", 1.0, 1.0),
        make_row("q-002", 0.5, None, forbidden_ok=False, latency=300.0),
    ]
    agg = aggregate(rows, skipped=[{"id": "q-003", "reason": "x"}], errors=[])
    assert agg["n_evaluated"] == 2 and agg["n_skipped"] == 1 and agg["n_errors"] == 0
    assert agg["faithfulness"] == 0.75
    assert agg["context_recall"] is None  # all rows missing -> None, never invented
    assert agg["source_overlap"] == 1.0
    assert agg["forbidden_violations"] == 1
    assert agg["latency_ms"] == 200.0


def test_aggregate_of_empty_run():
    agg = aggregate([], skipped=[{"id": "q-001", "reason": "x"}], errors=[])
    assert agg["n_evaluated"] == 0 and agg["faithfulness"] is None


# ---- diff ----


def test_row_passed():
    assert row_passed(make_row("q", 0.7, 0.5))
    assert not row_passed(make_row("q", 0.69, 0.5))
    assert not row_passed(make_row("q", 0.9, 0.0))
    assert row_passed(make_row("q", 0.9, None))  # no golden sources (e.g. security)


def test_diff_results():
    a = {
        "aggregate": {"faithfulness": 0.6, "latency_ms": 100.0},
        "rows": [make_row("q-1", 0.5, 1.0), make_row("q-2", 0.9, 1.0)],
    }
    b = {
        "aggregate": {"faithfulness": 0.8, "latency_ms": 150.0},
        "rows": [make_row("q-1", 0.9, 1.0), make_row("q-2", 0.1, 1.0)],
    }
    d = diff_results(a, b)
    assert d["metrics"]["faithfulness"] == {"a": 0.6, "b": 0.8, "delta": 0.2}
    assert d["metrics"]["context_recall"]["delta"] is None
    assert d["fail_to_pass"] == ["q-1"] and d["pass_to_fail"] == ["q-2"]


# ---- record ----

PAYLOAD = {
    "profile": "naive",
    "filter": None,
    "timestamp": "20261022T101500Z",
    "aggregate": {
        "n_evaluated": 30,
        "n_skipped": 14,
        "faithfulness": 0.8123,
        "context_precision": 0.7,
        "context_recall": None,
        "answer_relevancy": 0.9,
        "source_overlap": 0.5,
        "forbidden_violations": 0,
        "latency_ms": 812.4,
    },
}


def test_build_log_row():
    row = record.build_log_row(PAYLOAD, 1, "BASELINE | naive", "eval/results/x_naive.json")
    assert row.startswith("| 2026-10-22 10:15 | 1 | naive | 30/14 | 0.812 | 0.700 | - | 0.900 |")
    assert "BASELINE \\| naive" in row  # pipes escaped so the table doesn't break
    assert "`eval/results/x_naive.json`" in row


def test_append_inserts_above_end_marker_and_keeps_old_rows(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text(
        f"intro\n{record.START_MARKER}\n| header |\n| old row |\n{record.END_MARKER}\noutro\n",
        encoding="utf-8",
    )
    record.append_to_claude_md("| new row |", md)
    text = md.read_text(encoding="utf-8")
    assert text.index("| old row |") < text.index("| new row |") < text.index(record.END_MARKER)
    assert text.endswith("outro\n")


def test_append_requires_markers(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EVAL-LOG markers"):
        record.append_to_claude_md("| row |", md)


# ---- ragas adapter (no network: dataset building and NaN handling only) ----


def test_build_dataset_and_reference():
    ds = ragas_adapter.build_dataset(
        [
            {
                "question": "What is a Pod?",
                "answer": "A pod...",
                "contexts": ["ctx"],
                "golden_answer_keywords": ["pod", "container"],
            },
            {"question": "poem", "answer": "no", "contexts": [], "golden_answer_keywords": []},
        ]
    )
    first, second = (cast(SingleTurnSample, s) for s in ds.samples)
    assert (first.user_input, first.response, first.retrieved_contexts) == (
        "What is a Pod?",
        "A pod...",
        ["ctx"],
    )
    assert first.reference == "pod, container"
    assert (second.reference or "").startswith("The assistant should decline")


def test_clean_turns_nan_into_none():
    assert ragas_adapter._clean(float("nan")) is None
    assert ragas_adapter._clean(None) is None
    assert ragas_adapter._clean(0.123456) == 0.1235
