"""Pure helpers of the Streamlit UI (no browser, no running API)."""

import json

import httpx

from scripts import streamlit_app as ui


def test_api_headers():
    assert ui._api_headers("tok") == {"Authorization": "Bearer tok"}
    assert ui._api_headers(None) == {}


def test_safe_json():
    assert ui._safe_json(httpx.Response(200, json={"a": 1})) == {"a": 1}
    assert ui._safe_json(httpx.Response(200, json=[1])) == {"data": [1]}
    assert ui._safe_json(httpx.Response(500, text="oops")) == {"raw": "oops"}


# ---- demo questions ----


def test_every_golden_has_a_technique():
    """A new demonstrates_feature tag needs a TECHNIQUES entry, or the demo shows a bare tag."""
    goldens = ui._read_goldens(ui.GOLDENS_PATH)
    assert len(goldens) >= 45
    assert {g["demonstrates_feature"] for g in goldens} <= set(ui.TECHNIQUES)
    assert set(ui.SECURITY_LAYERS) <= {g["id"] for g in goldens}


def test_read_goldens_missing_or_broken_file(tmp_path):
    assert ui._read_goldens(tmp_path / "missing.yaml") == []
    broken = tmp_path / "broken.yaml"
    broken.write_text("- question: [unclosed", encoding="utf-8")
    assert ui._read_goldens(broken) == []
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text("- question: What is a Pod?\n- just a string\n- {id: q-x}\n", "utf-8")
    assert ui._read_goldens(mixed) == [{"question": "What is a Pod?"}]


def test_technique_text():
    name, text, flags = ui._technique({"id": "q-014", "demonstrates_feature": "hyde"})
    assert name == "HyDE" and flags == {"enable_hyde": True} and text
    # security goldens explain their own layer
    _, text, _ = ui._technique({"id": "q-041", "demonstrates_feature": "security"})
    assert "Prompt injection" in text
    assert ui._technique({"demonstrates_feature": "new_tag"}) == ("new_tag", "", {})


def test_group_by_technique_keeps_the_technique_order():
    goldens = [
        {"id": "a", "demonstrates_feature": "security"},
        {"id": "b", "demonstrates_feature": "baseline"},
        {"id": "c", "demonstrates_feature": "unknown"},
        {"id": "d", "demonstrates_feature": "baseline"},
    ]
    groups = ui._group_by_technique(goldens)
    assert list(groups) == ["baseline", "security", "unknown"]
    assert [g["id"] for g in groups["baseline"]] == ["b", "d"]


def test_demo_flags_start_from_the_defaults():
    defaults = {"search_mode": "hybrid", "top_k": 5, "enable_crag": True}
    hyde = ui._demo_flags({"demonstrates_feature": "hyde"}, defaults)
    assert hyde == {
        "search_mode": "hybrid",
        "top_k": 5,
        "enable_rerank": False,
        "enable_hyde": True,
        "enable_crag": True,
        "enable_self_reflective": False,
    }
    # a technique with no flags of its own resets to the defaults
    plain = ui._demo_flags({"demonstrates_feature": "baseline"}, defaults)
    assert not plain["enable_hyde"] and plain["enable_crag"]


# ---- eval dashboard ----


def row(qid, faith, overlap: float | None = 1.0, forbidden_ok=True, **extra):
    return {
        "id": qid,
        "metrics": {"faithfulness": faith},
        "source_overlap": {"overlap": overlap},
        "forbidden": {"passed": forbidden_ok},
        **extra,
    }


def test_row_status():
    assert ui._row_status(row("a", 0.9, 0.5)) == "pass"
    assert ui._row_status(row("a", 0.9, None)) == "pass"  # no golden sources (security)
    assert ui._row_status(row("a", 0.5)) == "fail"
    assert ui._row_status(row("a", 0.9, 0.0)) == "fail"
    assert ui._row_status(row("a", 0.9, forbidden_ok=False)) == "fail"
    assert ui._row_status(row("a", None)) == "fail"


def categories(r):
    return [c for c, _ in ui._fail_reasons(r)]


def test_fail_reasons():
    assert ui._fail_reasons(row("a", 0.9)) == []
    [(cat, text)] = ui._fail_reasons(row("a", 0.4, answer="Use a probe.", grounding_score=1.0))
    assert cat == "low faithfulness" and "grounding check" in text
    miss = row("a", 0.9, 0.0, golden_sources=["a.pdf"], context_sources=["b.pdf"])
    [(cat, text)] = ui._fail_reasons(miss)
    assert cat == "retrieval miss" and "a.pdf" in text and "b.pdf" in text
    blocked = row(
        "a", 0.0, None, demonstrates_feature="security",
        blocked={"blocked": True, "reason": "banned_phrase", "false_block": False},
    )  # fmt: skip
    assert categories(blocked) == ["expected decline"]
    vague = row(
        "a", 0.0, 0.0, demonstrates_feature="self_rag",
        answer="Your question is too vague for me to answer well.",
    )  # fmt: skip
    assert categories(vague) == ["expected decline"]  # a clarifying question needs no sources
    assert categories(row("a", 0.0, answer="I don’t know.")) == ["declined"]
    assert categories(row("a", None)) == ["no score"]


def test_shared_ids_and_means_compare_only_common_questions():
    a = {"rows": [row("q-1", 0.2), row("q-2", 0.4), row("q-9", 1.0)]}
    b = {"rows": [row("q-2", 0.8), row("q-1", 0.6), row("q-5", 0.0)]}
    shared = ui._shared_ids(a, b)
    assert shared == ["q-2", "q-1"]  # run B's order; q-9 and q-5 aren't in both
    assert ui._metric_means(a["rows"], shared)["faithfulness"] == 0.3
    assert ui._metric_means(b["rows"], shared)["faithfulness"] == 0.7
    assert ui._metric_means(b["rows"], shared)["answer_relevancy"] is None  # no values


def test_comparison_rows_show_both_runs_and_the_flip():
    a = {"rows": [row("q-1", 0.2, question="Q?")]}
    b = {"rows": [row("q-1", 0.9, question="Q?")]}
    [entry] = ui._comparison_rows(a, b, ["q-1"])
    assert entry["faith A"] == 0.2 and entry["faith B"] == 0.9 and entry["Δ faith"] == 0.7
    assert entry["status"] == "fail → pass"


def test_judge_context_defaults_to_text():
    assert ui._judge_context({}) == "text"  # E1-E6 result files have no judge_context
    assert ui._judge_context({"judge_context": "source+text"}) == "source+text"


def test_format_result_choice(tmp_path):
    full = tmp_path / "20261022T101500Z_naive.json"
    full.write_text(
        json.dumps({"profile": "naive", "timestamp": "20261022T101500Z", "rows": [{}] * 35,
                    "questions_file": "golden_questions.yaml"}),
        encoding="utf-8",
    )  # fmt: skip
    subset = tmp_path / "20261027T220316Z_naive_q001-q005.json"
    subset.write_text(
        json.dumps({"profile": "naive", "timestamp": "20261027T220316Z", "rows": [{}] * 5,
                    "questions_file": "goldens_first5.yaml", "judge_context": "source+text"}),
        encoding="utf-8",
    )  # fmt: skip
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert ui._format_result_choice(full) == "naive · 20261022T101500Z · 35 questions · judge: text"
    assert ui._format_result_choice(subset) == (
        "naive · 20261027T220316Z · 5 questions (goldens_first5.yaml) · judge: source+text"
    )
    assert ui._format_result_choice(broken) == "broken.json"
    named = tmp_path / "naive_rag_v2.json"  # renamed result files show their name first
    named.write_text(
        json.dumps({"profile": "naive", "timestamp": "20261027T220316Z", "rows": [{}] * 5}),
        encoding="utf-8",
    )
    assert ui._format_result_choice(named) == (
        "naive_rag_v2 · naive · 20261027T220316Z · 5 questions · judge: text"
    )


def test_eval_files_sorted_by_run_time_not_name(tmp_path, monkeypatch):
    """Renamed files (naive_rag_v2.json) no longer start with the timestamp."""
    for name, ts in [("20261022T173834Z_naive.json", "20261022T173834Z"),
                     ("naive_rag_v2.json", "20261027T220316Z"),
                     ("q3_hybrid+hyde.json", "20261027T235756Z")]:  # fmt: skip
        (tmp_path / name).write_text(json.dumps({"timestamp": ts, "rows": []}), encoding="utf-8")
    monkeypatch.setattr(ui, "RESULTS_DIR", tmp_path)
    assert [f.name for f in ui._list_eval_files()] == [
        "q3_hybrid+hyde.json",
        "naive_rag_v2.json",
        "20261022T173834Z_naive.json",
    ]


def test_suggest_flags_and_retry_check():
    flags, why = ui._suggest_flags([("retrieval miss", "x")])
    assert flags == {"enable_hyde": True} and len(why) == 1
    assert ui._suggest_flags([("expected decline", "x")]) == ({}, [])
    golden = {"golden_sources": ["deployment.txt"], "golden_answer_keywords": ["maxSurge"]}
    check = ui._retry_check(golden, {"answer": "maxSurge ...", "sources": ["deployment.txt"]})
    assert check["found_sources"] == ["deployment.txt"] and check["keywords_found"] == ["maxSurge"]
