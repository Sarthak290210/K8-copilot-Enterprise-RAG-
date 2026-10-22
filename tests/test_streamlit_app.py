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


def row(faith, overlap, forbidden_ok=True):
    return {
        "metrics": {"faithfulness": faith},
        "source_overlap": {"overlap": overlap},
        "forbidden": {"passed": forbidden_ok},
    }


def test_row_status():
    assert ui._row_status(row(0.9, 0.5)) == "pass"
    assert ui._row_status(row(0.9, None)) == "pass"  # no golden sources (security)
    assert ui._row_status(row(0.5, 1.0)) == "fail"
    assert ui._row_status(row(0.9, 0.0)) == "fail"
    assert ui._row_status(row(0.9, 1.0, forbidden_ok=False)) == "fail"
    assert ui._row_status(row(None, 1.0)) == "fail"


def test_format_result_choice(tmp_path):
    good = tmp_path / "20261022T101500Z_naive.json"
    good.write_text(
        json.dumps(
            {"profile": "naive", "timestamp": "20261022T101500Z", "aggregate": {"n_evaluated": 35}}
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "broken.json"
    bad.write_text("{", encoding="utf-8")
    assert ui._format_result_choice(good) == "naive · 20261022T101500Z · 35 evaluated"
    assert ui._format_result_choice(bad) == "broken.json"
