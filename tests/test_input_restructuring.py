"""L5: token counting and truncation."""

from app.security import input_restructuring as ir


def test_short_input_is_unchanged():
    assert ir.restructure_input("What is a Pod?") == ("What is a Pod?", "unchanged")


def test_long_input_is_cut_to_the_limit(monkeypatch):
    monkeypatch.setattr(ir.settings, "max_input_tokens", 20)
    text, action = ir.restructure_input("pods " * 200)
    assert action == "truncated" and ir.count_tokens(text) <= 20


def test_estimate_when_tiktoken_is_unavailable(monkeypatch):
    monkeypatch.setattr(ir, "_encoding", lambda: None)
    assert ir.count_tokens("x" * 40) == 10
    assert ir.truncate_text("x" * 400, max_tokens=10) == ("x" * 40, "truncated")
