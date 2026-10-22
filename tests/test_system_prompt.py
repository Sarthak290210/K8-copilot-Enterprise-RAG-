from app.security.system_prompt import build_system_prompt


def test_prompt_rules():
    prompt = build_system_prompt()
    assert "Answer ONLY from the provided context" in prompt
    assert "[1]" in prompt  # citation format the grounding check (Phase 6) relies on
    assert "don't know" in prompt
