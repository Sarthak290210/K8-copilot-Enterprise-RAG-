import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {"cerebras_api_key": "k", "database_url": "postgresql://u:p@h/db"}


def make(**overrides):
    # _env_file=None: only the values passed here (and the test env vars) are used.
    return Settings(_env_file=None, **(REQUIRED | overrides))  # pyright: ignore[reportCallIssue]


def test_short_jwt_secret_fails_at_startup():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        make(jwt_secret="too-short")


def test_empty_jwt_secret_fails_at_startup():
    with pytest.raises(ValidationError):
        make(jwt_secret="")


def test_32_char_jwt_secret_is_accepted():
    assert make(jwt_secret="x" * 32).jwt_secret.get_secret_value() == "x" * 32


def test_defaults_match_plan():
    s = make(jwt_secret="x" * 32)
    assert s.llm_model_answer == "gpt-oss-120b"
    assert s.llm_model_grader == "gpt-oss-120b"
    assert s.eval_judge_model == "qwen-3.8-27b"
    assert s.cerebras_base_url == "https://api.cerebras.ai/v1"
    assert s.embedding_model == "BAAI/bge-base-en-v1.5"
    assert s.embedding_dim == 768
    assert s.reranker_max_length == 256
    assert (s.rate_limit_requests, s.rate_limit_window_seconds) == (20, 60)
