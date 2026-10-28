from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LLM & Embeddings ===
    # Cerebras (OpenAI-compatible API). Switched from Groq on 2026-10-22: Groq's free tier
    # (200k tokens/day per model) couldn't fit one eval run, and its paid tier was closed.
    # Cerebras free trial: 1M tokens/day and 5 requests/min per model.
    cerebras_api_key: SecretStr
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    llm_model_answer: str = "gpt-oss-120b"
    llm_model_grader: str = "gpt-oss-120b"  # Cerebras has no gpt-oss-20b
    # Ragas judge for faithfulness / answer_relevancy: a different model family from the answer
    # model (no self-grading bias) with its own daily token budget.
    eval_judge_model: str = "qwen-3.8-27b"
    # How many questions answer_relevancy asks the judge to reverse-engineer from each answer
    # (Ragas default 3). 1 saves ~15-20% of judge tokens; used from E3 on (E1/E2 ran with 3).
    eval_relevancy_strictness: int = 3
    # Minimum gap between calls through llm_service (0 = off). The free trial allows
    # gpt-oss-120b only 5 requests/min and 150/hour, so long eval runs set ~24 s (150/hour)
    # via the environment instead of failing on 429s. Off for the API.
    llm_min_interval_seconds: float = 0.0
    # gpt-oss reasoning depth: more = slower and more tokens.
    llm_reasoning_effort: Literal["low", "medium", "high"] = "low"
    embedding_provider: Literal["sentence-transformers"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    hf_token: SecretStr | None = None

    # === Vector DB ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # === Postgres ===
    database_url: SecretStr
    # Text2SQL (Phase 4) runs as the read-only role copilot_ro (SELECT on the ops tables only).
    # The seed script gives it LOGIN with this password; None = Text2SQL can't run.
    copilot_ro_password: SecretStr | None = None
    sql_statement_timeout_ms: int = 5000
    sql_max_rows: int = 100

    # === Cache (Upstash) ===
    upstash_redis_url: str | None = None
    upstash_redis_token: SecretStr | None = None
    cache_ttl_embeddings: int = 604800  # 7 days
    cache_ttl_rag: int = 3600  # 1 hour
    cache_ttl_sql_gen: int = 86400  # 24 hours
    cache_ttl_sql_result: int = 900  # 15 min
    cache_ttl_intent: int = 86400  # 24 hours
    # Phase 5: false = every tier is skipped (e.g. to eval retrieval changes without cached
    # answers). Each process also keeps an in-memory copy in front of Redis (user's choice
    # 2026-10-26: a warm answer in < 100 ms; Redis alone costs ~44 ms per lookup).
    cache_enabled: bool = True
    cache_memory_max_entries: int = 1000
    # Blended gpt-oss-120b price (~$0.35/M in, $0.75/M out) for metadata.cost_saved_usd.
    llm_cost_per_million_tokens: float = 0.5

    # === Doc dedup ===
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: str = "./cache"
    s3_cache_bucket: str = "adv-rag-cache"
    aws_region: str = "us-east-1"

    # === Web search ===
    tavily_api_key: SecretStr | None = None

    # === Auth ===
    jwt_secret: SecretStr
    jwt_expiration_minutes: int = 60

    # === Seed users (scripts/seed_db.py; a user is skipped if its password is unset) ===
    seed_admin_password: SecretStr | None = None
    seed_analyst_password: SecretStr | None = None

    # === Rate limit + budget ===
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    max_tokens_per_user_daily: int = 100_000
    auth_login_rate_limit_per_min: int = 5
    auth_register_rate_limit_per_hour: int = 3

    # === Input restructuring ===
    max_input_tokens: int = 3000
    reserved_context_tokens: int = 1000
    reserved_output_tokens: int = 1000

    # === Security thresholds ===
    prompt_injection_threshold: float = 0.9  # user's choice 2026-10-27: q-042 scored 0.809
    toxicity_threshold: float = 0.75
    output_toxicity_threshold: float = 0.5
    max_validation_retries: int = 2
    # Phase 6: llm-guard models (injection + toxicity, ONNX; user's choice) for the input guard
    # and output moderation; off only in tests.
    security_models_enabled: bool = True
    # A guard that fails to load or crashes blocks the request (true) or lets it through with
    # a warning (false, dev only).
    security_fail_closed: bool = True
    # Grounding check (L10): every answer's claims checked against its chunks (user's choice).
    grounding_enabled: bool = True
    # >= pass; between warn and pass -> warning. 0.9 since 2026-10-28 (user's choice; was 0.8):
    # one unsupported claim in six (q-014, 0.83) now triggers the one rewrite.
    grounding_pass_score: float = 0.9
    grounding_warn_score: float = 0.5  # < warn -> answer replaced by "couldn't verify"

    # === Retrieval defaults ===
    hyde_num_hypotheses: int = 3
    hyde_enabled_by_default: bool = False
    # Chunks a HyDE request answers from (at least; a larger top_k flag wins). Free replay
    # 2026-10-28 (user's choice): HyDE recall 0.80 at 5 -> 0.87 at 8, ~+1-2k tokens per answer.
    hyde_top_k: int = 8
    hybrid_search_enabled: bool = True
    rrf_k: int = 60
    # Hybrid search keeps at most this many chunks per document (0 = no cap); not applied to
    # rerank candidates. Free replay 2026-10-28 (user's choice): recall +0.035, precision +0.008.
    max_chunks_per_source: int = 2
    reranker_backend: Literal["local", "voyage"] = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Tokens of (question + chunk) the local cross-encoder reads. 256 (user's choice,
    # 2026-10-22): ~1.2 s per 20-chunk rerank on this CPU vs ~3.6 s at the model's full 512.
    reranker_max_length: int = 256
    voyage_api_key: SecretStr | None = None
    voyage_model: str = "rerank-2.5"
    reranker_initial_top_k: int = 20
    reranking_enabled_by_default: bool = False  # E2: reranking hurt on this corpus
    crag_relevance_threshold: float = 0.7
    crag_ambiguous_threshold: float = 0.5
    crag_enabled_by_default: bool = True
    reflection_min_score: float = 0.8  # user's choice 2026-10-26 (was 0.85)
    max_reflection_retries: int = 2
    self_reflective_enabled_by_default: bool = False

    # === Server ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # === Logging ===
    log_json: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_is_strong(cls, v: SecretStr) -> SecretStr:
        # A short HS256 key can be brute-forced, letting anyone forge admin tokens.
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # required fields come from .env


settings = get_settings()
