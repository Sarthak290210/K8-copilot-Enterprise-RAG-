"""Shapes of the data moving through the system (requests, retrieved chunks, answers)."""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SEARCH_MODES = ("dense", "sparse", "hybrid")
MAX_TOP_K = 20


class RetrievedChunk(BaseModel):
    """One retrieved piece of a document."""

    text: str
    source: str  # file name (or URL for web results)
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)  # chunk_index, headings, page


class RetrievedChunkPreview(BaseModel):
    """Short version of a chunk for the UI."""

    source: str
    score: float
    snippet: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, length: int = 200) -> "RetrievedChunkPreview":
        snippet = " ".join(chunk.text.split())[:length]
        return cls(source=chunk.source, score=round(chunk.score, 4), snippet=snippet)


class CRAGEvaluation(BaseModel):
    """The CRAG grader's verdict on the retrieved chunks."""

    scores: list[float] = Field(default_factory=list)  # one 0-1 relevance score per chunk
    max_score: float | None = None
    # ungraded = the grader call failed: chunks are kept as they are, confidence unknown.
    action: Literal["correct", "ambiguous", "incorrect", "ungraded"] = "ungraded"
    reason: str = ""
    web_used: bool = False
    # Is the question about Kubernetes / IT operations? Only then may CRAG search the web
    # (user's rule, 2026-10-26). None = the grader didn't say (treated as "no web").
    in_scope: bool | None = None
    # No clear subject or goal ("explain", "what is wrong"): never searched on the web (E3 got
    # dictionary definitions); if the docs don't help, the user is asked for more detail.
    too_vague: bool | None = None
    tokens_used: int = 0


class ReflectionResult(BaseModel):
    """The Self-RAG grader's verdict on a generated answer: four criteria, each 0-1, and
    `score` = their mean (user's design, 2026-10-26)."""

    relevance: float = 0.0  # addresses what the user asked
    accuracy: float = 0.0  # every claim backed by the context
    completeness: float = 0.0  # covers what the question needs (as far as the context allows)
    clarity: float = 0.0  # clear and well organised
    score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    refined_question: str | None = None
    tokens_used: int = 0


class ResponseMetadata(BaseModel):
    intent: str = "rag"
    flags_used: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_used: int = 0  # every LLM call for this answer (HyDE, CRAG, answer, reflection)
    cache_hits: dict[str, bool] = Field(default_factory=dict)  # tier -> hit (Phase 5)
    cost_saved_usd: float = 0.0  # estimated price of the LLM tokens the cache hits skipped
    model: str | None = None
    hyde_hypotheses: list[str] = Field(default_factory=list)
    crag: CRAGEvaluation | None = None
    reflection_iterations: int = 0  # extra answer attempts made by Self-RAG
    reflection_score: float | None = None
    refined_question: str | None = None  # the question the returned answer was written for
    generated_sql: str | None = None  # Phase 4: the SQL that ran (sql / hybrid intents)
    sql_row_count: int | None = None


class PendingSQLBlock(BaseModel):
    """SQL waiting for a human: the agent paused before running it (Phase 4)."""

    query_id: uuid.UUID  # the paused run (LangGraph thread); send it to /query/sql/execute
    sql: str
    explanation: str = ""
    intent: str = "sql"


class ChatResponse(BaseModel):
    """What the API returns for a question."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float | None = None  # set by real graders from Phase 3; never hard-coded
    chunks: list[RetrievedChunkPreview] = Field(default_factory=list)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    query_id: uuid.UUID | None = None  # the agent run this answer belongs to
    pending_sql: PendingSQLBlock | None = None  # set when the SQL needs approval first


class QueryFlags(BaseModel):
    """Per-request feature switches; anything left out uses rag_service.DEFAULT_FLAGS.

    Validated so a client can't pick top_k 0 / -1 / 100000, and lax pydantic parsing turns
    "false" / "0" into a real False (a plain truthiness check would treat "false" as on).
    """

    model_config = ConfigDict(extra="forbid")

    search_mode: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    enable_rerank: bool | None = None
    enable_hyde: bool | None = None
    enable_crag: bool | None = None
    enable_self_reflective: bool | None = None

    @field_validator("search_mode")
    @classmethod
    def _known_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in SEARCH_MODES:
            raise ValueError(
                f"Unknown search_mode '{value}'. Use one of: {', '.join(SEARCH_MODES)}"
            )
        return value


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    flags: QueryFlags | None = None  # e.g. {"search_mode": "hybrid", "top_k": 5}


class SqlExecuteRequest(BaseModel):
    """Approve (optionally with edited SQL) or reject the SQL of a paused run."""

    query_id: uuid.UUID
    approved: bool
    edited_sql: str | None = Field(default=None, max_length=10_000)
