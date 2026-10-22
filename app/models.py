"""Shapes of the data moving through the system (requests, retrieved chunks, answers)."""

from typing import Any

from pydantic import BaseModel, Field


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


class ResponseMetadata(BaseModel):
    intent: str = "rag"
    flags_used: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_used: int = 0
    cache_hits: dict[str, bool] = Field(default_factory=dict)
    model: str | None = None


class ChatResponse(BaseModel):
    """What the API returns for a question."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float | None = None  # set by real graders from Phase 3; never hard-coded
    chunks: list[RetrievedChunkPreview] = Field(default_factory=list)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    flags: dict[str, Any] | None = None  # e.g. {"search_mode": "dense", "top_k": 5}
