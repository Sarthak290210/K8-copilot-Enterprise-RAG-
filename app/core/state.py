"""The shared notebook every graph node reads and writes (Phase 4).

Only JSON-friendly values (dicts, lists, strings, numbers) go in, because the Postgres
checkpointer saves the state after every node; Pydantic objects are stored as model_dump()s.
Plain fields (no reducers): each node overwrites what it returns, so retries never append.
"""

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    question: str
    user_id: int | None
    flags: dict[str, Any]
    started_at: float  # time.time() when the run started (latency includes approval wait)
    intent: str  # sql | rag | hybrid
    tokens: int  # every LLM call in this run

    # SQL path
    generated_sql: str | None
    sql_explanation: str | None
    sql_approved: bool | None
    sql_rows: list[dict[str, Any]] | None
    sql_error: str | None

    # docs path (RetrievedChunk / CRAGEvaluation model_dump()s)
    retrieved_chunks: list[dict[str, Any]]
    crag: dict[str, Any] | None

    # result: a ChatResponse model_dump(mode="json") and the full chunks it used
    response: dict[str, Any]
    final_chunks: list[dict[str, Any]]
