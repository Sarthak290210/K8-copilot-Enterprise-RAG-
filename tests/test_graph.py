"""The Phase 4 agent graph with an in-memory checkpointer and every service mocked."""

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.core import graph as agent
from app.models import ChatResponse, ResponseMetadata, RetrievedChunk
from app.services.sql_service import SQLGenerationError


def chunk(source, text="docs text"):
    return RetrievedChunk(text=text, source=source, score=0.9, metadata={"chunk_index": 0})


@pytest.fixture
def fakes(monkeypatch):
    state = {
        "intent": "sql",
        "sql": "SELECT count(*) AS n FROM incidents WHERE severity = 'P1' LIMIT 100",
        "sql_error": None,
        "rows": [{"n": 392}],
        "executed": [],
        "llm": [],
        "rag_calls": 0,
    }

    async def classify_intent(question):
        return state["intent"], 10

    class FakeSQL:
        async def generate_sql(self, question):
            if state["sql_error"]:
                raise SQLGenerationError(state["sql_error"], tokens_used=70)
            return {"sql": state["sql"], "explanation": "counts P1 incidents", "tokens_used": 20}

        async def execute_sql(self, sql):
            state["executed"].append(sql)
            return state["rows"]

        async def data_end(self):
            import datetime as dt

            return dt.date(2026, 5, 16)

        async def data_notes(self):
            return "Current date: 2026-10-26\nDate ranges in the data:\n- incidents: ..."

    async def generate(system, user, **kwargs):
        state["llm"].append((system, user))
        return {
            "content": "There have been 392 P1 incidents.",
            "model": "m",
            "usage": {"total_tokens": 30},
        }

    async def run_rag_with_trace(question, flags):
        state["rag_calls"] += 1
        response = ChatResponse(
            answer="A StatefulSet is ... [1]",
            sources=["statefulset.html"],
            metadata=ResponseMetadata(tokens_used=100),
        )
        return response, [chunk("statefulset.html")]

    async def retrieve(question, flags, trace=None):
        return [chunk("logging.pdf", "kubectl logs --previous")]

    sql_service = FakeSQL()
    monkeypatch.setattr(agent, "classify_intent", classify_intent)
    monkeypatch.setattr(agent, "get_sql_service", lambda: sql_service)
    monkeypatch.setattr(agent.llm_service, "generate", generate)
    monkeypatch.setattr(agent.rag_service, "run_rag_with_trace", run_rag_with_trace)
    monkeypatch.setattr(agent.rag_service, "_retrieve", retrieve)
    return state


@pytest.fixture
def graph():
    return agent.build_graph(InMemorySaver())


async def test_rag_question_answers_without_sql(fakes, graph):
    fakes["intent"] = "rag"
    result = await agent.run_agent(graph, "What is a StatefulSet?", None, uuid.uuid4())
    assert not result.pending and result.response.answer == "A StatefulSet is ... [1]"
    assert result.response.metadata.intent == "rag"
    assert result.response.metadata.tokens_used == 10 + 100  # routing + the RAG pipeline
    assert fakes["executed"] == [] and fakes["rag_calls"] == 1
    assert [c.source for c in result.chunks] == ["statefulset.html"]


async def test_sql_question_pauses_before_running(fakes, graph):
    qid = uuid.uuid4()
    result = await agent.run_agent(graph, "How many P1 incidents?", None, qid)
    assert result.pending and result.response.query_id == qid
    pending = result.response.pending_sql
    assert pending is not None and pending.sql == fakes["sql"] and pending.intent == "sql"
    assert fakes["executed"] == []  # nothing runs before approval
    assert await agent.is_waiting_for_approval(graph, qid)


async def test_approve_runs_the_sql_and_explains_the_rows(fakes, graph):
    qid = uuid.uuid4()
    await agent.run_agent(graph, "How many P1 incidents?", None, qid)
    result = await agent.resume_agent(graph, qid, approved=True)
    assert fakes["executed"] == [fakes["sql"]]
    assert result.response.answer == "There have been 392 P1 incidents."
    meta = result.response.metadata
    assert meta.intent == "sql" and meta.generated_sql == fakes["sql"] and meta.sql_row_count == 1
    assert result.response.sources == ["query_results"]
    context = result.chunks[0].text  # the eval's context: query + what it computes + rows
    assert context.startswith(f"Query: {fakes['sql']}\nWhat it computes: counts P1 incidents")
    assert context.endswith("Results:\nn\n392")
    assert "Date ranges in the data" in fakes["llm"][0][1]
    assert meta.tokens_used == 10 + 20 + 30
    assert not await agent.is_waiting_for_approval(graph, qid)


async def test_reject_never_runs_the_sql(fakes, graph):
    qid = uuid.uuid4()
    await agent.run_agent(graph, "How many P1 incidents?", None, qid)
    result = await agent.resume_agent(graph, qid, approved=False)
    assert fakes["executed"] == [] and fakes["llm"] == []
    assert result.response.answer == agent.REJECTED_NOTE
    assert result.response.metadata.generated_sql is None


async def test_edited_sql_is_revalidated(fakes, graph):
    qid = uuid.uuid4()
    await agent.run_agent(graph, "q", None, qid)
    result = await agent.resume_agent(graph, qid, approved=True, edited_sql="DROP TABLE pods")
    assert fakes["executed"] == []
    assert "not a safe SELECT" in result.response.answer


async def test_edited_safe_sql_is_what_runs(fakes, graph):
    qid = uuid.uuid4()
    await agent.run_agent(graph, "q", None, qid)
    edited = "SELECT count(*) AS n FROM incidents WHERE severity = 'P2' LIMIT 10"
    await agent.resume_agent(graph, qid, approved=True, edited_sql=edited + ";")
    assert fakes["executed"] == [edited]


async def test_sql_generation_failure_skips_approval(fakes, graph):
    fakes["sql_error"] = "Could not generate a safe SELECT query for this question"
    result = await agent.run_agent(graph, "q", None, uuid.uuid4())
    assert not result.pending and "couldn't answer this from the database" in result.response.answer
    assert result.response.metadata.tokens_used == 10 + 70  # routing + the rejected attempts


async def test_hybrid_uses_rows_and_real_chunk_text(fakes, graph):
    fakes["intent"] = "hybrid"
    qid = uuid.uuid4()
    first = await agent.run_agent(
        graph, "How many crashing pods, and how do I see logs?", None, qid
    )
    assert first.pending and fakes["executed"] == []  # never SQL before approval, even hybrid
    result = await agent.resume_agent(graph, qid, approved=True)
    system, user = fakes["llm"][0]
    assert "DATABASE RESULTS" in system and "n\n392" in user
    assert "kubectl logs --previous" in user  # the real chunk text, not a file name
    assert result.response.sources == ["query_results", "logging.pdf"]
    assert result.response.metadata.intent == "hybrid"


async def test_bad_flags_fail_before_any_call(fakes, graph):
    with pytest.raises(ValueError, match="Invalid flags"):
        await agent.run_agent(graph, "q", {"top_k": 0}, uuid.uuid4())


async def test_zero_result_for_a_period_after_the_data_gets_a_code_note(fakes, graph):
    fakes["sql"] = (
        "SELECT count(*) AS n FROM incidents WHERE started_at >= CURRENT_DATE - INTERVAL '1 month'"
    )
    fakes["rows"] = [{"n": 0}]
    qid = uuid.uuid4()
    await agent.run_agent(graph, "How many P1 incidents last month?", None, qid)
    result = await agent.resume_agent(graph, qid, approved=True)
    assert "our fleet data ends on 2026-05-16" in result.response.answer


async def test_no_note_for_non_zero_or_absolute_queries(fakes, graph):
    qid = uuid.uuid4()  # default fake: no CURRENT_DATE, 392 rows counted
    await agent.run_agent(graph, "How many P1 incidents?", None, qid)
    result = await agent.resume_agent(graph, qid, approved=True)
    assert "fleet data ends" not in result.response.answer
