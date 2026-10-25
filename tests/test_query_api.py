import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import query as query_api
from app.api.auth import get_pool
from app.core.graph import AgentResult
from app.main import app
from app.middleware.auth import create_access_token
from app.models import ChatResponse, PendingSQLBlock

client = TestClient(app)  # no `with`: lifespan (real services) doesn't run


def auth(user_id=1):
    return {"Authorization": f"Bearer {create_access_token(user_id, 'u', False)}"}


class FakePool:
    """query_threads in memory: thread_id -> user_id."""

    def __init__(self):
        self.threads = {}

    def connection(self):
        pool = self

        class Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, sql, params):
                if sql.startswith("INSERT"):
                    pool.threads[params[0]] = params[1]
                    return None
                owner = pool.threads.get(params[0])

                class Cursor:
                    async def fetchone(self):
                        return None if owner is None else (owner,)

                return Cursor()

        return Conn()


@pytest.fixture
def agent(monkeypatch):
    """Fake graph runner: SQL questions pause, everything else answers."""
    state = {"calls": [], "waiting": set(), "resumed": []}
    pool = FakePool()

    async def run_agent(graph, question, flags, query_id, user_id=None):
        state["calls"].append((question, flags, user_id))
        if question.startswith("how many"):
            state["waiting"].add(query_id)
            pending = PendingSQLBlock(query_id=query_id, sql="SELECT 1", explanation="x")
            return AgentResult(
                ChatResponse(answer="review", query_id=query_id, pending_sql=pending)
            )
        return AgentResult(ChatResponse(answer="A Pod is ... [1]", query_id=query_id))

    async def is_waiting(graph, query_id):
        return query_id in state["waiting"]

    async def resume(graph, query_id, approved, edited_sql=None):
        state["waiting"].discard(query_id)
        state["resumed"].append((query_id, approved, edited_sql))
        return AgentResult(
            ChatResponse(answer="392" if approved else "rejected", query_id=query_id)
        )

    monkeypatch.setattr(query_api, "run_agent", run_agent)
    monkeypatch.setattr(query_api, "is_waiting_for_approval", is_waiting)
    monkeypatch.setattr(query_api, "resume_agent", resume)
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[query_api.get_graph] = lambda: object()
    state["pool"] = pool
    yield state
    app.dependency_overrides.clear()


def test_query_requires_token():
    assert client.post("/query", json={"question": "What is a Pod?"}).status_code == 401


def test_query_returns_the_agent_answer_and_records_the_owner(agent):
    r = client.post(
        "/query", json={"question": "What is a Pod?", "flags": {"top_k": 3}}, headers=auth(5)
    )
    assert r.status_code == 200 and r.json()["answer"] == "A Pod is ... [1]"
    assert agent["calls"] == [("What is a Pod?", {"top_k": 3}, 5)]
    query_id = uuid.UUID(r.json()["query_id"])
    assert agent["pool"].threads[query_id] == 5


def test_sql_question_returns_pending_sql(agent):
    r = client.post("/query", json={"question": "how many P1 incidents?"}, headers=auth())
    body = r.json()
    assert r.status_code == 200 and body["pending_sql"]["sql"] == "SELECT 1"
    assert body["pending_sql"]["query_id"] == body["query_id"]


def test_owner_can_approve(agent):
    query_id = client.post("/query", json={"question": "how many?"}, headers=auth(1)).json()[
        "query_id"
    ]
    r = client.post(
        "/query/sql/execute", json={"query_id": query_id, "approved": True}, headers=auth(1)
    )
    assert r.status_code == 200 and r.json()["answer"] == "392"
    assert agent["resumed"] == [(uuid.UUID(query_id), True, None)]


def test_another_user_cannot_approve(agent):
    """Phase 4 Done-when: another user's token can't approve your query_id (404)."""
    query_id = client.post("/query", json={"question": "how many?"}, headers=auth(1)).json()[
        "query_id"
    ]
    r = client.post(
        "/query/sql/execute", json={"query_id": query_id, "approved": True}, headers=auth(2)
    )
    assert r.status_code == 404 and agent["resumed"] == []


def test_unknown_query_id_is_404(agent):
    r = client.post(
        "/query/sql/execute", json={"query_id": str(uuid.uuid4()), "approved": True}, headers=auth()
    )
    assert r.status_code == 404


def test_cannot_approve_twice(agent):
    query_id = client.post("/query", json={"question": "how many?"}, headers=auth()).json()[
        "query_id"
    ]
    body = {"query_id": query_id, "approved": False}
    assert client.post("/query/sql/execute", json=body, headers=auth()).status_code == 200
    assert client.post("/query/sql/execute", json=body, headers=auth()).status_code == 409


def test_execute_requires_token():
    r = client.post("/query/sql/execute", json={"query_id": str(uuid.uuid4()), "approved": True})
    assert r.status_code == 401


def test_bad_flags_give_400(agent, monkeypatch):
    async def run_agent(graph, question, flags, query_id, user_id=None):
        raise ValueError("Invalid flags: search_mode: ...")

    monkeypatch.setattr(query_api, "run_agent", run_agent)
    r = client.post("/query", json={"question": "q"}, headers=auth())
    assert r.status_code == 400 and "Invalid flags" in r.json()["detail"]


def test_validation_rejects_empty_question(agent):
    assert client.post("/query", json={"question": ""}, headers=auth()).status_code == 422


def test_per_user_rate_limit(agent):
    limit = query_api.query_limiter.max_requests
    codes = [
        client.post("/query", json={"question": "q"}, headers=auth(7)).status_code
        for _ in range(limit + 1)
    ]
    assert codes[:limit] == [200] * limit and codes[limit] == 429
    # Another user has their own budget.
    assert client.post("/query", json={"question": "q"}, headers=auth(8)).status_code == 200


def test_query_defaults_are_public_and_match_rag_service():
    from app.services import rag_service

    r = client.get("/query/defaults")  # no token: the UI reads it before login
    assert r.status_code == 200 and r.json() == rag_service.DEFAULT_FLAGS


def test_bad_flags_are_rejected_before_any_work(agent):
    for flags in ({"top_k": 0}, {"top_k": 100000}, {"enable_crag": "maybe"}, {"nope": 1}):
        r = client.post("/query", json={"question": "q", "flags": flags}, headers=auth())
        assert r.status_code == 422, flags
    assert agent["calls"] == []


def test_string_booleans_are_parsed(agent):
    r = client.post(
        "/query",
        json={"question": "q", "flags": {"enable_crag": "false", "top_k": "3"}},
        headers=auth(),
    )
    assert r.status_code == 200 and agent["calls"][0][1] == {"enable_crag": False, "top_k": 3}
