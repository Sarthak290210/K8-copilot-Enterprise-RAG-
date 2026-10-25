from fastapi.testclient import TestClient

from app.api import query as query_api
from app.main import app
from app.middleware.auth import create_access_token
from app.models import ChatResponse

client = TestClient(app)  # no `with`: lifespan (real services) doesn't run


def auth(user_id=1):
    return {"Authorization": f"Bearer {create_access_token(user_id, 'u', False)}"}


def test_query_requires_token():
    assert client.post("/query", json={"question": "What is a Pod?"}).status_code == 401


def test_query_returns_rag_answer(monkeypatch):
    seen = {}

    async def fake_run_rag(question, flags):
        seen["args"] = (question, flags)
        return ChatResponse(answer="A Pod is ... [1]", sources=["pods.html"])

    monkeypatch.setattr(query_api, "run_rag", fake_run_rag)
    r = client.post(
        "/query", json={"question": "What is a Pod?", "flags": {"top_k": 3}}, headers=auth()
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "A Pod is ... [1]" and r.json()["sources"] == ["pods.html"]
    assert seen["args"] == ("What is a Pod?", {"top_k": 3})


def test_bad_flags_give_400(monkeypatch):
    async def fake_run_rag(question, flags):
        raise ValueError("search_mode 'hybrid' is not available yet")

    monkeypatch.setattr(query_api, "run_rag", fake_run_rag)
    r = client.post(
        "/query", json={"question": "q", "flags": {"search_mode": "hybrid"}}, headers=auth()
    )
    assert r.status_code == 400 and "not available yet" in r.json()["detail"]


def test_validation_rejects_empty_question():
    assert client.post("/query", json={"question": ""}, headers=auth()).status_code == 422


def test_per_user_rate_limit(monkeypatch):
    async def fake_run_rag(question, flags):
        return ChatResponse(answer="ok")

    monkeypatch.setattr(query_api, "run_rag", fake_run_rag)
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


def test_bad_flags_are_rejected_before_any_work(monkeypatch):
    async def fail_run_rag(question, flags):
        raise AssertionError("run_rag should not be called")

    monkeypatch.setattr(query_api, "run_rag", fail_run_rag)
    for flags in ({"top_k": 0}, {"top_k": 100000}, {"enable_crag": "maybe"}, {"nope": 1}):
        r = client.post("/query", json={"question": "q", "flags": flags}, headers=auth())
        assert r.status_code == 422, flags


def test_string_booleans_are_parsed(monkeypatch):
    seen = {}

    async def fake_run_rag(question, flags):
        seen["flags"] = flags
        return ChatResponse(answer="ok")

    monkeypatch.setattr(query_api, "run_rag", fake_run_rag)
    r = client.post(
        "/query",
        json={"question": "q", "flags": {"enable_crag": "false", "top_k": "3"}},
        headers=auth(),
    )
    assert r.status_code == 200 and seen["flags"] == {"enable_crag": False, "top_k": 3}
