"""SQLService with a mocked LLM and schema (no database)."""

import datetime as dt
import uuid
from decimal import Decimal

import pytest

from app.services import sql_service
from app.services.llm_service import LLMJSONError


@pytest.fixture
def service(monkeypatch):
    replies = []
    seen = []

    async def generate_with_json(system, user, model=None, temperature=0.0):
        seen.append((system, user))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"data": reply, "model": model, "usage": {"total_tokens": 50}}

    monkeypatch.setattr(sql_service.llm_service, "generate_with_json", generate_with_json)
    svc = sql_service.SQLService()
    svc._schema = "Current date: 2026-10-26\n\nTables:\n- incidents(...)"
    return svc, replies, seen


async def test_generate_sql_returns_a_safe_select(service):
    svc, replies, seen = service
    replies.append({"sql": "SELECT count(*) FROM incidents LIMIT 100;", "explanation": "counts"})
    result = await svc.generate_sql("How many incidents?")
    assert result == {
        "sql": "SELECT count(*) FROM incidents LIMIT 100",
        "explanation": "counts",
        "tokens_used": 50,
    }
    assert "Current date: 2026-10-26" in seen[0][0] and "LIMIT 100" in seen[0][0]


async def test_unsafe_sql_is_retried_once_with_the_error(service):
    svc, replies, seen = service
    replies += [{"sql": "DELETE FROM incidents"}, {"sql": "SELECT 1 FROM incidents"}]
    result = await svc.generate_sql("q")
    assert result["sql"] == "SELECT 1 FROM incidents" and result["tokens_used"] == 100
    assert "previous answer was rejected" in seen[1][1]


async def test_two_failures_raise(service):
    svc, replies, _ = service
    replies += [LLMJSONError("bad", {"total_tokens": 7}), {"sql": "SELECT * FROM users"}]
    with pytest.raises(sql_service.SQLGenerationError):
        await svc.generate_sql("q")


async def test_execute_refuses_unsafe_sql_before_connecting(monkeypatch):
    def no_connection(*args, **kwargs):
        raise AssertionError("must not connect")

    monkeypatch.setattr(sql_service.AsyncConnection, "connect", no_connection)
    with pytest.raises(sql_service.SQLGenerationError, match="Refusing"):
        await sql_service.SQLService().execute_sql("DROP TABLE pods")


def test_readonly_conninfo_needs_the_password(monkeypatch):
    monkeypatch.setattr(sql_service.settings, "copilot_ro_password", None)
    with pytest.raises(sql_service.SQLGenerationError, match="COPILOT_RO_PASSWORD"):
        sql_service.readonly_conninfo()


def test_serialize_row():
    uid = uuid.uuid4()
    row = {
        "d": Decimal("1.5"),
        "t": dt.datetime(2026, 5, 16, 14, 0, tzinfo=dt.UTC),
        "day": dt.date(2026, 5, 16),
        "dur": dt.timedelta(minutes=2),
        "id": uid,
        "s": "x",
    }
    assert sql_service._serialize_row(row) == {
        "d": 1.5,
        "t": "2026-05-16T14:00:00+00:00",
        "day": "2026-05-16",
        "dur": 120.0,
        "id": str(uid),
        "s": "x",
    }


def test_rows_as_text():
    assert sql_service.rows_as_text([]) == "(the query returned 0 rows)"
    text = sql_service.rows_as_text([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}], 2)
    assert text == "a | b\n1 | x\n2 | y\n... (1 more rows)"


async def test_data_notes_extracts_dates(service):
    svc, _, _ = service
    svc._schema = (
        "Current date: 2026-10-26\n\nTables:\n- x\n\nDate ranges in the data:\n"
        "- incidents.started_at: 2025-05-16 to 2026-05-16\n\nSample rows:\n- x"
    )
    assert await svc.data_notes() == (
        "Current date: 2026-10-26\nDate ranges in the data:\n"
        "- incidents.started_at: 2025-05-16 to 2026-05-16"
    )


async def test_generated_sql_is_cached(service, cache_on):
    svc, replies, seen = service
    replies.append({"sql": "SELECT count(*) FROM incidents LIMIT 100", "explanation": "counts"})
    first = await svc.generate_sql("How many incidents?")
    second = await svc.generate_sql("how many incidents?")
    assert len(seen) == 1 and second["sql"] == first["sql"] and second["tokens_used"] == 0


async def test_cached_sql_is_not_reused_on_another_day_or_schema(service, cache_on, monkeypatch):
    """'Last month' asked tomorrow (or after a re-seed changed the data ranges) is another
    query: the cache key holds today's date and the schema prompt."""
    svc, replies, seen = service
    sql = "SELECT count(*) FROM incidents LIMIT 100"
    replies.extend({"sql": sql, "explanation": "counts"} for _ in range(3))
    await svc.generate_sql("How many incidents last month?")

    class Tomorrow(dt.date):
        @classmethod
        def today(cls):
            return dt.date(2026, 11, 1)

    monkeypatch.setattr(sql_service.dt, "date", Tomorrow)
    await svc.generate_sql("How many incidents last month?")
    svc._schema += "\nThere is NO data after 2026-11-01."
    await svc.generate_sql("How many incidents last month?")
    assert len(seen) == 3


async def test_sql_results_are_cached_and_unsafe_sql_still_refused(monkeypatch, cache_on):
    connects = []

    class FakeCursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, query):
            pass

        async def fetchmany(self, n):
            return [{"n": Decimal("3")}]

    class FakeTransaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeConn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def set_read_only(self, value):
            pass

        def transaction(self):
            return FakeTransaction()

        async def execute(self, query):
            pass

        def cursor(self, row_factory=None):
            return FakeCursor()

    async def connect(conninfo):
        connects.append(1)
        return FakeConn()

    monkeypatch.setattr(sql_service.AsyncConnection, "connect", connect)
    monkeypatch.setattr(sql_service, "readonly_conninfo", lambda: "dbname=x")
    svc = sql_service.SQLService()
    query = "SELECT count(*) AS n FROM incidents"
    assert await svc.execute_sql(query) == [{"n": 3.0}]
    assert await svc.execute_sql(query.lower() + " ;") == [{"n": 3.0}]
    assert len(connects) == 1  # the second one came from the cache
    await cache_on.set_sql_result("DELETE FROM incidents", [{"n": 1}])
    with pytest.raises(sql_service.SQLGenerationError):
        await svc.execute_sql("DELETE FROM incidents")  # checked before any cache lookup
