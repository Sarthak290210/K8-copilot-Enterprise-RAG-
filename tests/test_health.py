import asyncio

from fastapi.testclient import TestClient

from app.api import admin
from app.main import app
from app.middleware.auth import User, require_admin

# TestClient without `with` doesn't run the lifespan, so no real services are contacted.
client = TestClient(app)


def _up(name):
    async def ping(_client):
        return admin.ServiceStatus(service=name, status="up", latency_ms=1.0)

    return ping


def _down(name):
    async def ping(_client):
        return admin.ServiceStatus(service=name, status="down", error="ConnectError")

    return ping


def _use_fakes(monkeypatch, postgres_up=True):
    app.dependency_overrides[require_admin] = lambda: User(id=1, username="admin", is_admin=True)
    for dep in (admin.get_pool, admin.get_qdrant, admin.get_redis, admin.get_llm):
        app.dependency_overrides[dep] = lambda: object()
    monkeypatch.setattr(
        admin, "ping_postgres", _up("postgres") if postgres_up else _down("postgres")
    )
    monkeypatch.setattr(admin, "ping_qdrant", _up("qdrant"))
    monkeypatch.setattr(admin, "ping_redis", _up("redis"))
    monkeypatch.setattr(admin, "ping_cerebras", _up("cerebras"))


def teardown_function():
    app.dependency_overrides.clear()


def test_root_and_public_health():
    assert client.get("/").json() == {"name": "k8-copilot", "status": "ok"}
    assert client.get("/health").json() == {"status": "ok"}


def test_admin_health_requires_auth():
    assert client.get("/admin/health").status_code == 401


def test_all_up_is_ok(monkeypatch):
    _use_fakes(monkeypatch)
    body = client.get("/admin/health").json()
    assert body["status"] == "ok"
    assert [s["service"] for s in body["services"]] == ["postgres", "qdrant", "redis", "cerebras"]


def test_one_down_is_degraded(monkeypatch):
    _use_fakes(monkeypatch, postgres_up=False)
    body = client.get("/admin/health").json()
    assert body["status"] == "degraded"
    assert body["services"][0] == {
        "service": "postgres",
        "status": "down",
        "latency_ms": None,
        "error": "ConnectError",
    }


async def test_ping_reports_exception_type_only():
    class Exploding:
        async def get_collections(self):
            raise RuntimeError("postgresql://user:SECRET@host/db")

    status = await admin.ping_qdrant(Exploding())  # type: ignore[arg-type]
    assert status.status == "down"
    assert status.error == "RuntimeError"  # the message (which could hold secrets) is dropped


async def test_ping_times_out(monkeypatch):
    monkeypatch.setattr(admin, "PING_TIMEOUT_SECONDS", 0.05)

    class Slow:
        async def ping(self):
            await asyncio.sleep(60)

    status = await admin.ping_redis(Slow())  # type: ignore[arg-type]
    assert (status.status, status.error) == ("down", "TimeoutError")


async def test_redis_not_configured():
    status = await admin.ping_redis(None)
    assert (status.status, status.error) == ("down", "NotConfigured")
