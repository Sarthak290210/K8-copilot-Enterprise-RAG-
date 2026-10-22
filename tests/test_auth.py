from typing import Annotated

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.middleware.auth import (
    JWT_ALGORITHM,
    User,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)

# A tiny app with one protected and one admin route, so no database is needed.
app = FastAPI()


@app.get("/protected")
async def protected(user: Annotated[User, Depends(get_current_user)]) -> dict:
    return user.model_dump()


@app.get("/admin-only")
async def admin_only(_: Annotated[User, Depends(require_admin)]) -> dict:
    return {"ok": True}


client = TestClient(app)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_hash_verify_round_trip():
    hashed = hash_password("correct-horse")
    assert hashed.startswith("$2b$12$")  # bcrypt, 12 rounds
    assert hashed != "correct-horse"
    assert verify_password("correct-horse", hashed)
    assert not verify_password("wrong-horse", hashed)


def test_same_password_gets_different_salts():
    assert hash_password("same-password") != hash_password("same-password")


def test_token_payload():
    token = create_access_token(7, "alice", is_admin=True)
    payload = jwt.decode(token, settings.jwt_secret.get_secret_value(), algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == "7"  # JWT spec: sub is a string
    assert payload["username"] == "alice"
    assert payload["is_admin"] is True
    assert payload["exp"] - payload["iat"] == settings.jwt_expiration_minutes * 60


def test_valid_token_returns_user():
    r = client.get("/protected", headers=bearer(create_access_token(7, "alice", False)))
    assert r.status_code == 200
    assert r.json() == {"id": 7, "username": "alice", "is_admin": False}


def test_missing_token_is_401():
    assert client.get("/protected").status_code == 401


def test_expired_token_is_401():
    token = create_access_token(7, "alice", False, expires_minutes=-1)
    r = client.get("/protected", headers=bearer(token))
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid or expired token"


def test_tampered_token_is_401():
    token = create_access_token(7, "alice", is_admin=False)
    forged = jwt.encode(
        {"sub": "7", "username": "alice", "is_admin": True},
        "an-attacker-key-that-is-not-our-secret",
    )
    assert client.get("/admin-only", headers=bearer(forged)).status_code == 401
    assert client.get("/admin-only", headers=bearer(token)).status_code == 403


@pytest.mark.parametrize(("is_admin", "status"), [(False, 403), (True, 200)])
def test_require_admin(is_admin, status):
    token = create_access_token(1, "bob", is_admin=is_admin)
    assert client.get("/admin-only", headers=bearer(token)).status_code == status
