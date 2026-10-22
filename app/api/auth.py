import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, LiteralString, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from app.config import settings
from app.middleware.auth import (
    User,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.middleware.rate_limiter import RateLimiter, rate_limit_by_ip

# Single source of truth for the users table; the seed script runs the same file.
USERS_MIGRATION = (
    Path(__file__).resolve().parents[2] / "seed" / "migrations" / "001_create_users.sql"
)

# Verified against when the username doesn't exist, so unknown and known users take the
# same bcrypt time and usernames can't be enumerated by response timing.
_DUMMY_PASSWORD_HASH = hash_password("timing-equalizer-not-a-real-password")

router = APIRouter(prefix="/auth", tags=["auth"])

# Per-IP limits on the unauthenticated endpoints: brute-force and signup-spam protection.
login_limiter = RateLimiter(
    max_requests=settings.auth_login_rate_limit_per_min, window_seconds=60, name="login"
)
register_limiter = RateLimiter(
    max_requests=settings.auth_register_rate_limit_per_hour, window_seconds=3600, name="register"
)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)  # bcrypt truncates past 72 bytes


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


async def ensure_users_table(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute(cast(LiteralString, USERS_MIGRATION.read_text(encoding="utf-8")))


def get_pool(request: Request) -> AsyncConnectionPool:
    return request.app.state.db_pool


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_by_ip(register_limiter))],
)
async def register(
    body: RegisterRequest,
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> UserOut:
    # bcrypt is CPU-heavy (~250 ms); a worker thread keeps the event loop free.
    password_hash = await asyncio.to_thread(hash_password, body.password)
    try:
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                "RETURNING id, username, is_admin, created_at",
                (body.username, password_hash),
            )
            row = await cur.fetchone()
    except UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Username already taken"
        ) from None
    logger.info("Registered user {}", body.username)
    return UserOut.model_validate(row)


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit_by_ip(login_limiter))],
)
async def login(
    body: LoginRequest,
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> TokenResponse:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = %s",
            (body.username,),
        )
        row = await cur.fetchone()
    stored_hash = row["password_hash"] if row else _DUMMY_PASSWORD_HASH
    password_ok = await asyncio.to_thread(verify_password, body.password, stored_hash)
    if row is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(row["id"], row["username"], row["is_admin"])
    return TokenResponse(access_token=token, expires_in=settings.jwt_expiration_minutes * 60)


@router.get("/me", response_model=UserOut)
async def me(
    user: Annotated[User, Depends(get_current_user)],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> UserOut:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT id, username, is_admin, created_at FROM users WHERE id = %s",
            (user.id,),
        )
        row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserOut.model_validate(row)
