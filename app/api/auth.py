from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
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

USERS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

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
        await conn.execute(USERS_TABLE_DDL)


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
    try:
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                "RETURNING id, username, is_admin, created_at",
                (body.username, hash_password(body.password)),
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
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> TokenResponse:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT username, password_hash, is_admin FROM users WHERE username = %s",
            (form.username,),
        )
        row = await cur.fetchone()
    if row is None or not verify_password(form.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires_in = settings.jwt_expiration_minutes * 60
    token = create_access_token(
        {"sub": row["username"], "is_admin": row["is_admin"]}, expires_delta=expires_in
    )
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=UserOut)
async def me(
    user: Annotated[User, Depends(get_current_user)],
    pool: Annotated[AsyncConnectionPool, Depends(get_pool)],
) -> UserOut:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT id, username, is_admin, created_at FROM users WHERE username = %s",
            (user.username,),
        )
        row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserOut.model_validate(row)
