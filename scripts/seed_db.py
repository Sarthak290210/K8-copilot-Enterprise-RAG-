"""Seed Postgres: run the SQL migrations in seed/mitgrations/, then insert demo users.

Run from the repo root with `make seed` (or `uv run python scripts/seed_db.py`).
Safe to re-run: 001 uses CREATE TABLE IF NOT EXISTS, 003 drops and recreates its own
tables, and demo users are upserted.
"""

from pathlib import Path
from typing import LiteralString, cast

import psycopg
from loguru import logger

from app.config import settings
from app.middleware.auth import hash_password

DATABASE_URL = settings.database_url.get_secret_value()
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "seed" / "mitgrations"

# Local/demo credentials only — never use these outside a dev database.
DEMO_USERS = [
    {"username": "admin", "password": "admin12345", "is_admin": True},
    {"username": "demo", "password": "demo12345", "is_admin": False},
]


def run_migrations(conn: psycopg.Connection) -> None:
    """Run every .sql file in MIGRATIONS_DIR in filename order, one transaction per file."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {MIGRATIONS_DIR}")

    for path in files:
        logger.info("Running migration {}", path.name)
        # Trusted, repo-controlled SQL; cast satisfies psycopg's LiteralString typing.
        sql = cast(LiteralString, path.read_text(encoding="utf-8"))
        # No parameters, so psycopg sends the whole file (many statements) in one go.
        with conn.transaction():
            conn.execute(sql)
    logger.info("Ran {} migration(s)", len(files))


def seed_users(conn: psycopg.Connection) -> None:
    with conn.transaction():
        for user in DEMO_USERS:
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s) "
                "ON CONFLICT (username) DO UPDATE "
                "SET password_hash = EXCLUDED.password_hash, is_admin = EXCLUDED.is_admin",
                (user["username"], hash_password(user["password"]), user["is_admin"]),
            )
            logger.info("Seeded user {} (admin={})", user["username"], user["is_admin"])


def main() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=10) as conn:
        run_migrations(conn)
        seed_users(conn)
    logger.info("Database seeded")


if __name__ == "__main__":
    main()
