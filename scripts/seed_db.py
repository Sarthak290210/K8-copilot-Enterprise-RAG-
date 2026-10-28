"""Seed Postgres (migrations + users) and Qdrant (document ingestion).

Run from the repo root with `make seed` (or `uv run python scripts/seed_db.py`):
    --docs-only             skip Postgres, only ingest documents
    --skip-docs             only Postgres
    --noise N|all           how many noisy_data files to sample (default 150)
    --noise-max-pages N     keep only the first N pages of each noise PDF (default 30, 0 = all)
    --skip-existing         don't re-ingest documents already in Qdrant (resume a long run)

Safe to re-run: 001 uses CREATE TABLE IF NOT EXISTS, 002 drops and recreates its own
tables, 003 re-grants the read-only role (and LOGIN is set from COPILOT_RO_PASSWORD), 004
adds query_threads if missing, existing users are left untouched, and documents
are upserted with deterministic IDs (re-ingesting overwrites, never duplicates).
"""

import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import LiteralString, cast

import psycopg
from loguru import logger
from psycopg import sql
from pydantic import SecretStr

from app.config import settings
from app.middleware.auth import hash_password

DATABASE_URL = settings.database_url.get_secret_value()
SEED_DIR = Path(__file__).resolve().parent.parent / "seed"
MIGRATIONS_DIR = SEED_DIR / "migrations"
DOCS_DIR = SEED_DIR / "docs"

NOISE_SEED = 42  # fixed, so every run (and every eval) uses the same noise sample
DEFAULT_NOISE_SAMPLE = 150
DEFAULT_NOISE_MAX_PAGES = 30  # Docling ~2.3 s/page on CPU; long books would take hours

# Passwords come from .env (SEED_ADMIN_PASSWORD / SEED_ANALYST_PASSWORD), never the repo.
SEED_USERS = [
    {"username": "admin", "password": settings.seed_admin_password, "is_admin": True},
    {"username": "analyst", "password": settings.seed_analyst_password, "is_admin": False},
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


def enable_readonly_login(conn: psycopg.Connection) -> None:
    """Give the read-only Text2SQL role (003) LOGIN with COPILOT_RO_PASSWORD from .env.

    The password can't live in a committed .sql file, so it's set here; psycopg's sql.Literal
    quotes it safely (DDL can't take %s parameters). Skipped with a warning if it isn't set.
    """
    password = settings.copilot_ro_password
    if password is None:
        logger.warning("COPILOT_RO_PASSWORD is not set: copilot_ro stays NOLOGIN (no Text2SQL)")
        return
    conn.execute(
        sql.SQL("ALTER ROLE copilot_ro WITH LOGIN PASSWORD {}").format(
            sql.Literal(password.get_secret_value())
        )
    )
    logger.info("copilot_ro can log in (read-only, ops tables only)")


def seed_users(conn: psycopg.Connection) -> None:
    with conn.transaction():
        for user in SEED_USERS:
            password = cast(SecretStr | None, user["password"])
            if password is None:
                logger.warning(
                    "Skipping user {}: its seed password is not set in .env", user["username"]
                )
                continue
            # DO NOTHING: re-seeding must not reset a password someone has since changed.
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s) "
                "ON CONFLICT (username) DO NOTHING",
                (user["username"], hash_password(password.get_secret_value()), user["is_admin"]),
            )
            if cur.rowcount:
                logger.info("Seeded user {} (admin={})", user["username"], user["is_admin"])
            else:
                logger.info("User {} already exists; left unchanged", user["username"])


def _collect_files(subdir: str) -> list[Path]:
    """Supported files under seed/docs/<subdir>, sorted (stable order across runs); none if the
    folder doesn't exist (seed/docs isn't in git, so attack_data may be missing)."""
    from app.services.document_processor import SUPPORTED_EXTENSIONS

    folder = DOCS_DIR / subdir
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _select_corpus(noise_sample_size: int | str) -> tuple[list[Path], list[Path]]:
    """All true_data and attack_data files + a seeded random sample of noisy_data ("all" =
    every file). attack_data (Phase 6) holds test documents with planted instructions, for the
    indirect prompt-injection golden; it's ingested whole, like the signal docs."""
    signal = _collect_files("true_data") + _collect_files("attack_data")
    noise = _collect_files("noisy_data")
    if noise_sample_size != "all" and int(noise_sample_size) < len(noise):
        noise = sorted(random.Random(NOISE_SEED).sample(noise, int(noise_sample_size)))
    return signal, noise


async def _ingest_one(
    processor,
    src: Path,
    idx: int,
    total: int,
    counters: dict[str, int],
    max_pages: int | None = None,
    replace: bool = False,
) -> int | None:
    """Process -> embed -> upsert one file; returns the chunk count, or None on failure.
    Errors are counted, never fatal. `replace` deletes the source's old chunks first (a changed
    file must not leave stale chunks behind)."""
    from app.services import embedding_service, vector_store

    start = time.perf_counter()
    try:
        chunks = processor.process_document(src, max_pages=max_pages)
        if not chunks:
            raise ValueError("no text extracted")
        vectors = embedding_service.embed_texts_sync([c["text"] for c in chunks])
        if replace:
            await vector_store.delete_source(src.name)
        await vector_store.upsert_chunks(chunks, vectors)
    except Exception as exc:
        counters["failed"] += 1
        logger.warning("[{}/{}] {} FAILED: {}: {}", idx, total, src.name, type(exc).__name__, exc)
        return None
    counters["files"] += 1
    counters["chunks"] += len(chunks)
    logger.info(
        "[{}/{}] {} -> {} chunks ({:.1f}s)",
        idx,
        total,
        src.name,
        len(chunks),
        time.perf_counter() - start,
    )
    return len(chunks)


async def _seed_docs(
    noise_sample_size: int | str, noise_max_pages: int | None, skip_existing: bool
) -> dict[str, int]:
    from app.services import vector_store
    from app.services.doc_cache_service import DocCacheService, compute_file_hash
    from app.services.document_processor import DocumentProcessor

    client = vector_store.get_client()
    try:
        await vector_store.ensure_collection()
        signal, noise = _select_corpus(noise_sample_size)
        # (file, page cap, is_noise)
        jobs = [(f, None, False) for f in signal] + [(f, noise_max_pages, True) for f in noise]
        existing = set(await vector_store.list_sources())
        if skip_existing:
            jobs = [job for job in jobs if job[0].name not in existing]
        logger.info(
            "Ingesting {} signal + {} noise files ({} to do; noise capped at {} pages)",
            len(signal),
            len(noise),
            len(jobs),
            noise_max_pages or "no",
        )
        # Noise PDFs skip Docling's table-structure model: on table-heavy manuals it took
        # 7+ min per 30 pages, and noise only needs to be realistic text, not parsed tables.
        processors: dict[bool, DocumentProcessor] = {}
        counters = {"files": 0, "chunks": 0, "failed": 0, "skipped": 0, "recorded": 0}
        doc_cache = DocCacheService()
        for idx, (src, max_pages, is_noise) in enumerate(jobs, start=1):
            # Dedup by content (Phase 5): the same bytes with the same page cap are skipped.
            content_hash = compute_file_hash(src)
            if await doc_cache.find_indexed(content_hash, max_pages) is not None:
                counters["skipped"] += 1
                logger.debug("[{}/{}] {} already ingested; skipped", idx, len(jobs), src.name)
                continue
            if src.name in existing and doc_cache.source_hash(src.name) is None:
                # Ingested before Phase 5 (no hash record yet): record its hash instead of
                # re-ingesting it (user's choice 2026-10-26; ~3 h of Docling otherwise).
                chunks = await vector_store.count_source(src.name)
                doc_cache.set_metadata(
                    content_hash,
                    {"source": src.name, "chunks": chunks, "max_pages": max_pages,
                     "recorded_without_ingest": True},
                )  # fmt: skip
                counters["recorded"] += 1
                continue
            if is_noise not in processors:
                processors[is_noise] = DocumentProcessor(table_structure=not is_noise)
            indexed = await _ingest_one(
                processors[is_noise], src, idx, len(jobs), counters, max_pages,
                replace=src.name in existing,
            )  # fmt: skip
            if indexed is not None:
                doc_cache.set_metadata(
                    content_hash, {"source": src.name, "chunks": indexed, "max_pages": max_pages}
                )
        logger.info(
            "Skipped {} already-ingested files; recorded hashes for {} older ingests",
            counters["skipped"],
            counters["recorded"],
        )
        counters["points_in_collection"] = await vector_store.count_points()
        return counters
    finally:
        await client.close()
        vector_store.get_client.cache_clear()


def seed_docs(
    noise_sample_size: int | str = DEFAULT_NOISE_SAMPLE,
    noise_max_pages: int | None = DEFAULT_NOISE_MAX_PAGES,
    skip_existing: bool = False,
) -> dict[str, int]:
    """Ingest seed/docs into Qdrant: chunk -> embed -> upsert. Returns counters."""
    counters = asyncio.run(_seed_docs(noise_sample_size, noise_max_pages, skip_existing))
    logger.info("Docs seeded: {}", counters)
    return counters


def print_summary(conn: psycopg.Connection) -> None:
    """Row counts per seeded table, so a run shows at a glance what's in the database."""
    tables = [
        "users",
        "clusters",
        "nodes",
        "deployments",
        "pods",
        "incidents",
        "alerts",
        "oncall_logs",
    ]
    for table in tables:
        # Table names come from the fixed list above, never from input.
        row = conn.execute(cast(LiteralString, f"SELECT count(*) FROM {table}")).fetchone()
        logger.info("  {:<12} {:>7} rows", table, row[0] if row else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Postgres and Qdrant.")
    parser.add_argument("--docs-only", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--noise", default=str(DEFAULT_NOISE_SAMPLE), help="N or 'all'")
    parser.add_argument("--noise-max-pages", type=int, default=DEFAULT_NOISE_MAX_PAGES)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.docs_only:
        with psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=10) as conn:
            run_migrations(conn)
            enable_readonly_login(conn)
            seed_users(conn)
            logger.info("Database seeded:")
            print_summary(conn)
    if not args.skip_docs:
        noise = args.noise if args.noise == "all" else int(args.noise)
        seed_docs(noise, args.noise_max_pages or None, args.skip_existing)


if __name__ == "__main__":
    main()
