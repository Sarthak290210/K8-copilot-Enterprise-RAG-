# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**k8-copilot** — an advanced RAG system over Kubernetes ops data: Text2SQL + core RAG + caching + LLM security, served via FastAPI with an optional Streamlit UI.

## Current state

The API foundation is built and runs in Docker: config, JWT auth, per-IP rate limiting on auth routes, admin health checks, Postgres seeding. Wiring verified end to end on 2026-10-21 (ruff + pyright clean, all routes tested live against the compose stack). The RAG / Text2SQL pipeline, caching layer, and LLM security are **not started yet**. No git commits yet. `.env` holds real secrets and is gitignored — keep it that way.

Check a file before assuming functionality exists.

## Layout

```
app/
  config.py              Settings (pydantic-settings) — every .env var, typed
  main.py                FastAPI app, lifespan (shared clients), public GET /health
  api/auth.py            /auth/register, /auth/login, /auth/me; get_pool dependency
  api/admin.py           admin-only health checks
  middleware/auth.py     password hashing, JWT, get_current_user, require_admin
  middleware/rate_limiter.py  Upstash client + sliding-window RateLimiter
scripts/
  seed_db.py             run seed/mitgrations/*.sql, upsert demo users
  serve.py               production entrypoint (uvicorn), used by the Dockerfile
seed/mitgrations/        (sic — misspelled dir name) 001_create_users.sql, 003_seed_k8s_ops.sql
Dockerfile, .dockerignore, docker-compose.yml
```

### `app/`

- `config.py` — `Settings` with a typed field for every `.env` var, plus `api_host` / `api_port` / `api_workers` (defaults `0.0.0.0` / `8000` / `1`). Secrets are `SecretStr` (use `.get_secret_value()`). `GROQ_API_KEY`, `DATABASE_URL`, `JWT_SECRET` are required (Pylance flags `Settings()` as missing args — suppressed with `# pyright: ignore[reportCallIssue]`, it's a false positive). Fixed choices are `Literal` (`STORAGE_BACKEND`, `RERANKER_BACKEND`, `LOG_LEVEL`). Import `settings` or `get_settings()`.
- `main.py` — `lifespan` uses an `AsyncExitStack` so everything opened is closed even if startup fails midway. Shared clients on `app.state`: `db_pool` (psycopg3 `AsyncConnectionPool` with explicit `connection_class=AsyncConnection` so type checkers resolve the pool type; opened with `wait=True, timeout=10` so startup fails fast if Postgres is down), `qdrant` (`AsyncQdrantClient`), `redis` (from `get_redis_client()`, or `None` if Upstash isn't configured), `groq` (`AsyncGroq`; `/openai/v1` is stripped from `GROQ_BASE_URL` because the SDK appends it). Creates the `users` table on startup. Configures loguru from `LOG_LEVEL` / `LOG_JSON`.
- `middleware/auth.py` — single source of auth logic: `User` (username, is_admin), `hash_password` / `verify_password` (passlib bcrypt, 12 rounds), `create_access_token(payload, expires_delta)` (adds `iat`/`exp`, HS256), `get_current_user` (HTTP Bearer → `User` from token claims, no DB hit; 401 if missing/invalid/expired), `require_admin` (403).
- `middleware/rate_limiter.py` — `get_redis_client()` (cached Upstash async client, shared with `main.py`), `RateLimiter(max_requests, window_seconds, name=...)`, and `rate_limit_by_ip(limiter)` FastAPI dependency (429 + `Retry-After`; `X-RateLimit-*` headers on success). Sliding window: one sorted set of timestamps per key (`ratelimit:{name}:{ip|user}:{id}`); ZREMRANGEBYSCORE + ZADD + ZCARD + EXPIRE + ZRANGE go in one Upstash pipeline (one HTTP request); `exec()` results are loosely typed, so they're `cast`/`int()`-ed explicitly. `is_allowed` / `is_allowed_ip` / `is_allowed_user` → `RateLimitResult(allowed, remaining, count, retry_after)`. Rejected requests also count. Fails open if Redis errors.
- `api/auth.py` — `POST /auth/register` (JSON; username 3–64 chars `[A-Za-z0-9_.-]`, password 8–72; returns the new user, not a token; 409 on duplicate; 3/hour per IP), `POST /auth/login` (form data; token carries `sub`=username and `is_admin`; unknown user and wrong password give the same 401; 5/min per IP), `GET /auth/me` (full user row from DB). `get_pool` returns the pool; routes borrow a connection with `async with pool.connection()`.
- `api/admin.py` — whole router requires admin. `GET /admin/health` runs `ping_postgres` / `ping_qdrant` / `ping_redis` / `ping_groq` concurrently with `asyncio.gather` and returns `HealthReport(status="ok"|"degraded", services=[ServiceStatus...])`. Per-service routes `/admin/health/{postgres,qdrant,redis,groq}` return one `ServiceStatus(service, status "up"|"down", latency_ms, error)`. Pings never raise, use a 3s timeout (`PING_TIMEOUT_SECONDS`), and report only the exception type (psycopg messages can leak the DB URL). Groq ping lists models (no tokens spent).

### Seed data (`scripts/seed_db.py`)

- Runs every `.sql` in `seed/mitgrations/` in filename order, one transaction per file (SQL cast to `LiteralString` for psycopg typing), then upserts demo users `admin` / `admin12345` (admin) and `demo` / `demo12345` (local-only credentials).
- `003_seed_k8s_ops.sql` is generated (don't hand-edit; generator `scripts/data_pipeline/03_generate_k8s_ops_db.py` doesn't exist yet). It **drops and recreates** its tables on every run: clusters (50), nodes (5k), deployments (10k), pods (50k), incidents (2k), alerts (100k), oncall_logs (20k). It inserts explicit IDs without `setval`, so new inserts into those tables would collide. It mentions a `002_seed_ecommerce.sql` that doesn't exist.
- The `users` table is defined twice (`001_create_users.sql` and `USERS_TABLE_DDL` in `api/auth.py`, slightly different types); whichever runs first wins.

## Running

```
docker compose up -d --build                           # postgres, qdrant, api
docker compose run --rm api python scripts/seed_db.py  # seed (or `make seed` from the host)
docker compose up -d --build api                       # after changing app code
docker compose logs -f api
docker compose down                                    # keep data; `down -v` deletes it
```

- API at http://localhost:8000 (Swagger at `/docs`; log in via `/auth/login`, then **Authorize** with the token).
- Compose overrides `DATABASE_URL` / `QDRANT_URL` with service names for the api container. `POSTGRES_PASSWORD` in `.env` must match the password in `DATABASE_URL` (user `postgres`, db `adv_rag`). Postgres 5432 and Qdrant 6333/6334 are published to the host, so `make api` / `make seed` on the host use the same databases.
- Images: `postgres:17.9-alpine`, `qdrant/qdrant:v1.19.1` (matches qdrant-client 1.19), API built from `Dockerfile` (python:3.12-slim + uv 0.13.0, non-root, healthcheck on `/health`). The API image is ~11 GB because sentence-transformers pulls GPU PyTorch; switching to CPU-only torch would shrink it a lot.
- Upstash Redis, Groq, Tavily, Voyage are cloud services configured via `.env`, not compose services.
- On Windows, async psycopg can't use the default `ProactorEventLoop`: `serve.py` and `make api` pass `asyncio:SelectorEventLoop` to uvicorn. Keep that when changing how the server starts.
- Docker Desktop on Windows depends on WSL 2; if the engine won't start, `wsl --shutdown` (or a reboot) usually fixes it.

## Known gaps / next steps

- RAG / Text2SQL pipeline, LangGraph orchestration, llm-guard checks: not started.
- No cache layer yet, so no cache stats/clear admin endpoints. No OpenAI or Tavily health pings.
- Rate limiting only on `/auth/login` and `/auth/register`. The general per-user limit (`RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`) and `MAX_TOKENS_PER_USER_DAILY` aren't wired to any route. Behind Docker's NAT every host request shares one IP (`172.18.0.1`).
- `X-RateLimit-*` headers are added on success and on 429, but not on other error responses (e.g. a wrong-password 401).
- The Dockerfile `HEALTHCHECK` hardcodes port 8000; setting `API_PORT` would make the container report unhealthy.
- Admin status is read from the JWT, so promotions/demotions apply only after re-login.
- Missing files referenced by the Makefile: `scripts/streamlit_app.py`, `scripts/validate_goldens.py`, `scripts/data_pipeline/`, `seed_docs()` in `seed_db.py` (for `make seed-docs`), `eval/`, `tests/`.
- `pyproject.toml` lists `sentence-transformers` twice and `psycopg2-binary` alongside psycopg 3 (app uses only psycopg 3; psycopg2 may be needed by vanna). `seed-docs` and `validate` are missing from `.PHONY`.

## Stack

- Python 3.12, managed with **uv** (`uv run ...`, never bare `pip`)
- Web: FastAPI + uvicorn
- LLM: Groq (OpenAI-compatible client), tiktoken for token budgeting
- Vector DB: Qdrant; embeddings via sentence-transformers; reranking via Voyage or local model
- Doc parsing: docling, pypdf
- Text2SQL: Vanna (OpenAI + Postgres) over Postgres (psycopg2 / psycopg3)
- Cache: Upstash Redis (TTLs per layer: embeddings, RAG, SQL gen, SQL result, intent); optional S3 storage
- Auth: JWT (pyjwt) + bcrypt (passlib), with rate limiting and per-user daily token caps
- Orchestration: LangGraph with Postgres checkpointing
- Security: llm-guard (prompt injection, toxicity on input/output)
- Web search fallback: Tavily (used by CRAG)
- Logging: loguru
- Eval: RAGAS

## Planned retrieval pipeline (from `.env` / Makefile)

Feature-flagged stages, evaluated as RAGAS profiles: `naive` → `hybrid` (dense + sparse, RRF fusion) → `hybrid+rerank` → `+hyde` (hypothetical doc embeddings) → `+crag` (corrective RAG with relevance thresholds + web fallback) → self-reflective answering with retry. Defaults for each are toggled via `*_ENABLED_BY_DEFAULT` env vars.

## Make targets

```
make install     # pin py3.12, create venv, uv sync --extra dev
make sync        # uv sync --extra dev
make api         # uvicorn app.main:app on :8000 (reload, selector loop)
make seed        # scripts/seed_db.py (works)
make lint        # ruff check .
make format      # ruff format .
make streamlit / seed-docs / seed-data / eval* / validate / test   # scripts not written yet — will fail
```

## Conventions

- Ruff: line length 100, target py312, rules `E,F,W,I,B,UP` (E501, B008 ignored — B008 allows FastAPI `Depends(...)` defaults).
- mypy with `ignore_missing_imports`. Pylance/pyright is what the editor shows; check the project with `uv run --with pyright pyright app scripts` (should be 0 errors).
- To list served routes, use `app.openapi()["paths"]` — this FastAPI version keeps included routers as `_IncludedRouter` entries, so iterating `app.routes` misses them.
- pytest with `asyncio_mode = "auto"`; tests live in `tests/`.
- All config comes from env vars via `app/config.py`; don't hard-code keys or model names.
- Shared clients are created once in `main.py`'s `lifespan` (registered on the `AsyncExitStack`) and read in routes via small `get_*(request)` dependencies — don't create new clients per request.
- Async everywhere for I/O (psycopg3 async pool, `AsyncQdrantClient`, Upstash async, `AsyncGroq`).
- Auth helpers live only in `app/middleware/auth.py`; routers import from there.
- SQL always uses `%s` placeholders.
- Error responses and logs must not leak secrets (report exception types, not messages, for DB errors).
- Never print or commit values from `.env`.
