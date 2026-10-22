# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**k8-copilot** — an advanced RAG system over Kubernetes ops data: Text2SQL + core RAG + caching + LLM security, served via FastAPI with an optional Streamlit UI.

## Current state

**Phase 0 of `plan.md` is complete (2026-10-22)**: API foundation in Docker (config, JWT auth with user id, per-IP rate limiting with in-memory fallback, admin health checks, `GET /`), migrations + seeding (incl. read-only role `copilot_ro`), eval harness (`eval/`, 44 goldens), 57 tests, and checkpoint E0 recorded. All Phase 0 Done-when checks passed live. **Phase 1 (basic dense RAG) is built** (2026-10-22): models, Groq LLM service, bge embeddings, Qdrant vector store, Docling document processor, RAG service, `/query`, `/documents`, doc ingestion in the seed script, Streamlit UI, eval invoker wired to RAG; 104 tests pass. Noise ingest done (148/150 PDFs, 7,285 chunks) and **E1 = the baseline recorded** (2026-10-22, see Evaluation). LLM provider switched from Groq to Cerebras. Remaining for Phase 1: rebuild the api image (new `openai` dependency) and run the live Done-when checks (/query, /documents/upload, Streamlit). Text2SQL, caching and LLM security are **not started yet**. Phases 2/3 are not to be started before E1 is recorded (user agreed: E1 must measure pure Phase 1 code). `plan.md` is the build plan (one phase at a time; wait for the user's "go"). `.env` holds real secrets and is gitignored — keep it that way; `.env.example` is the committed template.

Check a file before assuming functionality exists.

## Layout

```
app/
  config.py              Settings (pydantic-settings) — every .env var, typed
  main.py                FastAPI app, lifespan (shared clients), public GET / and GET /health
  models.py              RetrievedChunk, RetrievedChunkPreview, ResponseMetadata, ChatResponse, QueryRequest
  api/query.py           POST /query (logged-in users, per-user rate limit) -> rag_service.run_rag
  api/documents.py       POST /documents/upload (admin), GET /documents (any user)
  services/llm_service.py        Cerebras chat (OpenAI-compatible): generate / generate_with_json
  services/embedding_service.py  local bge: embed_texts / embed_query (query instruction prefix)
  services/vector_store.py       Qdrant: ensure_collection, upsert_chunks, search, delete_source, list_sources
  services/document_processor.py Docling convert + HybridChunker (512 tokens)
  services/rag_service.py        retrieve -> numbered context -> answer; run_rag / run_rag_with_trace
  security/system_prompt.py      basic system prompt (hardened in Phase 6)
  api/auth.py            /auth/register, /auth/login, /auth/me; get_pool dependency
  api/admin.py           admin-only health checks
  middleware/auth.py     password hashing, JWT, get_current_user, require_admin
  middleware/rate_limiter.py  Upstash client + sliding-window RateLimiter
scripts/
  seed_db.py             migrations + seed users (Postgres) and seed_docs() ingestion (Qdrant)
  serve.py               production entrypoint (uvicorn), used by the Dockerfile
  streamlit_app.py       UI: login/register, ask (with flag toggles), upload, eval dashboard
seed/migrations/         001_create_users.sql, 002_k8s_ops.sql (generated data), 003_readonly_role.sql
seed/docs/true_data/     47 real K8s docs (pdf/docx/html/txt) — the signal corpus (committed)
seed/docs/noisy_data/    820 unrelated PDFs — the noise corpus (gitignored, keep on disk)
eval/                    eval harness: golden_questions.yaml, schema, profiles, invokers,
                         ragas_adapter, post_checks, reporting, run_ragas, diff, record, results/
tests/                   pytest suite (104 tests; conftest isolates from the real .env, HF offline)
plan.md                  phase-by-phase build plan (Phases 0-7) + eval protocol
Dockerfile, .dockerignore, docker-compose.yml, Makefile
.env.example             template for .env (placeholders for secrets, real non-secret defaults)
```

### `app/`

- `config.py` — `Settings` with a typed field for every `.env` var, plus `api_host` / `api_port` / `api_workers` (defaults `0.0.0.0` / `8000` / `1`). Secrets are `SecretStr` (use `.get_secret_value()`). `CEREBRAS_API_KEY`, `DATABASE_URL`, `JWT_SECRET` are required, and `JWT_SECRET` must be ≥ 32 chars or startup fails (`field_validator`). `SEED_ADMIN_PASSWORD` / `SEED_ANALYST_PASSWORD` are optional (used only by the seed script). (Pylance flags `Settings()` as missing args — suppressed with `# pyright: ignore[reportCallIssue]`, it's a false positive). Fixed choices are `Literal` (`STORAGE_BACKEND`, `RERANKER_BACKEND`, `LOG_LEVEL`). Import `settings` or `get_settings()`.
- `main.py` — `lifespan` uses an `AsyncExitStack` so everything opened is closed even if startup fails midway. Shared clients on `app.state`: `db_pool` (psycopg3 `AsyncConnectionPool` with explicit `connection_class=AsyncConnection` so type checkers resolve the pool type; opened with `wait=True, timeout=10` so startup fails fast if Postgres is down), `qdrant` (`AsyncQdrantClient`), `redis` (from `get_redis_client()`, or `None` if Upstash isn't configured), `llm` (`AsyncOpenAI` pointed at `CEREBRAS_BASE_URL`). Creates the `users` table on startup by running `seed/migrations/001_create_users.sql` (the only definition of that table), and runs `vector_store.ensure_collection()`. Qdrant and LLM clients come from `vector_store.get_client()` / `llm_service.get_client()`; on shutdown they're closed and their caches cleared. Routers: auth, admin, query, documents. On shutdown it closes the Redis client and clears `get_redis_client`'s cache so a later startup gets a fresh client. Configures loguru from `LOG_LEVEL` / `LOG_JSON`.
- `middleware/auth.py` — single source of auth logic: `User` (id, username, is_admin), `hash_password` / `verify_password` (passlib bcrypt, 12 rounds, ~250 ms of CPU — call them via `asyncio.to_thread` in async routes), `create_access_token(user_id, username, is_admin, expires_minutes=None)` (payload `{sub: str(user_id), username, is_admin, iat, exp}`, HS256 — `sub` must be a string per the JWT spec), `get_current_user` (HTTP Bearer → `User(id=int(sub), ...)` from token claims, no DB hit; 401 if missing/invalid/expired/tampered), `require_admin` (403).
- `middleware/rate_limiter.py` — `get_redis_client()` (cached Upstash async client, shared with `main.py`), `RateLimiter(max_requests, window_seconds, name=...)`, and `rate_limit_by_ip(limiter)` FastAPI dependency (429 + `Retry-After`; `X-RateLimit-*` headers on success). Sliding window: one sorted set of timestamps per key (`ratelimit:{name}:{ip|user}:{id}`); ZREMRANGEBYSCORE + ZADD + ZCARD + EXPIRE + ZRANGE go in one Upstash pipeline (one HTTP request); `exec()` results are loosely typed, so they're `cast`/`int()`-ed explicitly. `is_allowed` / `is_allowed_ip` / `is_allowed_user` → `RateLimitResult(allowed, remaining, count, retry_after)`. Rejected requests also count. Fails open if Redis errors or takes longer than `REDIS_TIMEOUT_SECONDS` (2 s; the Upstash client has no timeout of its own, and its retry is disabled). Limiters resolve the client per call (`RateLimiter.redis` property), never caching a closed one. If Upstash isn't configured, the same sliding window runs in memory (`_memory_windows`, per process — not shared across workers/restarts; tests use it).
- `api/auth.py` — `POST /auth/register` (JSON; username 3–64 chars `[A-Za-z0-9_.-]`, password 8–72; returns the new user, not a token; 409 on duplicate; 3/hour per IP), `POST /auth/login` (JSON `LoginRequest {username, password}` — form data is rejected with 422; returns `{access_token, token_type, expires_in}`; unknown user and wrong password give the same 401 and take the same time — unknown users are checked against a dummy hash; bcrypt runs in a worker thread; 5/min per IP), `GET /auth/me` (full user row from DB, looked up by the token's user id). `get_pool` returns the pool; routes borrow a connection with `async with pool.connection()`.
- `models.py` — `RetrievedChunk(text, source, score, metadata)`, `RetrievedChunkPreview.from_chunk` (200-char snippet), `ResponseMetadata(intent, flags_used, latency_ms, tokens_used, cache_hits, model)`, `ChatResponse(answer, sources, confidence=None, chunks, metadata)`, `QueryRequest(question 1–2000 chars, flags)`. `confidence` stays None until real graders exist (Phase 3) — never hard-code it.
- `services/llm_service.py` — the only LLM caller (Cerebras via the `openai` SDK). `get_client()` is an `lru_cache`d `AsyncOpenAI` (max_retries=6: the SDK backs off on 429 and honours Retry-After; timeout 60 s), shared with `main.py`. `generate(system, user, model=None, temperature=0, json_mode=False)` → `{content, model, usage}`; `generate_with_json` → `{data, model, usage}` (raises `LLMJSONError`; returns usage too because Phase 6 sums actual tokens). Sends `reasoning_effort=settings.llm_reasoning_effort` (`LLM_REASONING_EFFORT`, default `low`) only to gpt-oss models (`is_reasoning_model`); optional params are omitted rather than sent as JSON null. Cerebras returns gpt-oss reasoning in a separate field, never in `content`.
- `services/embedding_service.py` — `get_model()` loads bge once (cuda if available, else cpu), **local cache first** (`local_files_only=True`, network only if not cached — DNS here is flaky). `embed_texts` (async, runs in a thread; normalised vectors) / `embed_texts_sync` (seed script) / `embed_query` (prepends the bge query instruction `"Represent this sentence for searching relevant passages: "` — queries only, never documents).
- `services/vector_store.py` — `get_client()` cached `AsyncQdrantClient` shared with `main.py` (`app.state.qdrant`). `ensure_collection()` (768-dim cosine + keyword index on `source`; run at startup), `upsert_chunks()` (point ID = uuid5(source::chunk_index): re-ingesting overwrites, never duplicates; batches of 256), `search()`, `delete_source()`, `list_sources()` (facet on `source`; needs server Qdrant — local `:memory:` mode ignores payload indexes), `count_points()`.
- `services/document_processor.py` — `DocumentProcessor(table_structure=True)`: Docling with **OCR off** (corpus PDFs are born-digital), `HybridChunker` with the bge tokenizer, max 512 tokens, `merge_peers=True`; chunk text = `chunker.contextualize()` (heading path + text). `.txt` has no Docling reader, so it's fed as an in-memory `.md` stream (source keeps the `.txt` name). `process_document(path, source=None, max_pages=None)` → `[{text, source, metadata{chunk_index, headings, page}}]`; `max_pages` → Docling `page_range`. Measured on this CPU: PDF ~2.3–2.8 s/page (tables off only saves ~20%), html 0.3 s, docx ~2 s; ~1.8 chunks/page, ~1.5k chars/chunk.
- `services/rag_service.py` — `DEFAULT_FLAGS = {search_mode: dense, top_k: 5}`; `_flag()` safe read; `_retrieve` (Phase 1: dense only — any other `search_mode` raises `ValueError` → 400); `_build_context` numbers chunks `[n] (source: x)`; `_generate`; `run_rag_with_trace` returns `(ChatResponse, full chunks)`; no chunks → `NOT_FOUND_ANSWER` without an LLM call; sources deduped in retrieval order. **Later phases extend `_retrieve` behind flags; the `naive` flags must keep running exactly this dense path** (Phase 2 adds a test for that, and `naive` is re-run after Phases 2/3 to confirm it still matches E1 within ±0.02).
- `security/system_prompt.py` — basic prompt: answer only from context, cite as plain `[1]`/`[2][3]` (gpt-oss otherwise uses its own `【1†L1-L7】` style), say you don't know otherwise.
- `api/query.py` — `POST /query` (`QueryRequest` → `ChatResponse`): `get_current_user`, per-user `RateLimiter(name="query")` (`RATE_LIMIT_REQUESTS`/`RATE_LIMIT_WINDOW_SECONDS`, 429 + Retry-After), `ValueError` → 400.
- `api/documents.py` — `POST /documents/upload` (admin; extension in `SUPPORTED_EXTENSIONS`, ≤ 20 MB → 413, empty → 400; only the base file name is used; Docling runs in a thread; old chunks of the same source are deleted before upsert; unreadable → 422), `GET /documents` (any user; distinct sources). Docling loads once (`get_processor`, lru_cache) on first upload.
- `api/admin.py` — whole router requires admin. `GET /admin/health` runs `ping_postgres` / `ping_qdrant` / `ping_redis` / `ping_cerebras` concurrently with `asyncio.gather` and returns `HealthReport(status="ok"|"degraded", services=[ServiceStatus...])`. Per-service routes `/admin/health/{postgres,qdrant,redis,cerebras}` return one `ServiceStatus(service, status "up"|"down", latency_ms, error)`. Pings never raise, use a 3s timeout (`PING_TIMEOUT_SECONDS`), and report only the exception type (psycopg messages can leak the DB URL). The Cerebras ping lists models (no tokens spent).

### Seed data (`scripts/seed_db.py`)

- **Docs (Qdrant)**: `seed_docs(noise_sample_size=150, noise_max_pages=30, skip_existing=False)` ingests **all** `true_data` files, then a **seeded random sample** (`NOISE_SEED = 42`, so every run/eval uses the same files) of `noisy_data`, each noise PDF capped at its first 30 pages (user's choice 2026-10-22: Docling would need ~7 h for the uncapped 150). Per file: process → embed → upsert, errors counted per file, never fatal. CLI: `--docs-only`, `--skip-docs`, `--noise N|all`, `--noise-max-pages N` (0 = all pages), `--skip-existing` (resume: skips sources already in Qdrant). Ingesting from the host uses Qdrant on 127.0.0.1:6333.
- Ingest results: true_data = 47 files → 653 chunks in ~5 min, 0 failures. Noise (150 PDFs, 30-page cap): ~1–2 min per PDF on this CPU (slower when other CPU-heavy jobs — tests, Docker builds — run at the same time; avoid that during an ingest). Qdrant's on-disk size is several times the raw vector size (WAL + unoptimised segments): ~374 MB after true_data + 12 noise PDFs.

- Runs every `.sql` in `seed/migrations/` in filename order, one transaction per file (SQL cast to `LiteralString` for psycopg typing), then inserts `admin` (admin) and `analyst` with passwords from `SEED_ADMIN_PASSWORD` / `SEED_ANALYST_PASSWORD` in `.env` (a user is skipped with a warning if unset). `ON CONFLICT DO NOTHING`: re-seeding never resets a changed password. Then logs row counts per table and calls the `seed_docs()` stub (Phase 1). Dev DB users: `admin` and `analyst` only (the old `demo` user was deleted; `admin` uses `SEED_ADMIN_PASSWORD`).
- `002_k8s_ops.sql` (renamed from `003_seed_k8s_ops.sql`; its header comment still says 003) is generated (don't hand-edit; generator `scripts/data_pipeline/03_generate_k8s_ops_db.py` doesn't exist yet). It **drops and recreates** its tables on every run: clusters (50), nodes (5k), deployments (10k), pods (50k), incidents (2k), alerts (100k), oncall_logs (20k). It inserts explicit IDs without `setval`, so new inserts into those tables would collide. It mentions a `002_seed_ecommerce.sql` that doesn't exist.
- `003_readonly_role.sql` creates role `copilot_ro` (NOLOGIN for now; Phase 4 adds LOGIN + a password) with `SELECT` on the 7 ops tables only — explicitly no access to `users`. It re-grants on every seed because 002 recreates the tables. Verified: SELECT on incidents works, SELECT on users and DELETE are denied.
- The ops data: incidents span 2025-05-16..2026-05-16 (all resolved), so "this month"-style questions return 0 rows; incidents have no category/SLA column; cluster names look like `k8s-production-useast1-001`.
- The `users` table is defined only in `001_create_users.sql` (`BIGSERIAL` id, `TEXT` username — matches the live DB; plan.md's `SERIAL`/`VARCHAR(64)` was not used). Foreign keys to `users.id` must be `BIGINT`.

## Git / GitHub

- Remote: `origin` → https://github.com/Sarthak290210/K8-copilot-Enterprise-RAG- (public; note capital `K` and trailing `-` in the repo name). Default branch `main` (renamed from `master`), tracking `origin/main`.
- History: `ae5071f feat: scaffold k8-copilot API foundation` (pushed 2026-10-21), then `568b58a feat: phase 0-1 - basic dense RAG, eval harness, E1 baseline` (70 files, pushed 2026-10-22). Use conventional-commit style messages (`feat:`, `fix:`, `docs:` ...).
- Not committed yet: `seed/docs/` (both `true_data/` and `noisy_data/`; user: "we will see that later"; noisy_data is third-party copyrighted PDFs, so a public repo should get a manifest of the sampled file names, not the files). `eval/results/*.json` **are committed** (user's choice). Never commit `.env`, `.venv/`, `__pycache__/`, `.ruff_cache/` (all ignored). Note: the local `POSTGRES_PASSWORD` is the default word `postgres` (ports are bound to 127.0.0.1).
- Git on this machine has `core.autocrlf=true`, so `git add` warns "LF will be replaced by CRLF". Harmless for the repo (stored as LF); adding a `.gitattributes` with `* text=auto eol=lf` would keep LF in the working tree for files copied into Linux containers. Not added yet.
- The repo is public: anything committed is visible. Before committing, check `git status` shows no `.env`.

## Running

```
docker compose up -d --build                           # postgres, qdrant, api
docker compose run --rm api python scripts/seed_db.py  # seed (or `make seed` from the host)
docker compose up -d --build api                       # after changing app code
docker compose logs -f api
docker compose down                                    # keep data; `down -v` deletes it
```

- All published ports bind to `127.0.0.1` (this machine only; Qdrant has no auth).
- `.env` uses `127.0.0.1` (not `localhost`) in `DATABASE_URL` / `QDRANT_URL`: on Windows `localhost` tries IPv6 `::1` first, which hangs ~2 min since Docker binds IPv4 only.
- API at http://localhost:8000 (Swagger at `/docs`; log in via `/auth/login`, then **Authorize** with the token).
- Compose overrides `DATABASE_URL` / `QDRANT_URL` with service names for the api container. `POSTGRES_PASSWORD` in `.env` must match the password in `DATABASE_URL` (user `postgres`, db `adv_rag`). Postgres 5432 and Qdrant 6333/6334 are published to the host, so `make api` / `make seed` on the host use the same databases.
- `torch` **and `torchvision`** are pinned to the `pytorch-cpu` index on Linux. If only torch is CPU-pinned, the container crash-loops at startup with `operator torchvision::nms does not exist` (docling imports torchvision via transformers).
- Images: `postgres:17.9-alpine`, `qdrant/qdrant:v1.19.1` (matches qdrant-client 1.19), API built from `Dockerfile` (python:3.12-slim + uv 0.13.0, `uv sync --frozen`, non-root `appuser`, healthcheck on `/health`). Torch is CPU-only on Linux via the `pytorch-cpu` uv index in `pyproject.toml` (`torch` is a direct dependency so the source pin applies); Windows uses the PyPI wheel. The Dockerfile pre-creates `/home/appuser/.cache/huggingface` owned by `appuser` so the `hf_cache` volume is writable.
- `seed/docs/` is mounted read-only into the api container (`./seed/docs:/app/seed/docs:ro`). `seed/docs/noisy_data/` (801 MB, 820 PDFs — the noise dataset, **keep it, don't delete**) is in `.gitignore` and `.dockerignore`: it stays on disk but isn't pushed or baked into the image. `true_data/` (1.3 MB) is committed.
- `scripts/serve.py` trusts `X-Forwarded-For` only from uvicorn's default `127.0.0.1` (set `FORWARDED_ALLOW_IPS` when a real proxy is added). Never use `"*"`: clients could spoof IPs and bypass the per-IP rate limits.
- Upstash Redis, Cerebras (LLM), Tavily, Voyage are cloud services configured via `.env`, not compose services.
- On Windows, async psycopg can't use the default `ProactorEventLoop`: `serve.py` and `make api` pass `asyncio:SelectorEventLoop` to uvicorn. Keep that when changing how the server starts.
- **Hugging Face model cache is on D:** `HF_HOME=D:\hf-cache` (user env var set with `setx` on 2026-10-23, user's request: C: had only 12 GB free). The old `C:\Users\Sarthak\.cache\huggingface` (1.5 GB: bge, docling, rerankers) was moved there. Terminals opened before that don't see the variable: prefix commands with `HF_HOME=D:/hf-cache` there, or the models re-download to C:. The Docker api container uses its own `hf_cache` volume.
- DNS on this machine is occasionally flaky (seen for huggingface.co on the host and in the container, and as a one-off `uv sync` build failure). Retry before debugging; right after the api container starts, the first `/admin/health` can show redis/groq `down` until DNS warms up.
- Docker Desktop on Windows depends on WSL 2; if the engine won't start, `wsl --shutdown` (or a reboot) usually fixes it.

## Code review fixes (2026-10-22)

A `/code-review` of the whole codebase found 10 issues; all were fixed and verified against the rebuilt compose stack (ruff + pyright clean). Don't reintroduce them:

| # | Issue | Fix | Verified |
|---|---|---|---|
| 1 | `forwarded_allow_ips="*"` let any client spoof `X-Forwarded-For` and bypass per-IP rate limits | `serve.py` trusts forwarded headers only from 127.0.0.1 (`FORWARDED_ALLOW_IPS` env var to change) | 7 logins with 5 spoofed IPs → all counted on the real IP, 6th → 429 |
| 2 | bcrypt (~250 ms) blocked the event loop; unknown usernames returned faster (enumeration by timing) | `asyncio.to_thread` for hash/verify; unknown users verified against `_DUMMY_PASSWORD_HASH` | Real user vs unknown user ≈ 0.48 s both |
| 3 | `hf_cache` volume was root-owned, so `appuser` couldn't download models | Dockerfile pre-creates `/home/appuser/.cache/huggingface` owned by `appuser` | `touch` in the container works |
| 4 | 801 MB `noisy_data` not ignored by git or Docker | Added to `.gitignore` / `.dockerignore`; mounted `./seed/docs:/app/seed/docs:ro` | 820 files visible in container, writes fail (read-only) |
| 5 | Upstash `pipe.exec()` had no timeout (+3 s retry); closed cached client reused after a second lifespan | `asyncio.wait_for(..., REDIS_TIMEOUT_SECONDS=2)`, `rest_retries=0`; limiter resolves client per call; lifespan clears the cache on shutdown | Hung Redis → fails open after 2.0 s |
| 6 | No minimum length for `JWT_SECRET` | `field_validator` in `config.py`: < 32 chars → startup error | Short secret rejected, 32 chars accepted |
| 7 | GPU PyTorch made the image ~11 GB | `torch` pinned to the `pytorch-cpu` uv index on Linux | Image 11.3 GB → 3.28 GB, `torch 2.14.1+cpu` |
| 8 | Demo passwords hardcoded in a public repo, reset on every seed; ports open on all interfaces | `SEED_ADMIN_PASSWORD` / `SEED_ANALYST_PASSWORD` from `.env`, `ON CONFLICT DO NOTHING`; ports bound to `127.0.0.1` | `docker ps` shows `127.0.0.1:` ports |
| 9 | Unused `openai`, `vanna`, `psycopg2-binary`, duplicate `sentence-transformers` | Removed from `pyproject.toml`, relocked | Not importable in the container |
| 10 | `users` table defined twice with different types | Only `001_create_users.sql` (BIGSERIAL/TEXT, matching the live DB); startup runs that file | — |

Follow-ups done: `SEED_ADMIN_PASSWORD` / `SEED_ANALYST_PASSWORD` are set in `.env` (random, never printed), `analyst` was created and `admin` was moved to the new password. Still open: the unused `vanna_*` settings remain in `config.py` / `.env`.

## Known gaps / next steps

- RAG / Text2SQL pipeline, LangGraph orchestration, llm-guard checks: not started.
- No cache layer yet, so no cache stats/clear admin endpoints. No Tavily health ping yet (Phase 3).
- Rate limiting only on `/auth/login` and `/auth/register`. The general per-user limit (`RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`) and `MAX_TOKENS_PER_USER_DAILY` aren't wired to any route. Behind Docker's NAT every host request shares one IP (`172.18.0.1`).
- `X-RateLimit-*` headers are added on success and on 429, but not on other error responses (e.g. a wrong-password 401).
- The Dockerfile `HEALTHCHECK` hardcodes port 8000; setting `API_PORT` would make the container report unhealthy.
- Admin status is read from the JWT, so promotions/demotions apply only after re-login.
- **LLM provider: Cerebras since 2026-10-22.** Groq's free tier allowed only 200k tokens/day per model (one E1 run needs ~265k+), its Developer tier was closed ("temporarily unavailable"), and its `qwen/qwen3.8-27b` had a hidden 1,000 output-tokens/minute limit that rejected judge requests. Cerebras free trial limits seen in response headers: `gpt-oss-120b` **5 requests/min, 150/hour, 2,400/day, 30k tokens/min, 1M tokens/day**; `qwen-3.8-27b` 450 requests/min, 150k tokens/min, 216M tokens/day. Cerebras has no gpt-oss-20b. The answer model's 5 req/min (150/hour) is the bottleneck: ~35 answer calls per E1 run. The free trial needs a verified payment method; its $5 credit expires 30 days after it's granted. The Docker api image must be rebuilt (`groq` → `openai` dependency) before the container is used again.
- Not written yet (later phases): `scripts/data_pipeline/` (for `make seed-data`, also the generator of `002_k8s_ops.sql`). `seed_docs()` is a stub until Phase 1.
- `openai`, `vanna` and `psycopg2-binary` were removed from `pyproject.toml` (unused; Text2SQL is planned as our own `sql_service`). The `vanna_*` settings remain in `config.py` / `.env` but nothing reads them.
- `eval/seed_questions.yaml` is the original (unfixed) golden file, superseded by `eval/golden_questions.yaml`; kept for reference, not used by any code.
- No goldens are tagged `sparse` or `dense` (the originals were re-tagged `hybrid`), so `eval.schema` warns about those two features. E2's `sparse_only` profile still runs over all goldens.

## Stack

- Python 3.12, managed with **uv** (`uv run ...`, never bare `pip`)
- Web: FastAPI + uvicorn
- LLM: Cerebras (`openai` SDK against `https://api.cerebras.ai/v1`): `gpt-oss-120b` answers/grades, `qwen-3.8-27b` judges evals; tiktoken for token budgeting. (Groq until 2026-10-22, see Known gaps.)
- Vector DB: Qdrant; embeddings via sentence-transformers; reranking via Voyage or local model
- Doc parsing: docling, pypdf
- Text2SQL (Phase 4): our own `sql_service` (sqlglot SELECT-only check + read-only role `copilot_ro`) over Postgres via psycopg3
- Cache: Upstash Redis (TTLs per layer: embeddings, RAG, SQL gen, SQL result, intent); optional S3 storage
- Auth: JWT (pyjwt) + bcrypt (passlib), with rate limiting and per-user daily token caps
- Orchestration: LangGraph with Postgres checkpointing
- Security: llm-guard (prompt injection, toxicity on input/output)
- Web search fallback: Tavily (used by CRAG)
- Logging: loguru
- Eval: Ragas 0.4.3 (dev extra) judged by Cerebras qwen via langchain-openai

## Planned retrieval pipeline (from `.env` / Makefile)

Feature-flagged stages, evaluated as RAGAS profiles: `naive` → `hybrid` (dense + sparse, RRF fusion) → `hybrid+rerank` → `+hyde` (hypothetical doc embeddings) → `+crag` (corrective RAG with relevance thresholds + web fallback) → self-reflective answering with retry. Defaults for each are toggled via `*_ENABLED_BY_DEFAULT` env vars.

## Make targets

`make` is **not installed** on this Windows machine (user chose not to install it): run the `uv run ...` line under each target in the `Makefile` directly.

```
make install / sync      # uv sync --extra dev
make up / down           # docker compose up -d --build / down
make seed                # uv run python scripts/seed_db.py
make api                 # uvicorn on :8000 (reload, selector loop)
make test                # uv run pytest tests/ -v
make lint / format       # ruff check . / ruff format .
make validate            # uv run python -m eval.schema
make eval-baseline|sparse|hybrid|rerank|hyde|crag|all   # one eval profile
make eval-diff A=<a.json> B=<b.json>                    # uv run python -m eval.diff A B
make eval-record FILE=<json> PHASE=<n> NOTE="..."       # uv run python -m eval.record ...
make streamlit                                          # uv run streamlit run scripts/streamlit_app.py (UI on :8501)
make seed-docs / seed-data                              # docs only (seed_docs()) / data pipeline (not built)
```

## Conventions

- Ruff: line length 100, target py312, rules `E,F,W,I,B,UP` (E501, B008 ignored — B008 allows FastAPI `Depends(...)` defaults).
- mypy with `ignore_missing_imports`. Pylance/pyright is what the editor shows; check the project with `uv run --with pyright pyright app scripts eval tests` (should be 0 errors).
- To list served routes, use `app.openapi()["paths"]` — this FastAPI version keeps included routers as `_IncludedRouter` entries, so iterating `app.routes` misses them.
- pytest with `asyncio_mode = "auto"`; tests live in `tests/` (`uv run pytest tests -q`). `tests/conftest.py` sets test env vars before importing `app` (env vars beat `.env`), so tests never use real secrets; empty `UPSTASH_*` forces the in-memory rate limiter. Mock Cerebras/Tavily/Voyage in tests. Every new file gets a `tests/test_<file>.py`.
- All config comes from env vars via `app/config.py`; don't hard-code keys or model names.
- Shared clients are created once in `main.py`'s `lifespan` (registered on the `AsyncExitStack`) and read in routes via small `get_*(request)` dependencies — don't create new clients per request.
- Async everywhere for I/O (psycopg3 async pool, `AsyncQdrantClient`, Upstash async, `AsyncOpenAI`).
- Auth helpers live only in `app/middleware/auth.py`; routers import from there.
- SQL always uses `%s` placeholders.
- Error responses and logs must not leak secrets (report exception types, not messages, for DB errors).
- Never print or commit values from `.env`. When adding a new setting, add it to `app/config.py`, `.env`, and `.env.example` (placeholder if secret).

## Evaluation

Every eval run is recorded here (protocol: `plan.md` §2). faithfulness and answer_relevancy are judged by Cerebras `qwen-3.8-27b` (`settings.eval_judge_model`, a different model family from the `gpt-oss-120b` answer model); context_precision / context_recall are computed from source file names (`post_checks.id_context_scores`, 0 tokens); answer_relevancy uses the local bge embeddings (`BAAI/bge-base-en-v1.5`). Log dates are **UTC** (from the result file's timestamp).

Harness notes (`eval/ragas_adapter.py`, verified 2026-10-22 with real Groq calls):
- The reference answer (`ground_truth`) is the **bare keyword list** (user's choice: goldens have no reference answers). Phrasings like "A correct answer covers: ..." made context_recall always 0; the bare list scores 1.0 on a matching context and 0.0 on an unrelated one. Recall/precision are still rough with keyword references.
- `LangchainLLMWrapper(ChatOpenAI, bypass_n=True)`: separate calls instead of `n > 1` (answer_relevancy needs 3), which not every provider supports.
- `BgeEmbeddings` wraps sentence-transformers directly: ragas 0.4.3's `HuggingfaceEmbeddings` can't be instantiated. Its loaded model attribute is `encoder`, because Ragas expects `.model` to be a string.
- `langchain-community<0.4` is pinned because ragas 0.4.3 imports a module removed in 0.4.
- Failed judgements are stored as `null`, never invented; aggregates ignore them.
- **Checkpoint/resume** (added after E1 attempt 1 lost everything): `run_ragas` saves to `eval/results/checkpoints/<profile>.json` (gitignored) after every answered and every scored question, scoring one question at a time. Re-running the same command resumes: answered questions aren't asked again, and only rows that are unscored, scored by another judge, or have a `null` metric are re-judged. If every metric of a question comes back empty (Groq daily limit), it stops with exit code 2 and keeps the checkpoint. `--fresh` ignores the checkpoint. If any question errored (e.g. the answer model hit its daily limit), no result file is written (exit code 3) and the checkpoint is kept, so a re-run asks only those; `--allow-errors` writes a result anyway. The checkpoint is deleted when the result file is written. Note: the **answers** also use 120b's 200k/day, ~2.3k tokens each.
- **answer_relevancy strictness** = `EVAL_RELEVANCY_STRICTNESS` (`settings.eval_relevancy_strictness`; Ragas default 3 = the judge writes 3 guessed questions per answer). E1 and E2 ran with **3**; from **E3 on it's 1** (user's choice 2026-10-23, to save ~15-20% of judge tokens). Set `.env` to 1 only after E2's three runs finish. Each result file records it as `answer_relevancy_strictness`. Measured judge cost with thinking off and strictness 3: ~7.5k tokens/question (5.3k in, 2.2k out) ≈ $0.30 per 35-question run.
- A gpt-oss judge gets `reasoning_effort=settings.llm_reasoning_effort` (`low`; Groq's default cost ~3.3k tokens per judge job). qwen gets `reasoning_effort="none"`: on Cerebras it thinks silently by default (113 vs 8 output tokens on a tiny judge call), and every token is deducted from the free trial's **$5 credit** (qwen ≈ $0.99/M input, $1.49/M output per one aggregator). E1 was judged with thinking on (see the Baseline caveat).
- **Metric change (2026-10-22, user-approved "option 1")**: context_precision / context_recall are ID-based, from the source file of each retrieved chunk vs `golden_sources` (`context_sources` is stored per row, in rank order). precision is rank-aware like Ragas' (mean precision@k at relevant ranks), recall = fraction of golden files retrieved; both `null` for goldens without sources. The LLM versions cost ~11k of ~17k judge tokens per question.

**After every run:**
1. Run `uv run python -m eval.run_ragas --profile <name>`. It writes `eval/results/<timestamp>_<profile>.json`.
2. Run `uv run python -m eval.record eval/results/<file>.json --phase <N> --note "<one line>"`. This appends a row between the EVAL-LOG markers. Don't edit or delete old rows, and never invent numbers. If a run fails, record the failure.
3. Write 3–5 plain-English bullets under that checkpoint in **Eval Observations**:
   - what changed compared with the previous checkpoint and with the E1 baseline;
   - which question IDs failed, and why;
   - one action item for the next phase.
4. **Regression rule:** if any core metric (faithfulness, context_precision, context_recall, answer_relevancy) drops by more than **0.05** versus the previous best profile, stop, tell the user, and investigate.

### Checkpoints

| Checkpoint | Phase | Feature under test | Profile(s) |
|---|---|---|---|
| E0 | 0 | Harness smoke test (no RAG yet) | `naive` (all rows skipped) |
| **E1** | 1 | **Basic RAG (dense) — BASELINE** | `naive` |
| E2 | 2 | Sparse, hybrid RAG (dense + BM25 + RRF), and reranking | `sparse_only`, `hybrid`, `hybrid+rerank` |
| E3 | 3 | HyDE, CRAG, Self-RAG | `hybrid+rerank+hyde`, `hybrid+rerank+crag`, `all` |
| E4 | 4 | Text2SQL + intent routing | `all` (with sql + hybrid goldens) |
| E5 | 5 | Caching | `all` cold, then `all` warm |
| E6 | 6 | Security layers | `all --filter security`, then `all` |

### Baseline (E1, `naive`)

**faithfulness 0.510 / context_precision 0.511 / context_recall 0.625 / answer_relevancy 0.592 / source_overlap 0.625** — 35 evaluated, 9 skipped (sql/hybrid intents, Phase 4), 0 errors, 1 forbidden hit. `eval/results/20261022T173834Z_naive.json` (2026-10-22, commit `ae5071f` + uncommitted Phase 0/1 work). Setup: answers `gpt-oss-120b`, judge `qwen-3.8-27b` (both Cerebras), ctx metrics ID-based, corpus 47 true docs + 148 noise PDFs (7,285 chunks). Every later checkpoint is compared to this line, with the same judge and metric definitions. **Caveat:** E1 was judged with qwen's hidden thinking **on**; from 2026-10-22 the judge runs with `reasoning_effort="none"` (cost: E1's judging used ~600k qwen tokens, ~17k/question, billed against the $5 trial credit). The user chose not to re-score E1. So faithfulness / answer_relevancy deltas vs E1 include a small judge-setting effect; context_precision / context_recall / source_overlap are unaffected (no LLM). Before applying the 0.05 regression rule to faithfulness or answer_relevancy, keep this in mind.

### Eval Results Log

<!-- EVAL-LOG:START -->
| date | phase | profile | n_eval/n_skipped | faith | ctx_prec | ctx_rec | ans_rel | src_overlap | forbidden | latency_ms | result file | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-21 22:11 | 0 | naive | 0/44 | - | - | - | - | - | 0 | - | `eval/results/20261021T221133Z_naive.json` | harness smoke test - all skipped |
| 2026-10-22 17:38 | 1 | naive | 35/9 | 0.510 | 0.511 | 0.625 | 0.592 | 0.625 | 1 | 10616.949 | `eval/results/20261022T173834Z_naive.json` | BASELINE naive dense RAG (Cerebras: 120b answers, qwen judge, ID-based ctx metrics) |
| 2026-10-22 18:50 | 2 | sparse_only | 35/9 | 0.536 | 0.515 | 0.656 | 0.621 | 0.656 | 1 | 12424.349 | `eval/results/20261022T185003Z_sparse_only.json` | sparse_only: BM25 only |
| 2026-10-22 19:06 | 2 | hybrid | 35/9 | 0.497 | 0.590 | 0.641 | 0.622 | 0.641 | 1 | 10641.429 | `eval/results/20261022T190611Z_hybrid.json` | hybrid: dense+BM25, RRF k=60 |
| 2026-10-22 19:21 | 2 | hybrid+rerank | 35/9 | 0.480 | 0.528 | 0.609 | 0.540 | 0.609 | 1 | 10844.066 | `eval/results/20261022T192106Z_hybrid+rerank.json` | hybrid+rerank: local ms-marco-MiniLM-L-6, 256 tok, 20->5 |
<!-- EVAL-LOG:END -->

### Eval Observations

#### E0 — harness smoke test
- Harness works end to end: 44 goldens loaded and validated, 0 evaluated, 44 skipped ("RAG not implemented"), 0 errors; result JSON written and recorded with `eval.record`.
- No scores exist yet by design (Phase 0 has no RAG), so there is nothing to compare; E1 sets the baseline.
- Separately verified the Ragas judge on hand-made rows: matching context gave faithfulness/precision/recall/relevancy 1.0; an unrelated context gave 0.0/0.0/0.0 with relevancy 0.55.
- Action for Phase 1 (done): `ServiceInvoker.invoke` calls `rag_service.run_rag_with_trace` and skips `sql`/`hybrid` intents until Phase 4; E1 will evaluate 35 rows (44 minus 5 sql and 4 hybrid). `web_fallback` (CRAG) goldens do run and are expected to fail until Phase 3.

#### E1 — basic RAG (baseline)
- Attempt 1 (2026-10-22 17:32 local, 120b judge): **failed, no result file.** Answering all 35 questions worked, but Ragas judging hit Groq's **daily token limit for `openai/gpt-oss-120b` (TPD 200,000)** at job 36/140 (~26%), slowing to ~100 s per job, and Claude Code then killed the process for low system memory. `run_ragas` only writes its JSON at the end, so the generated answers were lost too.

- Attempt 2 (Groq, 20b judge): only 10 of 35 answered before 120b's daily limit; partial result deleted, not recorded. Then the provider moved to Cerebras.
- **Attempt 3 = the baseline** (2026-10-22 22:51–23:10 local, ~19 min): faith 0.510 / ctx_prec 0.511 / ctx_rec 0.625 / ans_rel 0.592 / overlap 0.625. 3 faithfulness judgements came back `null` (q-005, q-022, q-033). Latency mean 10.6 s, but that's inflated by ~57 s waits on the 5 requests/min limit every 5th question; normal answers take 0.5–1 s.
- By feature (faith / ctx_prec / ctx_rec / ans_rel): baseline 0.73/0.58/0.75/0.92, hybrid 0.57/0.84/0.90/0.94, rerank 0.82/0.77/1.00/0.66, wild 0.59/0.83/1.00/0.70 — dense retrieval already finds the right file for most direct questions. **Failing groups: hyde 0.31/0.14/0.20/0.61** (paraphrased questions miss: q-016 "misbehaving tenant ... CPU on a shared computer" retrieved Intel/GPU noise PDFs instead of resource-quotas/limit-range → "I don't know"; q-014 "compute resources reserved" got resource-quotas instead of pod-priority/eviction), **crag 0.25/0/0/0** (expected: off-corpus questions like q-017 Seattle weather and q-019 K8s 1.34 release need the web fallback, so the model correctly says "I don't know"), **self_rag 0.77/0.35/0.38/0.61** (vague queries q-023 "explain" / q-024 "what is wrong" retrieve pure noise — "Writing R Extensions", thread papers — and the model answers from it), **security 0.09/0.48/1.00/0.20** (q-042 echoed the email and card number → the 1 forbidden hit; q-041/043/044 correctly decline, which Ragas scores 0).
- Other examples: q-003 (container crash-looping) retrieved pod-v1 API reference 4× instead of logging.pdf / kubectl cheatsheet — a near-duplicate-chunk problem; q-009 and q-036 retrieved the golden file but the model still said "I don't know" (the relevant chunk wasn't specific enough at rank 2+).
- Action for Phase 2: hybrid (BM25) should fix exact-term misses like q-003/q-036 and reranking should push the right chunk to the top (watch ctx_prec on hyde/self_rag goldens). Re-run `naive` after Phase 2 to confirm it still matches this line within ±0.02. Phase 6 must stop PII echo (q-042).

#### E2 — hybrid RAG + reranking
- Runs (2026-10-23 00:04–00:52 local, same 35 questions as E1, judge qwen thinking off, strictness 3): sparse_only faith 0.536 / prec 0.515 / rec 0.656 / rel 0.621; **hybrid 0.497 / 0.590 / 0.641 / 0.622** (best precision); hybrid+rerank 0.480 / 0.528 / 0.609 / 0.540. vs E1 (0.510/0.511/0.625/0.592): hybrid lifts context_precision **+0.079** and recall +0.016; faith/rel deltas vs E1 carry the judge-setting caveat.
- **Regression rule triggered by hybrid+rerank** (vs the best E2 profile): answer_relevancy −0.082 vs hybrid, context_precision −0.062 vs hybrid, faithfulness −0.056 vs sparse_only. Flipped pass→fail: q-010, q-011; fail→pass: q-005, q-013.
- Investigation: relevancy fell mostly from 3 new "I don't know" answers (q-007, q-038, q-040 → relevancy 0): on q-007 the reranker kept the right file (pod-v1) but chose chunks without the nodeSelector example. Precision fell because it moved golden chunks down (q-003 logging.pdf, q-011 secret.txt, q-033). A free retrieval-only replay over the 35 questions (ID-based metrics, 0 LLM tokens) reproduced the eval exactly (hybrid 0.590/0.641, rerank@256 0.528/0.609) and showed **rerank@512 is worse still (0.501)** — so the cause is the model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on web search), not the 256-token cut. The replay script is the cheap way to test any reranker before spending eval credit.
- sparse_only beat naive on recall (0.656) and BM25 found exact-term chunks dense missed (`kubectl rollout undo` → kubectl cheatsheet #1; q-003 logging.pdf into the top 5 under hybrid). "CrashLoopBackOff" isn't in the corpus at all. hyde/crag goldens still fail (Phase 3).
- Reranker search (user's choice, free replay, 0 LLM tokens, all @256 tokens; bar = hybrid 0.590 / 0.641): ms-marco-MiniLM-L-6 0.528 / 0.609 (1.1 s/q), ms-marco-MiniLM-L-12 0.501 / 0.578 (2.0 s/q), mxbai-rerank-xsmall-v1 0.521 / 0.594 (4.2 s/q), bge-reranker-base 0.558 / 0.594 (7.6 s/q). **No local reranker beats plain hybrid on this corpus**, so `hybrid+rerank` was not re-run. The downloaded models are in `D:\hf-cache`.
- **Phase 2 result: hybrid (dense + BM25, RRF k=60) without rerank is the best profile.** Reranking stays available behind `enable_rerank` (default off; `RERANKER_MODEL` stays ms-marco-MiniLM-L-6). Action for Phase 3: build on `hybrid`; HyDE targets the hyde goldens (q-014/015/016/038 at ctx_prec 0 under every E2 profile), CRAG the 4 crag goldens.

#### E3 — HyDE / CRAG / Self-RAG
_Not run yet._

#### E4 — Text2SQL + routing
_Not run yet._

#### E5 — caching (cold vs warm)
_Not run yet._

#### E6 — security
_Not run yet._

### Final Comparison

_Written at Phase 7 from the full log: baseline E1 → best profile, which phase moved each metric and why, and the recommended production default flags._
