<div align="center">

# ☸️ k8-copilot

**An enterprise RAG assistant for Kubernetes operations**

It answers from Kubernetes documentation *and* from a live ops database, picks the right source for each question, asks a human before any SQL runs, and checks every answer against its sources.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-DC244C)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Upstash-Redis-DC382D?logo=redis&logoColor=white)
![Cerebras](https://img.shields.io/badge/LLM-Cerebras%20gpt--oss--120b-F15A29)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-456%20passing-brightgreen)
![Ragas](https://img.shields.io/badge/eval-Ragas-8A2BE2)

[Tech stack](#-tech-stack) · [Features](#-what-it-does) · [Architecture](#-architecture) · [Quick start](#-quick-start) · [Results](#-evaluation-results) · [Optimizations](#-latency-and-cost-optimizations)

</div>

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| **API** | FastAPI + uvicorn · Python 3.12 · [uv](https://docs.astral.sh/uv/) |
| **LLM** | Cerebras (OpenAI-compatible): `gpt-oss-120b` answers and grades · `qwen-3.8-27b` judges the evals |
| **Retrieval** | Qdrant (768-dim cosine) · `BAAI/bge-base-en-v1.5` embeddings · BM25 (`rank-bm25`) · reciprocal rank fusion · Docling (PDF / DOCX / HTML) |
| **Advanced RAG** | HyDE · Corrective RAG (CRAG) with Tavily web search · Self-RAG · cross-encoder reranking (optional) |
| **Agent** | LangGraph with a Postgres checkpointer (a paused SQL run survives restarts) |
| **Text2SQL** | Our own service: sqlglot `SELECT`-only check · read-only Postgres role · human approval |
| **Data** | PostgreSQL 17 (users, agent threads, generated K8s ops data: clusters, nodes, pods, incidents, alerts) |
| **Cache / limits** | Upstash Redis + in-memory copy (5 cache tiers, rate limits, token budgets) |
| **Security** | llm-guard + ONNX Runtime (prompt injection, toxicity) · regex PII redaction · spotlighting · grounding check |
| **Auth** | JWT (pyjwt) · bcrypt (passlib) |
| **Eval** | Ragas 0.4 (faithfulness, answer relevancy) + ID-based context precision / recall |
| **UI / Ops** | Streamlit · Docker Compose · ruff · pyright · pytest |

## ✨ What it does

| | Feature | How |
|---|---|---|
| 🔎 | **Hybrid retrieval** | Dense embeddings + BM25 merged with RRF, at most 2 chunks per document, over a collection where ~91% of the chunks are deliberately unrelated "noise" |
| 🧭 | **Corrective RAG** | A grader scores the chunks; weak ones trigger a web search (Tavily), but only for Kubernetes / IT-ops questions. Off-topic requests are declined, vague ones get a request for detail |
| 💭 | **HyDE** (per request) | The LLM writes 3 hypothetical doc passages; dense search uses them and BM25 searches the question + passages; the answer sees the top 8 chunks |
| 🗄️ | **Text2SQL with approval** | A LangGraph agent routes each question to `rag`, `sql` or `hybrid`. Generated SQL is checked by sqlglot, runs as a read-only role, and **pauses for the user's approval** |
| ⚡ | **Caching** | 5 tiers (intent, embeddings, RAG answers, SQL, SQL results). A repeated question returns in ~0.1 s for ~0 tokens. Uploads are de-duplicated by content hash |
| 🛡️ | **10 security layers** | Validation, JWT, rate limits, daily token budget, injection + toxicity guard, PII redaction, spotlighting, hardened prompt, output validation, moderation |
| ✅ | **Grounding check** | Every answer is split into claims and each is checked against the sources (summaries and words like "guarantees" included). Below 0.9 it is rewritten once; below 0.5 it is replaced |
| 📊 | **Eval dashboard** | Streamlit compares two eval runs question by question, explains each failure, and lets you re-run it with suggested settings |

## 🏗️ Architecture

```mermaid
flowchart LR
    U[User / Streamlit UI] -->|JWT| API[FastAPI]
    API --> SEC_IN[Input layers<br/>validation, rate limit, token budget,<br/>injection + toxicity guard, PII redaction]
    SEC_IN --> AGENT{LangGraph agent<br/>route intent}
    AGENT -->|rag| RAG[Hybrid search<br/>dense + BM25 + RRF<br/>optional HyDE]
    RAG --> CRAG[CRAG grader<br/>+ web search if needed]
    CRAG --> ANS[Answer LLM<br/>spotlighted context]
    AGENT -->|sql| SQLGEN[Text2SQL<br/>sqlglot check]
    SQLGEN --> APPROVE[/Human approval/]
    APPROVE --> DB[(Postgres<br/>read-only role)]
    DB --> ANS
    AGENT -->|hybrid| RAG
    AGENT -->|hybrid| SQLGEN
    ANS --> GROUND[Grounding check<br/>rewrite below 0.9]
    GROUND --> SEC_OUT[Output layers<br/>moderation, PII redaction]
    SEC_OUT --> API

    RAG --- Q[(Qdrant)]
    CRAG --- T[Tavily]
    ANS --- LLM[Cerebras gpt-oss-120b]
    API --- R[(Upstash Redis<br/>cache, limits, budget)]
```

<details>
<summary><b>Request order for <code>POST /query</code></b></summary>

```
L1 validation -> JWT -> rate limit -> L6 token budget check
-> L5 token limit -> L2 injection / toxicity guard -> PII redaction (question)
-> agent: route -> retrieve -> L8 spotlighting + L3 hardened prompt -> LLM -> L9 output clean-up
-> L10 grounding check (one rewrite below 0.9) -> L7 moderation + PII redaction (answer)
-> L6 consume the tokens actually used -> response
```

</details>

## 🚀 Quick start

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, and a Cerebras account. Upstash Redis and Tavily are optional (limits and cache fall back to memory; no web search without Tavily).

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Configure: copy the template and fill in the secrets
cp .env.example .env        # CEREBRAS_API_KEY, DATABASE_URL, JWT_SECRET (>= 32 chars), ...

# 3. Start Postgres and Qdrant
docker compose up -d postgres qdrant

# 4. Create tables, seed users and ops data, ingest the documents
uv run python scripts/seed_db.py

# 5. Run the API (http://localhost:8000, Swagger at /docs) and the UI (http://localhost:8501)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop asyncio:SelectorEventLoop
uv run streamlit run scripts/streamlit_app.py
```

> [!NOTE]
> The document corpus is **not** in this repository. Put your Kubernetes docs (pdf, docx, html, txt) in `seed/docs/true_data/` and, optionally, unrelated "noise" PDFs in `seed/docs/noisy_data/`. `seed/docs/attack_data/` holds documents with planted instructions for the indirect prompt-injection test. Seeding records a content hash per file, so a re-run skips files already indexed.

> [!TIP]
> On Windows keep `--loop asyncio:SelectorEventLoop` (async psycopg can't use the Proactor loop). `docker compose up -d --build` runs everything, the API included, in containers.

## 📊 Evaluation results

Every phase ended with an evaluation checkpoint (E1–E6) on golden questions, and each run was compared with the previous best on the **same question IDs**.

### The journey: same 33 questions at every checkpoint

| Checkpoint | Faithfulness | Context precision | Context recall | Answer relevancy | Forbidden outputs |
|---|---|---|---|---|---|
| Phase 1 baseline (dense RAG) | 0.544 | 0.545 | 0.667 | 0.628 | 1 |
| Phase 2 hybrid search | 0.527 | 0.630 | 0.683 | 0.660 | 1 |
| Phase 3 + HyDE, CRAG, Self-RAG | 0.647 | 0.718 | 0.683 | 0.777 | 2 |
| Phase 4 agent (defaults) | 0.600 | 0.729 | 0.741 | 0.738 | 0 |
| Phase 5 caching (cold) | 0.676 | 0.724 | 0.750 | 0.757 | 0 |
| Phase 6 security (E6) | 0.642 | 0.709 | 0.750 | 0.744 | 0 |
| **Phase 7 fixes (current)** ¹ | **0.740** | **0.824** | **0.869** | **0.739** | **0** |
| **Baseline → current** | **+0.196** | **+0.279** | **+0.202** | **+0.111** | 1 → 0 |

### Phase 6 → now: all 45 questions

| Metric | E6 (recorded) | Current ¹ | Change |
|---|---|---|---|
| Questions passing ² | 26 / 45 (31 under the new refusal rule) | **41 / 45** | **+15** |
| Faithfulness | 0.686 | **0.758** | +0.072 |
| Context precision | 0.785 | **0.870** | +0.085 |
| Context recall | 0.807 | **0.895** | +0.088 |
| Answer relevancy | 0.771 | 0.767 | −0.004 |
| Real (non-security) questions: faithfulness | 0.776 | **0.846** | +0.070 |
| Intent accuracy · SQL keyword match | 0.977 · 1.000 | 0.977 · 1.000 | — |
| Forbidden outputs · false blocks | 0 · 0 | 0 · 0 | — |
| Attacks blocked at the door | 28.6% | 28.6% | — |
| Grounding score (mean) | 0.968 | 0.950 | stricter check ³ |

<sub>¹ **Compiled, not one run:** the 26 questions that passed E6 keep their E6 answers; the 19 that failed use their latest re-run (subset runs on 2026-10-28, see `eval/results/default_v3.json`). The re-runs' judge also sees each chunk's source file. A full re-run of the 45 questions is the next comparable checkpoint.
² Pass = faithfulness ≥ 0.7, at least one expected document retrieved, no forbidden words. Questions that *should* be refused pass when they are refused.
³ The grounding check now also checks summary sentences and strong words ("guarantees", "always") and needs 0.9 instead of 0.8, so it flags more: 2 answers warned, 1 replaced (q-014).</sub>

### How the 19 Phase 6 failures were fixed

| Cause | Questions | Fix | Result |
|---|---|---|---|
| Correct refusals scored as failures | q-017, q-018, q-041, q-043, q-044 | Pass rule for should-decline questions (CRAG scope check / input guard did the refusing) | ✅ 5 pass |
| Retrieval missed paraphrased questions | q-003, q-015, q-016, q-038 | **Hybrid + HyDE** (BM25 on question + passages, top 8, max 2 per doc) + HyDE answer prompt | ✅ 4 pass (q-016 unstable) |
| Hallucination (details beyond the docs) | q-020, q-039, q-042 | "Only from the documents" prompt + **strict grounding check** (rewrite below 0.9) | ✅ 3 pass (faith 0.39–0.47 → 0.89–1.00) |
| Text lost at ingestion | q-036, q-040 | Docling's over-long "headings" kept as body text; 6 PDFs re-indexed | ✅ 2 pass (q-040 0.42 → 0.84) |
| Judge couldn't see source names | q-008 | Judge and grounding check see each chunk's source | ✅ pass |
| Unhelpful "I don't know" | q-009 | Hybrid + rerank + answer-the-covered-part prompt | ✅ pass (0 → 1.00) |
| Still open | q-014 (retrieval), q-023, q-024 (vague, need Self-RAG) | Hallucination now caught (q-014 answer replaced); retrieval fix and adaptive RAG planned | ❌ 3 fail |

<details>
<summary><b>What moved the scores across phases</b></summary>

- **Precision:** hybrid search (+0.085) and CRAG (+0.088) did most of the work; the per-document cap and the ingestion fix added more after Phase 7. Reranking *lowered* precision on this corpus (MS MARCO models) and stays off.
- **Recall:** the narrower CRAG rule in Phase 4 (web only for in-scope questions) kept good local documents; HyDE and the re-index lifted it again after Phase 7.
- **Faithfulness:** CRAG in Phase 3, then the "only from the documents" prompt and the strict grounding check after Phase 7.
- **Noise to keep in mind:** the judge alone moves faithfulness by about ±0.05 on identical answers.

</details>

<details>
<summary><b>How the evaluation works</b></summary>

- **Golden questions:** `eval/golden_questions.yaml` holds 45 questions, each tagged with the feature it tests: baseline docs, exact-term (hybrid), paraphrased (HyDE), off-corpus (CRAG), vague (Self-RAG), SQL, hybrid SQL + docs, and security (prompt injection, PII, a toxic request, off-topic requests, an injection planted in a document). Each lists its expected source files, answer keywords and *forbidden* keywords (text that must never appear, e.g. a planted fake secret).
- **Metrics:** faithfulness and answer relevancy from Ragas, judged by `qwen-3.8-27b` (a different model family from the answering `gpt-oss-120b`); context precision (rank-aware) and recall computed from the source files of the retrieved chunks (deterministic, 0 tokens); checks in code for forbidden words, intent accuracy, SQL keywords, block / false-block rates and the grounding score.
- **Profiles:** each run fixes the pipeline flags: `naive`, `hybrid`, `hybrid+rerank`, `hybrid+hyde`, `hybrid+crag`, `all`, and `defaults` (the shipped settings, through the full agent; SQL auto-approved).
- **Rules:** compare on the same question IDs; any core metric dropping by more than 0.05 is investigated before going on; read answers question by question; every run checkpoints after each question and resumes after a crash; an answer whose grader step failed (e.g. a provider outage) is retried, not scored.
- **Free screening first:** retrieval changes (rerankers, chunk caps, HyDE merges) are tested with a retrieval-only replay at 0 LLM tokens before any paid run.

```bash
uv run python -m eval.run_ragas --profile defaults                 # -> eval/results/<time>_defaults.json
uv run python -m eval.run_ragas --profile defaults --filter security
uv run python -m eval.run_ragas --profile defaults --clear-cache   # cold-cache run
uv run python -m eval.diff A.json B.json                           # compare two runs
```

The full run log and per-question notes are in [`CLAUDE.md`](CLAUDE.md#evaluation).

</details>

## ⚡ Latency and cost optimizations

Measured on this project's CPU-only laptop unless marked as an estimate.

<details open>
<summary><b>Latency</b></summary>

| Change | Before → after |
|---|---|
| 5-tier cache (memory + Upstash Redis) | Repeated question 7.1 s → **9.4 ms** in the server (~0.1 s end to end); eval median 72 s → 0.1 s |
| In-memory copy in front of Redis | Skips the ~44 ms Upstash round trip on a repeat |
| Toxicity model without llm-guard's padding to 512 tokens | 662 ms → **42 ms** per question, same verdicts; whole input guard ≈ 90 ms (ONNX) |
| Attacks stopped before the agent | ~7 s and ~4.6k tokens → **0.1–0.3 s and 0 tokens** |
| Off-topic or "nothing found" answered in code | One gpt-oss call (~1.5–2 s) skipped |
| BM25 index built once, not per query | ~4.8 s per query → milliseconds |
| Reranker left off | Avoids +1.1 s (MiniLM) to +13–27 s (gte) per query, and it lowered quality here |
| Content-hash dedup when seeding | Re-ingesting 195 files (~3 h, estimate) → recorded in 27 s, then skipped |
| Noise PDFs capped at 30 pages, no table model | 7+ min → ~1 min per noise PDF |
| CPU-only PyTorch in the Docker image | 11.3 GB → **3.28 GB** |

</details>

<details open>
<summary><b>Cost</b> (LLM tokens on Cerebras' $5 free trial)</summary>

| Change | Saving |
|---|---|
| ID-based context precision / recall instead of an LLM judge | ~11k of ~17k judge tokens per question (−65%) |
| Judge with hidden "thinking" turned off | ~17k → ~7.5k judge tokens per question (−56%) |
| Answer-relevancy strictness 3 → 1 | ~15–20% fewer judge tokens |
| `defaults` instead of `all` (no HyDE / Self-RAG) | 5.7k → **4.2k** answer tokens per question (−26%) at similar quality |
| Answer cache on a repeated question set | 192k → **39k** tokens (−80%) |
| Grounding check sees source names | q-008: needless rewrite gone, 8.7k → **4.6k** tokens |
| Only question vectors in the embedding cache | ~1.3 MB in Upstash instead of ~100 MB (estimate) |
| Free retrieval-only replays | Rerankers, chunk caps and HyDE merges tested at 0 tokens |
| Daily token budget per user | Caps one user at 100k tokens a day, counted from actual usage |

</details>

**Spent on purpose:** the grounding check adds a grader call per new answer, plus a rewrite and a second check for weak ones: **+59% tokens** (192k → 306k per full eval) and ~1.5 s per answer. In exchange, every answer gets a real confidence score and no unsupported answer is returned. With the defaults, a new question costs about 6.9k tokens (≈ $0.003) and takes ~8.5–9 s unpaced; a cached repeat costs ~0 tokens.

## 🔌 API

| Method | Path | Who | What |
|---|---|---|---|
| POST | `/auth/register`, `/auth/login` | public (rate-limited per IP) | create a user, get a JWT |
| GET | `/auth/me` | user | the current user |
| POST | `/query` | user | ask a question: an answer, or a pending SQL block to approve |
| POST | `/query/sql/execute` | user (only the asker) | approve (optionally edit) or reject that SQL |
| GET | `/query/defaults` | public | the default pipeline flags |
| POST | `/documents/upload` | admin | ingest a document (same bytes are not processed twice) |
| GET | `/documents` | user | indexed documents |
| GET | `/admin/health`, `/admin/health/{service}` | admin | Postgres, Qdrant, Redis, Cerebras, Tavily checks |
| GET / POST | `/admin/cache/stats`, `/admin/cache/clear` | admin | cache hit rates, empty a tier |
| GET | `/health` | public | liveness |

Refusals are 4xx responses with a reason code in `detail.reason`: `input_blocked` (400), `rate_limited` (429), `budget_exceeded` (429).

## 🛠️ Commands

The `Makefile` lists every command. Without `make`, run the `uv run ...` line under each target.

| Command | What it does |
|---|---|
| `make test` | pytest (456 tests; external services are mocked) |
| `make lint` / `make format` | ruff |
| `make seed` | migrations, users, ops data, document ingestion |
| `make api` / `make streamlit` | run the API / UI |
| `make eval-defaults` | Ragas eval of the default pipeline over the golden questions |
| `make eval-hyde` / `make eval-crag` / `make eval-all` | evals of the other profiles |
| `make eval-cold` / `make eval-warm` | defaults with an empty cache, then again with a warm one |
| `make eval-diff A=... B=...` | compare two eval result files |

## 🔐 Security notes

- **Secrets** live only in `.env` (gitignored); `.env.example` is the template. `JWT_SECRET` must be at least 32 characters or startup fails.
- **Network:** all published Docker ports bind to `127.0.0.1`. `X-Forwarded-For` is trusted only from localhost, so per-IP limits can't be bypassed by spoofing it.
- **SQL** never runs before the asker approves it, and it runs as a role that can only read the ops tables (not `users`).
- **Fail closed:** if the injection or toxicity models fail to load, requests are blocked (`SECURITY_FAIL_CLOSED=true`).
- **Not DDoS protection:** the rate limits stop abuse from one user or IP. Put a CDN/WAF or a reverse proxy in front for real deployments.

## 🗺️ Roadmap

- [ ] A full `defaults` evaluation of all the Phase 7 changes in one run
- [ ] Adaptive RAG: escalate per question (hybrid → CRAG → HyDE → web), so easy questions stay cheap
- [ ] Retrieval fix for q-014-style questions
- [ ] Rebuild the Docker API image and deploy
