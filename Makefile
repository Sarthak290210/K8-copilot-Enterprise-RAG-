.PHONY: help install sync up down seed seed-docs seed-data api streamlit test lint format validate \
	eval eval-baseline eval-sparse eval-hybrid eval-rerank eval-hyde eval-crag eval-all eval-defaults \
	eval-diff eval-record

# On Windows without `make`, run the `uv run ...` line under a target directly.

help:
	@echo "k8-copilot — available commands"
	@echo ""
	@echo "  make install       create venv & install all deps (one-time)"
	@echo "  make sync          sync deps with pyproject.toml"
	@echo "  make up / down     start / stop postgres + qdrant + api (docker compose)"
	@echo "  make seed          migrations + seed users (+ doc ingestion from Phase 1)"
	@echo "  make api           FastAPI backend on :8000 (reload)"
	@echo "  make streamlit     Streamlit UI on :8501 (Phase 1)"
	@echo "  make test          pytest"
	@echo "  make lint/format   ruff check / ruff format"
	@echo "  make validate      validate eval/golden_questions.yaml"
	@echo "  make eval-<name>   run one eval profile (baseline sparse hybrid rerank hyde crag all)"
	@echo "  make eval-diff A=<a.json> B=<b.json>"
	@echo "  make eval-record FILE=<result.json> PHASE=<n> NOTE=\"<text>\""


install:
	uv python pin 3.12
	uv venv --python 3.12
	uv sync --extra dev

sync:
	uv sync --extra dev

up:
	docker compose up -d --build

down:
	docker compose down

seed:
	uv run python scripts/seed_db.py

seed-docs:
	uv run python -c "from scripts.seed_db import seed_docs; seed_docs()"

seed-data:
	bash scripts/data_pipeline/run_all.sh

api:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --loop asyncio:SelectorEventLoop

streamlit:
	uv run streamlit run scripts/streamlit_app.py


test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

validate:
	uv run python -m eval.schema


eval-baseline:
	uv run python -m eval.run_ragas --profile naive

eval-sparse:
	uv run python -m eval.run_ragas --profile sparse_only

eval-hybrid:
	uv run python -m eval.run_ragas --profile hybrid

eval-rerank:
	uv run python -m eval.run_ragas --profile hybrid+rerank

eval-hyde:
	uv run python -m eval.run_ragas --profile hybrid+hyde --filter hyde

eval-crag:
	uv run python -m eval.run_ragas --profile hybrid+crag --filter crag

eval-all:
	uv run python -m eval.run_ragas --profile all

eval-defaults:  ## Phase 4+: the shipped defaults (hybrid + CRAG) through the agent
	uv run python -m eval.run_ragas --profile defaults

# Baseline + all, then compare the newest of each.
eval: eval-baseline eval-all
	@latest_naive=$$(ls -t eval/results/*_naive.json 2>/dev/null | head -1); \
	latest_all=$$(ls -t eval/results/*_all.json 2>/dev/null | head -1); \
	uv run python -m eval.diff $$latest_naive $$latest_all

eval-diff:
	@test -n "$(A)" && test -n "$(B)" || (echo "Usage: make eval-diff A=<a.json> B=<b.json>" && exit 1)
	uv run python -m eval.diff $(A) $(B)

eval-record:
	@test -n "$(FILE)" && test -n "$(PHASE)" || (echo "Usage: make eval-record FILE=<json> PHASE=<n> NOTE=\"...\"" && exit 1)
	uv run python -m eval.record $(FILE) --phase $(PHASE) --note "$(NOTE)"
