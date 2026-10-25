-- Phase 4: who owns each agent run (LangGraph thread). POST /query/sql/execute checks it, so
-- only the user who asked can approve (or reject) the SQL waiting in a paused run.
-- Also run at API startup (app/api/query.py), like 001.

CREATE TABLE IF NOT EXISTS query_threads (
    thread_id   UUID        PRIMARY KEY,
    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_threads_user ON query_threads(user_id);
