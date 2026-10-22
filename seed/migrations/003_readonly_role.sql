-- Read-only role for Text2SQL (Phase 4): SELECT on the K8s ops tables, nothing else.
-- It deliberately has NO access to `users` (password hashes).
-- NOLOGIN for now; Phase 4 gives it LOGIN + a password from .env when sql_service needs it.
-- Runs after 002 on every seed: 002 drops/recreates the ops tables, which drops their
-- grants, so they are re-granted here each time.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'copilot_ro') THEN
        CREATE ROLE copilot_ro NOLOGIN;
    END IF;
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO copilot_ro', current_database());
END
$$;

GRANT USAGE ON SCHEMA public TO copilot_ro;
REVOKE ALL ON users FROM copilot_ro;
GRANT SELECT ON clusters, nodes, deployments, pods, incidents, alerts, oncall_logs TO copilot_ro;
