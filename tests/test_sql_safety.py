"""sql_service.is_select_only: the first of Text2SQL's three safety layers."""

import pytest

from app.services.sql_service import is_select_only

SAFE = [
    "SELECT count(*) FROM incidents WHERE severity = 'P1'",
    "select avg(mttr_minutes) from incidents limit 100",
    "WITH c AS (SELECT node_id, count(*) n FROM pods GROUP BY 1) SELECT * FROM c LIMIT 5",
    "SELECT c.name, count(*) FROM pods p JOIN nodes n ON p.node_id = n.node_id "
    "JOIN clusters c ON n.cluster_id = c.cluster_id WHERE p.status = 'CrashLoopBackOff' "
    "GROUP BY c.name ORDER BY 2 DESC LIMIT 1",
    "SELECT name FROM clusters UNION SELECT name FROM deployments",
    "SELECT * FROM public.incidents LIMIT 3",
]

MALICIOUS = [
    "DROP TABLE incidents",
    "SELECT 1; DELETE FROM incidents",
    "SELECT * FROM incidents; DROP TABLE pods",
    "UPDATE incidents SET severity = 'P4'",
    "INSERT INTO incidents (severity) VALUES ('P1')",
    "WITH d AS (DELETE FROM incidents RETURNING *) SELECT * FROM d",
    "SELECT pg_sleep(10)",
    "SELECT * FROM incidents WHERE pg_sleep(5) IS NULL",
    "SELECT current_setting('is_superuser')",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT * FROM users",
    "SELECT password_hash FROM users WHERE username = 'admin'",
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_stat_activity",
    "SELECT * INTO stolen FROM incidents",
    "SELECT * FROM incidents FOR UPDATE",
    "COPY incidents TO '/tmp/out.csv'",
    "GRANT ALL ON incidents TO public",
    "CREATE TABLE x AS SELECT * FROM incidents",
    "SET statement_timeout = 0",
    "",
    "not sql at all ((",
]


@pytest.mark.parametrize("query", SAFE)
def test_safe_selects_are_allowed(query):
    assert is_select_only(query)


@pytest.mark.parametrize("query", MALICIOUS)
def test_malicious_sql_is_rejected(query):
    assert not is_select_only(query)
