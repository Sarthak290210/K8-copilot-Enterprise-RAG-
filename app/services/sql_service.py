"""Text2SQL: turn a question into one safe SELECT over the K8s ops tables, and run it.

Three layers of safety (fix vs reference, which only checked `startswith("SELECT")`):
1. `is_select_only` (sqlglot): exactly one SELECT statement, no DML/DDL, no dangerous functions,
   and only the ops tables (never `users`).
2. It runs as the read-only role `copilot_ro` (SELECT on the ops tables, nothing else), in a
   read-only transaction with a statement timeout.
3. At most SQL_MAX_ROWS rows are fetched.
A human approves every query before it runs (the LangGraph approval step, Phase 4).
"""

import asyncio
import datetime as dt
import uuid
from decimal import Decimal
from functools import lru_cache
from typing import Any, LiteralString, cast

import sqlglot
from loguru import logger
from psycopg import AsyncConnection, sql
from psycopg.abc import QueryNoTemplate
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.config import settings
from app.services import llm_service

OPS_TABLES = ("clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs")

# Low-cardinality columns whose values the model must spell exactly ('CrashLoopBackOff', 'P1').
ENUM_COLUMNS = {
    "clusters": ["environment", "region", "k8s_version"],
    "nodes": ["status", "node_type"],
    "deployments": ["namespace"],
    "pods": ["status"],
    "incidents": ["severity"],
    "alerts": ["severity", "alertname"],
}
DATE_COLUMNS = {
    "incidents": "started_at",
    "alerts": "fired_at",
    "oncall_logs": "paged_at",
    "pods": "created_at",
}

# Anything that writes, changes the schema, or isn't a plain query.
_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
    "TruncateTable", "Command", "Grant", "Revoke", "Copy", "Into", "Lock", "Set",
    "Transaction", "Commit", "Rollback", "Use", "Pragma",
)  # fmt: skip
_FORBIDDEN_NODES = tuple(
    cls for name in _FORBIDDEN_NODE_NAMES if isinstance(cls := getattr(exp, name, None), type)
)
# Functions that sleep, read files, signal backends or change settings.
DANGEROUS_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file",
    "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export", "lo_get", "dblink", "dblink_exec",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "set_config",
    "current_setting", "pg_advisory_lock", "query_to_xml", "pg_logdir_ls",
}  # fmt: skip
_QUERY_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)


class SQLGenerationError(ValueError):
    """The model couldn't produce a safe SELECT for the question. `tokens_used` = what the
    failed attempts still cost."""

    def __init__(self, message: str, tokens_used: int = 0) -> None:
        super().__init__(message)
        self.tokens_used = tokens_used


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def is_select_only(query: str) -> bool:
    """True only for exactly one read-only SELECT (CTEs/UNION allowed) over the ops tables."""
    try:
        statements = [s for s in sqlglot.parse(query, read="postgres") if s is not None]
    except SqlglotError:
        return False
    if len(statements) != 1 or not isinstance(statements[0], _QUERY_ROOTS):
        return False
    root = statements[0]
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return False
        if isinstance(node, exp.Func) and _function_name(node) in DANGEROUS_FUNCTIONS:
            return False
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        if table.db and table.db.lower() != "public":
            return False  # pg_catalog.*, information_schema.*, other schemas
        if name not in OPS_TABLES and name not in cte_names:
            return False
    return True


def _serialize_value(value: Any) -> Any:
    """DB value -> JSON-safe value."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in row.items()}


async def _fetch_all(
    conn: AsyncConnection, query: QueryNoTemplate, params: Any = None
) -> list[dict[str, Any]]:
    """Rows as dicts (a dict_row cursor; typed per cursor so type checkers follow it)."""
    async with conn.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(query, params)
        return await cursor.fetchall()


def readonly_conninfo() -> str:
    """DATABASE_URL with the read-only role's user and password swapped in."""
    if settings.copilot_ro_password is None:
        raise SQLGenerationError("Text2SQL is not configured (COPILOT_RO_PASSWORD is not set)")
    return make_conninfo(
        settings.database_url.get_secret_value(),
        user="copilot_ro",
        password=settings.copilot_ro_password.get_secret_value(),
        connect_timeout=5,
    )


SQL_SYSTEM_PROMPT = """You write PostgreSQL queries for a Kubernetes operations database.
Rules:
- Exactly one SELECT statement (CTEs allowed). Never INSERT/UPDATE/DELETE/DDL.
- Use only the tables and exact column names in the schema below.
- Use the exact category values listed (e.g. status = 'CrashLoopBackOff', severity = 'P1').
- Always end with LIMIT {max_rows} or a smaller LIMIT.
- For relative dates ("last month", "this week") use the current date given below. If the
  data's date range doesn't cover the period, still write the query (it may return 0 rows).
- "Currently" / "right now" means the current status columns (pods.status, nodes.status,
  alerts.resolved = false), not a time filter.
Return JSON: {{"sql": "<the query>", "explanation": "<one sentence on what it computes>"}}.

{schema}"""


class SQLService:
    """Schema-aware Text2SQL. The schema prompt is built once (lazily) and cached."""

    def __init__(self) -> None:
        self._schema: str | None = None
        self._data_end: dt.date | None = None  # newest date in the time-stamped tables
        self._lock = asyncio.Lock()

    async def data_end(self) -> dt.date | None:
        await self.schema_context()
        return self._data_end

    async def schema_context(self) -> str:
        if self._schema is None:
            async with self._lock:
                if self._schema is None:
                    self._schema = await self._build_schema_context()
        return self._schema

    async def data_notes(self) -> str:
        """The current date and each table's date range (for answers about time periods)."""
        lines = (await self.schema_context()).split("\n")
        start = lines.index("Date ranges in the data:")
        end = lines.index("", start) if "" in lines[start:] else len(lines)
        return "\n".join([lines[0], *lines[start:end]])

    async def _build_schema_context(self) -> str:
        """Columns + types, foreign keys, category values, date ranges and 2 sample rows per
        table, read as copilot_ro (so it only ever sees what queries can use)."""
        lines = [f"Current date: {dt.date.today().isoformat()}", "", "Tables:"]
        async with await AsyncConnection.connect(readonly_conninfo()) as conn:
            columns = await _fetch_all(
                conn,
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = ANY(%s) "
                "ORDER BY table_name, ordinal_position",
                (list(OPS_TABLES),),
            )
            fks = await _fetch_all(
                conn,
                # pg_constraint, not information_schema: the latter hides constraints from
                # roles that don't own the table (copilot_ro).
                "SELECT c.conrelid::regclass::text AS table_name, a.attname AS column_name, "
                "c.confrelid::regclass::text AS ref_table, af.attname AS ref_column "
                "FROM pg_constraint c "
                "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] "
                "JOIN pg_attribute af ON af.attrelid = c.confrelid AND af.attnum = c.confkey[1] "
                "WHERE c.contype = 'f' AND c.conrelid::regclass::text = ANY(%s) "
                "ORDER BY 1, 2",
                (list(OPS_TABLES),),
            )
            for table in OPS_TABLES:
                cols = ", ".join(
                    f"{c['column_name']} {c['data_type']}"
                    for c in columns
                    if c["table_name"] == table
                )
                lines.append(f"- {table}({cols})")
            lines += ["", "Foreign keys:"]
            lines += [
                f"- {f['table_name']}.{f['column_name']} -> {f['ref_table']}.{f['ref_column']}"
                for f in fks
            ]
            lines += ["", "Category values:"]
            for table, cols in ENUM_COLUMNS.items():
                for col in cols:
                    query = sql.SQL("SELECT DISTINCT {} AS v FROM {} ORDER BY 1 LIMIT 30").format(
                        sql.Identifier(col), sql.Identifier(table)
                    )
                    values = [r["v"] for r in await _fetch_all(conn, query)]
                    lines.append(f"- {table}.{col}: {', '.join(map(str, values))}")
            lines += ["", "Date ranges in the data:"]
            newest: dt.date | None = None
            for table, col in DATE_COLUMNS.items():
                query = sql.SQL(
                    "SELECT min({c})::date AS lo, max({c})::date AS hi FROM {t}"
                ).format(c=sql.Identifier(col), t=sql.Identifier(table))
                found = await _fetch_all(conn, query)
                if found and found[0]["hi"] is not None:
                    rng = found[0]
                    lines.append(f"- {table}.{col}: {rng['lo']} to {rng['hi']}")
                    newest = max(newest or rng["hi"], rng["hi"])
            if newest is not None:
                self._data_end = newest
                lines.append(f"- There is NO data after {newest}.")
            lines += ["", "Sample rows:"]
            for table in OPS_TABLES:
                query = sql.SQL("SELECT * FROM {} LIMIT 2").format(sql.Identifier(table))
                for row in await _fetch_all(conn, query):
                    lines.append(f"- {table}: {_serialize_row(row)}")
        return "\n".join(lines)

    async def generate_sql(self, question: str) -> dict[str, Any]:
        """{"sql", "explanation", "tokens_used"}. Retries once with the error if the first
        query isn't a safe SELECT; raises SQLGenerationError if the retry fails too."""
        system = SQL_SYSTEM_PROMPT.format(
            max_rows=settings.sql_max_rows, schema=await self.schema_context()
        )
        user = f"Question: {question}"
        tokens = 0
        for attempt in (1, 2):
            try:
                result = await llm_service.generate_with_json(
                    system, user, model=settings.llm_model_answer
                )
            except llm_service.LLMJSONError as exc:
                tokens += llm_service.tokens_spent(exc)
                problem = "your reply was not valid JSON"
            else:
                tokens += result["usage"]["total_tokens"]
                query = str(result["data"].get("sql") or "").strip().rstrip(";").strip()
                if query and is_select_only(query):
                    explanation = str(result["data"].get("explanation") or "")[:500]
                    return {"sql": query, "explanation": explanation, "tokens_used": tokens}
                problem = "the query was not a single SELECT over the listed tables"
            logger.info("SQL generation attempt {} rejected: {}", attempt, problem)
            user = (
                f"Question: {question}\n\nYour previous answer was rejected: {problem}. Try again."
            )
        raise SQLGenerationError(
            "Could not generate a safe SELECT query for this question", tokens_used=tokens
        )

    async def execute_sql(self, query: str) -> list[dict[str, Any]]:
        """Run an approved query as copilot_ro: re-checked, read-only, timed out, row-capped."""
        if not is_select_only(query):
            raise SQLGenerationError("Refusing to run: not a single read-only SELECT")
        async with await AsyncConnection.connect(readonly_conninfo()) as conn:
            await conn.set_read_only(True)
            async with conn.transaction():
                await conn.execute(
                    sql.SQL("SET LOCAL statement_timeout = {}").format(
                        sql.Literal(int(settings.sql_statement_timeout_ms))
                    )
                )
                # Checked by is_select_only above and run by a SELECT-only role; the cast only
                # satisfies psycopg's LiteralString typing.
                async with conn.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(cast(LiteralString, query))
                    rows = await cursor.fetchmany(settings.sql_max_rows)
        return [_serialize_row(row) for row in rows]


@lru_cache
def get_sql_service() -> SQLService:
    return SQLService()


def rows_as_text(rows: list[dict[str, Any]], max_rows: int = 50) -> str:
    """Rows as a compact pipe table for prompts (and as the eval context)."""
    if not rows:
        return "(the query returned 0 rows)"
    header = list(rows[0])
    lines = [" | ".join(header)]
    lines += [" | ".join(str(row.get(col, "")) for col in header) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)
