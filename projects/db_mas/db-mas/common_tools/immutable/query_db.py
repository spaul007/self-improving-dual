"""query_db: shared by all 5 specialist agents -- their common investigation tool.

Immutable: this tool's schema/implementation is off-limits to any automated
prompt/tool optimizer. It's the system's actual interface to the benchmark
environment (matching the original MARBLE `query_db` contract) -- an
optimizer rewriting its description or parameters would risk changing what
the benchmark is actually testing, not just how well the agents perform it.
"""
import re
from typing import List

import config
from environment.db_conn import get_conn

QUERY_DB_TOOL = {
    "type": "function",
    "function": {
        "name": "query_db",
        "description": (
            "Query the PostgreSQL database with the given SQL statement. "
            "Keep to read-only investigation queries against the system catalogs/stats "
            "views; avoid queries that could hang the database. Recommended to run one "
            "query at a time. You will get the result of the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "The SQL statement to execute. Useful tables/views: "
                        "pg_stat_statements (detailed query statistics), "
                        "pg_locks and pg_stat_activity (lock waits/contention, blocked queries), "
                        "pg_indexes and pg_stat_user_indexes (index definitions/usage), "
                        "pg_stat_all_tables, pg_stat_user_tables, pg_stat_progress_vacuum "
                        "(vacuum/autovacuum/dead-tuple statistics). "
                        "Example: SELECT query, total_exec_time FROM pg_stat_statements "
                        "ORDER BY total_exec_time DESC LIMIT 10; "
                        "Avoid unbounded LIMITs (keep them under 100)."
                    ),
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}


def split_sql_statements(sql: str) -> List[str]:
    statements = re.split(r";\s*\n", sql)
    parts = [stmt.strip() for stmt in statements if stmt.strip()]
    if not parts:
        # no newline-separated statements found; fall back to the whole string
        stripped = sql.strip()
        return [stripped] if stripped else []
    return parts


def _format_rows(colnames: List[str], rows: list) -> str:
    lines = []
    if colnames:
        lines.append(" | ".join(colnames))
    for row in rows[: config.QUERY_RESULT_MAX_ROWS]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > config.QUERY_RESULT_MAX_ROWS:
        lines.append(f"... ({len(rows) - config.QUERY_RESULT_MAX_ROWS} more rows truncated)")
    text = "\n".join(lines) if lines else "(no rows returned)"
    if len(text) > config.QUERY_RESULT_MAX_CHARS:
        text = text[: config.QUERY_RESULT_MAX_CHARS] + "\n... (truncated)"
    return text


def query_db(sql: str) -> str:
    """Run one or more ';'-separated SQL statements and return a human-readable result string."""
    conn = None
    try:
        conn = get_conn(application_name="specialist_query")
        cur = conn.cursor()
        last_colnames: List[str] = []
        last_rows: list = []
        for statement in split_sql_statements(sql):
            cur.execute(statement)
            if cur.description:
                last_colnames = [d[0] for d in cur.description]
                last_rows = cur.fetchall()
            else:
                last_colnames, last_rows = [], []
        conn.commit()
        cur.close()
        return _format_rows(last_colnames, last_rows)
    except Exception as e:  # noqa: BLE001 - surfaced to the agent as a tool result
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return f"ERROR: {e}"
    finally:
        if conn is not None:
            conn.close()
