"""IMMUTABLE — the benchmark's database environment contract.

``query_db(sql)`` is the ONE tool MARBLE's database RCA benchmark gives the
investigators, backed there by a live Postgres-in-Docker stack with per-task
anomaly injection. That stack is single-instance / fixed-port / stateful and
must run strictly serially, which is incompatible with a batch of concurrent
MAS rollouts. So this repo **replays a per-task snapshot** instead:

- ``snapshot/record_db_cache.py`` runs the real environment ONCE per task
  offline and records ``data/marble-db/db_cache/<task_id>.json``:
      {"queries": {normalized_sql: explanation_text},
       "tables":  {table_name: dump_text}}
- Here, ``query_db`` replays from the snapshot of the CURRENTLY-active task,
  selected via a ``contextvars.ContextVar`` so concurrent tasks each see their
  own snapshot.

Replay semantics (identical to MASPO_v2's tools/db_registry.py):
  exact          — the SQL (normalized) matches a recorded battery query.
  table_fallback — not recorded, but the SQL references a known diagnostic
                   table: its recorded dump is returned, with a note that
                   filters/ordering/LIMIT were not applied.
  miss           — neither; an error message points the agent at the tables
                   that ARE snapshotted.

The tool schema and result shape mirror MARBLE's ``query_db_handler`` so agents
see byte-compatible behaviour. Rewriting this file changes the environment the
benchmark measures against, not how well the agents perform — an automated
prompt/tool optimizer must never touch it.
"""

import contextvars
import json
import re
import threading
from typing import Any, Dict, List, Optional

# Per-task active snapshot. Set by set_db_cache(...) at the start of each
# task's run (see mas_workflow.run_task) and read inside query_db_replay.
_CURRENT_DB_CACHE: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("current_db_cache", default=None)
)

# ---------------------------------------------------------------------------
# Cache-coverage counters: how often query_db is answerable from the recorded
# snapshot. Persisted into the raw results payload by run_inference.py (MASPO
# only printed them). Guarded by a lock in case tool handlers ever run on
# worker threads.
# ---------------------------------------------------------------------------
_STATS_LOCK = threading.Lock()
_STATS = {"exact": 0, "table_fallback": 0, "miss": 0, "unbound": 0, "total": 0}


def _bump(kind: str) -> None:
    with _STATS_LOCK:
        _STATS["total"] += 1
        _STATS[kind] = _STATS.get(kind, 0) + 1


def reset_db_stats() -> None:
    with _STATS_LOCK:
        for k in _STATS:
            _STATS[k] = 0


def get_db_stats() -> Dict[str, int]:
    with _STATS_LOCK:
        return dict(_STATS)


def format_db_stats(label: str = "") -> str:
    s = get_db_stats()
    tot = s["total"] or 1
    covered = s["exact"] + s["table_fallback"]
    return (
        f"[query_db coverage{(' ' + label) if label else ''}] "
        f"{s['total']} calls | exact={s['exact']} "
        f"table_fallback={s['table_fallback']} "
        f"MISS={s['miss']} ({100*s['miss']/tot:.1f}%) "
        f"unbound={s['unbound']} | covered={100*covered/tot:.1f}%"
    )


# Diagnostic tables an agent might reference; used for the miss-fallback (return
# the cached full dump of the referenced table when exact SQL isn't cached).
_KNOWN_DB_TABLES = [
    "pg_stat_statements",
    "pg_locks",
    "pg_stat_activity",
    "pg_stat_user_indexes",
    "pg_stat_all_indexes",
    "pg_indexes",
    "pg_stat_all_tables",
    "pg_stat_user_tables",
    "pg_stat_progress_vacuum",
]


def set_db_cache(cache: Optional[Dict[str, Any]]):
    """Bind the active per-task snapshot for the current async context.

    Returns the ContextVar token so the caller can reset() it afterwards.
    Called per-task in mas_workflow.run_task.
    """
    return _CURRENT_DB_CACHE.set(cache)


def reset_db_cache(token) -> None:
    try:
        _CURRENT_DB_CACHE.reset(token)
    except (ValueError, LookupError):
        pass


def load_db_cache(path: str) -> Dict[str, Any]:
    """Load a recorded snapshot JSON; tolerate missing/broken files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"queries": {}, "tables": {}}
        data.setdefault("queries", {})
        data.setdefault("tables", {})
        return data
    except Exception:
        return {"queries": {}, "tables": {}}


def _normalize_sql(sql: str) -> str:
    """Canonicalize an SQL string for cache lookup: lowercase, collapse
    whitespace, strip a trailing semicolon. snapshot/record_db_cache.py uses
    THIS function when it writes the `queries` keys, so recording and replay
    can never disagree."""
    s = (sql or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";").strip()
    return s


def _referenced_table(sql_norm: str) -> Optional[str]:
    """Return the first known diagnostic table referenced in the SQL, if any."""
    for t in _KNOWN_DB_TABLES:
        if t in sql_norm:
            return t
    return None


def _result(status: str, explanation: str, replay: str) -> Dict[str, Any]:
    """MARBLE-shape result dict plus a `replay` coverage tag.

    The `replay` key is bookkeeping for trajectories/coverage stats; the tool
    loop strips it before the result is fed back to the model, so the
    model-visible payload stays MARBLE-shaped.
    """
    return {
        "status": status,
        "function_name": "query_db",
        "explanation": explanation,
        "replay": replay,
    }


def query_db_replay(sql: str) -> Dict[str, Any]:
    """Replay a query against the active task's recorded snapshot.

    Exact (normalized) match -> cached result. Miss but the SQL references a
    known diagnostic table -> that table's cached dump, flagged as unfiltered.
    Total miss / no snapshot -> a clear message that still points the agent at
    the diagnostic tables. Never raises.
    """
    cache = _CURRENT_DB_CACHE.get()
    if cache is None:
        _bump("unbound")
        return _result(
            "error",
            "No database snapshot is bound for this task; query_db is unavailable. "
            "Proceed using the evidence already gathered by teammates.",
            replay="unbound",
        )

    queries: Dict[str, str] = cache.get("queries", {}) or {}
    tables: Dict[str, str] = cache.get("tables", {}) or {}
    norm = _normalize_sql(sql)

    if norm in queries:
        _bump("exact")
        return _result("success", f"Query executed. Result: {queries[norm]}", replay="exact")

    table = _referenced_table(norm)
    if table and table in tables:
        _bump("table_fallback")
        return _result(
            "success",
            f"No exact snapshot for this query. Showing the recorded snapshot of "
            f"`{table}` (filters/ordering/LIMIT not applied). Result: {tables[table]}",
            replay="table_fallback",
        )

    _bump("miss")
    avail = ", ".join(sorted(tables.keys())) or "none"
    return _result(
        "error",
        "This query is not in the recorded snapshot for this task. "
        f"Snapshotted diagnostic tables available: {avail}. "
        "Query one of those tables (e.g. pg_stat_statements, pg_locks, "
        "pg_stat_user_indexes, pg_stat_all_tables) to get evidence.",
        replay="miss",
    )


# ---------------------------------------------------------------------------
# OpenAI-format tool schema + dispatcher, consumed by the ReAct loop in
# llm_client.acall_with_tools. The description mirrors MARBLE's / crewai's
# query_db so agents see the same affordances.
# ---------------------------------------------------------------------------
TOOL_DESCRIPTIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Query the PostgreSQL database with the given SQL statement to "
                "diagnose a performance issue. Keep to diagnostic tables and make "
                "sure the query won't hang the database. One query at a time. "
                "Useful tables: pg_stat_statements (query stats, e.g. SELECT query, "
                "total_exec_time FROM pg_stat_statements ORDER BY total_exec_time "
                "DESC LIMIT 10), pg_locks / pg_stat_activity (lock waits, blocked "
                "queries), pg_indexes / pg_stat_user_indexes (index definitions and "
                "usage), pg_stat_all_tables / pg_stat_user_tables (vacuum/analyze "
                "stats, dead tuples). Don't use LIMIT higher than ~100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL statement to execute.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_HANDLERS = {
    "query_db": query_db_replay,
}


def apply_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a single tool call.

    ALWAYS returns a JSON-serializable dict and NEVER raises, so a malformed
    call can't crash the agent or the run.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _result("error", f"Unknown tool: {name}", replay="miss")
    try:
        return handler(**(arguments or {}))
    except TypeError as e:
        return _result("error", f"Invalid arguments for {name}: {e}", replay="miss")
    except Exception as e:  # noqa: BLE001 — resilience: never let a tool crash the run
        return _result("error", f"{type(e).__name__}: {e}", replay="miss")
