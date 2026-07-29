"""Record a per-task snapshot of MARBLE's live database environment so the MAS
can REPLAY the query_db tool offline (fast, concurrency-safe, docker-free).

Why: MARBLE's DBEnvironment is a single live Postgres-in-Docker stack on fixed
ports with per-task anomaly injection, and must run strictly serially. The MAS
runs many concurrent tasks, which the live stack cannot serve. So we run the
real environment ONCE per task here, execute a fixed battery of canonical
diagnostic queries, and store the results. At inference time
tools/immutable/query_db.py replays query_db from these snapshots.

For each task we write data/marble-db/db_cache/<task_id>.json:
    {
      "task_id": <id>,
      "root_causes": [...],           # gold, for reference
      "queries": {normalized_sql: explanation_text},   # exact-match replay
      "tables":  {table_name: dump_text},              # miss-fallback replay
    }

Requirements: docker (compose) available, the MARBLE package importable, and
psycopg2 in this interpreter (the `crew_ai` conda env is known-good). This is
the ONLY step that touches docker; run it once. Tasks are processed serially
(one Docker stack at a time) — budget a few minutes per task.

Usage (from the db_mas repo root):
    python snapshot/record_db_cache.py                 # all 100 tasks
    python snapshot/record_db_cache.py --task-id 1     # one task
    python snapshot/record_db_cache.py --warmup 20     # more settle time
"""

import argparse
import json
import os
import subprocess
import sys
import time

# --- make the db_mas root and the MARBLE package importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_ROOT = os.path.dirname(_THIS_DIR)
_MARBLE_ROOT = os.getenv("MARBLE_ROOT", "/groups/AIC-MV/sudipta.paul/code/rsi/MARBLE")
for p in (_MAS_ROOT, _MARBLE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
# Recording MUST normalize SQL exactly like replay does, so the `queries` keys
# written here are the keys query_db looks up at inference time.
from tools.immutable.query_db import _normalize_sql  # noqa: E402

DEFAULT_DATASET = os.path.join(
    _MARBLE_ROOT, "multiagentbench", "database", "database_main.jsonl"
)
DEFAULT_OUT_DIR = str(config.DB_CACHE_DIR)

# Canonical diagnostic queries an investigator is likely to issue (mirrors the
# five investigator profiles + the tables named in the task output_format).
# NOTE: keep in sync with MASPO_v2's record_db_cache.py if you want snapshots
# byte-comparable across the two repos.
BATTERY_QUERIES = [
    "SELECT query, calls, total_exec_time, mean_exec_time, rows FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 20;",
    "SELECT query, calls, total_exec_time, rows FROM pg_stat_statements WHERE query LIKE 'INSERT%' ORDER BY total_exec_time DESC LIMIT 20;",
    "SELECT query, calls, total_exec_time, rows FROM pg_stat_statements WHERE query LIKE 'SELECT%' ORDER BY total_exec_time DESC LIMIT 20;",
    "SELECT query, calls, total_plan_time, rows, mean_plan_time FROM pg_stat_statements WHERE query LIKE 'VACUUM%';",
    "SELECT * FROM pg_locks LIMIT 50;",
    "SELECT pid, state, wait_event_type, wait_event, query FROM pg_stat_activity LIMIT 50;",
    "SELECT * FROM pg_stat_user_indexes LIMIT 50;",
    "SELECT * FROM pg_indexes LIMIT 100;",
    "SELECT * FROM pg_stat_all_tables LIMIT 50;",
    "SELECT relname, n_dead_tup, n_live_tup, last_vacuum, last_autovacuum, vacuum_count, autovacuum_count FROM pg_stat_user_tables LIMIT 50;",
    "SELECT * FROM pg_stat_progress_vacuum LIMIT 20;",
]

# Full-ish per-table dumps for the miss-fallback (returned when an agent's exact
# SQL isn't in `queries` but references one of these diagnostic tables). In
# practice this is the channel nearly ALL agent queries are served from.
TABLE_DUMPS = {
    "pg_stat_statements": "SELECT query, calls, total_exec_time, mean_exec_time, rows FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 30;",
    "pg_locks": "SELECT * FROM pg_locks LIMIT 100;",
    "pg_stat_activity": "SELECT pid, state, wait_event_type, wait_event, query FROM pg_stat_activity LIMIT 100;",
    "pg_stat_user_indexes": "SELECT * FROM pg_stat_user_indexes LIMIT 100;",
    "pg_indexes": "SELECT * FROM pg_indexes LIMIT 200;",
    "pg_stat_all_tables": "SELECT * FROM pg_stat_all_tables LIMIT 100;",
    "pg_stat_user_tables": "SELECT relname, n_dead_tup, n_live_tup, last_vacuum, last_autovacuum, vacuum_count, autovacuum_count FROM pg_stat_user_tables LIMIT 100;",
    "pg_stat_progress_vacuum": "SELECT * FROM pg_stat_progress_vacuum LIMIT 50;",
}


def _make_env(env_cfg):
    """Build MARBLE's DBEnvironment without sudo: docker up -> init SQL ->
    anomaly injection, minus the init-time debug prints."""
    from marble.environments.base_env import BaseEnvironment
    from marble.environments.db_env import DBEnvironment
    from marble.environments.db_utils.diagnostic_kb import DiagnosticKB

    marble_env_dir = os.path.dirname(
        os.path.abspath(__import__("marble.environments.db_env", fromlist=["x"]).__file__)
    )

    class NoSudoDBEnvironment(DBEnvironment):
        def __init__(self, config_, name="DBEnv"):
            BaseEnvironment.__init__(self, name, config_)
            self.kb = DiagnosticKB()
            self.current_dir = marble_env_dir
            self.start_docker_containers()
            self.initialize_database(config_)   # runs init_sql + injects anomalies
            self.register_actions()

        def start_docker_containers(self):
            compose_dir = os.path.join(self.current_dir, "db_env_docker")
            subprocess.run(["docker", "compose", "down", "-v"], cwd=compose_dir, check=True)
            subprocess.run(["docker", "compose", "up", "-d", "--remove-orphans"],
                           cwd=compose_dir, check=True)

        def shutdown(self):
            compose_dir = os.path.join(self.current_dir, "db_env_docker")
            subprocess.run(["docker", "compose", "down", "-v"], cwd=compose_dir, check=False)

    return NoSudoDBEnvironment(env_cfg)


def _run_query(env, sql):
    """Execute one SQL via MARBLE's query_db_handler; return its explanation text."""
    try:
        result = env.query_db_handler(sql)
        if isinstance(result, dict):
            return str(result.get("explanation", result))
        return str(result)
    except Exception as e:  # noqa: BLE001 — record the error text rather than crash
        return f"error: {type(e).__name__}: {e}"


def record_task(rec, out_dir, warmup):
    task_id = rec["task_id"]
    env_cfg = dict(rec.get("environment", {}))
    print(f"[task {task_id}] setting up environment (docker + init + anomalies)...", flush=True)

    # Make MARBLE's `python main.py` anomaly subprocess resolve to THIS interpreter.
    interp_dir = os.path.dirname(sys.executable)
    if not os.environ.get("PATH", "").startswith(interp_dir):
        os.environ["PATH"] = interp_dir + os.pathsep + os.environ.get("PATH", "")

    env = _make_env(env_cfg)
    try:
        if warmup > 0:
            print(f"[task {task_id}] warming up {warmup}s for stats to accumulate...", flush=True)
            time.sleep(warmup)

        queries = {}
        for sql in BATTERY_QUERIES:
            queries[_normalize_sql(sql)] = _run_query(env, sql)

        tables = {}
        for table, sql in TABLE_DUMPS.items():
            tables[table] = _run_query(env, sql)

        snapshot = {
            "task_id": task_id,
            "root_causes": rec.get("task", {}).get("root_causes", []),
            "queries": queries,
            "tables": tables,
        }
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{task_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"[task {task_id}] wrote {out_path}", flush=True)
    finally:
        try:
            env.shutdown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Record per-task DB snapshots for query_db replay.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="MARBLE database_main.jsonl (full simulation configs)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Record only this task_id (default: all).")
    parser.add_argument("--warmup", type=int, default=15,
                        help="Seconds to wait after anomaly injection before snapshotting.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Record at most this many tasks (debug).")
    args = parser.parse_args()

    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.task_id is not None and rec.get("task_id") != args.task_id:
                continue
            records.append(rec)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Recording {len(records)} task snapshot(s) to {args.out_dir}/ (serial).")
    ok, failed = 0, []
    for rec in records:
        try:
            record_task(rec, args.out_dir, args.warmup)
            ok += 1
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't stop the sweep
            print(f"[task {rec.get('task_id')}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(rec.get("task_id"))
    print(f"Done. {ok} recorded, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
