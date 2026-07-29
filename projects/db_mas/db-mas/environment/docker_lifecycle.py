"""Docker Compose lifecycle for the Postgres environment (rootless, no sudo)."""
import os
import re
import subprocess
import time
from typing import List, Optional

import psycopg2
from psycopg2 import OperationalError

import config
from environment.db_conn import get_conn


def split_sql_statements(sql: str) -> List[str]:
    """Split a multi-statement SQL script into individual statements.

    Mirrors MARBLE's db_env.py splitter (split on ';' followed by a newline)
    so per-statement errors can be isolated during init_sql execution.
    """
    statements = re.split(r";\s*\n", sql)
    return [stmt.strip() for stmt in statements if stmt.strip()]


def compose_up(project_name: str, port: Optional[int] = None) -> None:
    # POSTGRES_HOST_PORT drives the compose file's "${POSTGRES_HOST_PORT:-5432}:5432"
    # mapping -- lets multiple tasks' containers coexist on distinct host ports
    # for parallel execution (see mas_workflow.run_many(max_workers>1)).
    env = dict(os.environ)
    env["POSTGRES_HOST_PORT"] = str(port or config.DB_CONFIG["port"])
    subprocess.run(
        [
            "docker", "compose",
            "-p", project_name,
            "-f", config.DOCKER_COMPOSE_FILE,
            "up", "-d",
        ],
        check=True,
        env=env,
    )


def compose_down(project_name: str) -> None:
    subprocess.run(
        [
            "docker", "compose",
            "-p", project_name,
            "-f", config.DOCKER_COMPOSE_FILE,
            "down", "-v",
        ],
        check=True,
    )


def check_db_connection() -> bool:
    try:
        conn = get_conn(application_name="lifecycle_check")
        conn.close()
        return True
    except OperationalError:
        return False


def wait_for_ready(timeout_s: int = None) -> None:
    timeout_s = timeout_s or config.DOCKER_READY_TIMEOUT_S
    start = time.time()
    while not check_db_connection():
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Postgres did not become ready within {timeout_s}s"
            )
        time.sleep(1)

    conn = get_conn(application_name="lifecycle_init")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
    cur.close()
    conn.close()


def run_init_sql(init_sql: str) -> None:
    if not init_sql:
        return
    conn = get_conn(application_name="lifecycle_init_sql")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET client_min_messages TO WARNING;")
    for statement in split_sql_statements(init_sql):
        try:
            cur.execute(statement)
        except Exception as e:  # noqa: BLE001 - report and continue, mirrors MARBLE behavior
            print(f"[init_sql] error executing statement: {e}")
    cur.close()
    conn.close()
