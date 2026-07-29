"""One-time environment setup/teardown for a single benchmark task.

This is the benchmark's actual environment contract -- bringing up a live
database, loading its schema, and injecting its anomalies -- not the
multi-agent harness that diagnoses it afterward. Mirrors why `query_db` is an
immutable tool (see common_tools/immutable/query_db.py): this defines *what*
is being tested, not how well the agents perform, so nothing here should ever
be touched by an automated harness/prompt optimizer. The harness -- the part
that's fair game to change -- lives in mas_workflow.run_task instead.
"""
import time
from typing import Any, Dict, Optional

import config
from environment import anomaly_injection, docker_lifecycle


def setup_task_environment(
    task: Dict[str, Any], project_name: str, port: Optional[int] = None
) -> Dict[str, float]:
    """Bring up a fresh Postgres for `task`, load its schema, inject its
    anomalies. Returns a timing breakdown for the two phases.

    Also resolves and pins the DB port this process's connections will use
    (see mas_workflow.run_task's docstring note on why mutating this global
    is safe under the process-per-task parallelism model).
    """
    resolved_port = port or config.DB_CONFIG["port"]
    config.DB_CONFIG["port"] = resolved_port

    timing: Dict[str, float] = {}

    t0 = time.time()
    docker_lifecycle.compose_up(project_name, port=resolved_port)
    docker_lifecycle.wait_for_ready()
    docker_lifecycle.run_init_sql(task["environment"]["init_sql"])
    timing["env_setup_s"] = time.time() - t0

    t0 = time.time()
    anomaly_injection.inject_anomalies(task["environment"]["anomalies"])
    timing["anomaly_injection_s"] = time.time() - t0
    time.sleep(config.POST_ANOMALY_SETTLE_S)

    return timing


def teardown_task_environment(project_name: str) -> None:
    """Always called regardless of task success/failure -- never skipped, so a
    failed run doesn't leak a Docker container/volume/network."""
    try:
        docker_lifecycle.compose_down(project_name)
    except Exception as e:  # noqa: BLE001 - don't mask the primary result/error with a teardown failure
        print(f"[task_setup] warning: compose_down failed for {project_name}: {e}")
