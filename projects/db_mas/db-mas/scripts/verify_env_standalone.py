"""Step 1 verification: Docker + SQL + anomaly injection smoke test, no LLM involved."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark
from common_tools.immutable.query_db import query_db
from environment import anomaly_injection, docker_lifecycle

PROJECT_NAME = "db_mas_verify"


def main() -> None:
    task = benchmark.load_tasks([1])[0]
    print(f"Loaded task {task['task_id']}: {task['task']['content'][:70]}...")

    try:
        print("Starting docker compose...")
        docker_lifecycle.compose_up(PROJECT_NAME)
        docker_lifecycle.wait_for_ready()
        print("Postgres is ready.")

        print("Running init_sql...")
        docker_lifecycle.run_init_sql(task["environment"]["init_sql"])

        print("Injecting a short INSERT_LARGE_DATA anomaly (smoke test, short duration)...")
        anomaly_injection.insert_large_data(
            threads=20, duration=5, ncolumns=5, nrows=1000, colsize=50, table_name="table1"
        )

        print("Querying pg_stat_statements for recent INSERT activity...")
        result = query_db(
            "SELECT query, calls FROM pg_stat_statements "
            "WHERE query ILIKE 'INSERT%' ORDER BY calls DESC LIMIT 5;"
        )
        print(result)
        assert "ERROR" not in result, "query_db returned an error"
        print("\nSmoke test PASSED.")
    finally:
        print("Tearing down docker compose...")
        docker_lifecycle.compose_down(PROJECT_NAME)


if __name__ == "__main__":
    main()
