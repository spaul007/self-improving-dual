"""Central configuration for db-mas: LLM endpoint, DB connection, and run limits."""
import os

MODEL = os.getenv("VLLM_MODEL", "openai/Qwen/Qwen3.6-35B-A3B")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://gpu-aic-mv-01-st-p5-node-4:18036/v1")
API_KEY = os.getenv("VLLM_API_KEY", "dummy")

DB_CONFIG = {
    "dbname": "sysbench",
    "user": "test",
    "password": "Test123_456",
    "host": "localhost",
    "port": 5432,
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSONL = os.path.join(
    PROJECT_ROOT, "..", "MARBLE", "multiagentbench", "database", "database_main.jsonl"
)
RESULTS_RAW_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
RESULTS_SCORED_DIR = os.path.join(PROJECT_ROOT, "results", "scored")
RESULTS_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "summary.json")

DOCKER_COMPOSE_FILE = os.path.join(PROJECT_ROOT, "environment", "docker-compose.yml")
DOCKER_READY_TIMEOUT_S = 60
POST_ANOMALY_SETTLE_S = 5

# Base host port for parallel task workers (run_many(max_workers>1)): worker
# slot i binds to PARALLEL_DB_PORT_BASE + i. Deliberately a disjoint range from
# DB_CONFIG["port"] (5432) so a parallel batch can never collide with a
# concurrently-running sequential/manual single-task run on the default port.
PARALLEL_DB_PORT_BASE = 15432

# Agent loop limits
MAX_SPECIALIST_TOOL_CALLS = 8
MAX_COORDINATOR_FOLLOWUPS = 1  # ask_specialist: at most once, ever, per task
MAX_COORDINATOR_QUERY_CALLS = 5  # query_db: uncapped in kind, bounded in count

# Anomaly injection defaults (not present in the benchmark's per-task spec -- see plan)
ANOMALY_DURATION_S = 60
DEFAULT_NINDEX = 5
DEFAULT_TABLE_NAME = "table1"

# query_db tool output truncation
QUERY_RESULT_MAX_ROWS = 200
QUERY_RESULT_MAX_CHARS = 4000
