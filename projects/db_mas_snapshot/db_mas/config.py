"""Central configuration: env vars, model/endpoint defaults, tunable limits.

Single source of truth — no endpoint, model name, path or limit should be
hard-coded anywhere else in the repo.
"""

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROMPT_CFG_PATH = Path(os.getenv("MAS_PROMPT_CFG", ROOT / "mas_prompt_cfg.yaml"))

DATA_DIR = ROOT / "data"
DATASET_PATH = Path(
    os.getenv("MAS_DB_DATASET", DATA_DIR / "marble-db" / "database_tasks.jsonl")
)
# Per-task recorded DB snapshots (one <unique_id>.json per task). The query_db
# tool replays from these — no live database is needed at inference time.
DB_CACHE_DIR = Path(os.getenv("MAS_DB_CACHE_DIR", DATA_DIR / "marble-db" / "db_cache"))

RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SCORED_DIR = RESULTS_DIR / "scored"

# Keys in the database tasks jsonl.
PROBLEM_KEY = "problem"
ID_KEY = "unique_id"
GOLD_KEY = "root_causes"          # gold labels (list)
LABELS_KEY = "labels"             # allowed candidate labels (list)

# The five candidate root causes MARBLE's database benchmark scores against.
# Fixed across all 100 tasks; per-task jsonl rows carry the same list.
LABELS = [
    "INSERT_LARGE_DATA",
    "LOCK_CONTENTION",
    "VACUUM",
    "REDUNDANT_INDEX",
    "FETCH_LARGE_DATA",
]


# --------------------------------------------------------------------------
# LLM endpoint / model configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str = "dummy_api_key"


# The MAS uses a single "main" model for all agents and the compression tool.
MAIN_LLM = LLMConfig(
    model=os.getenv("MAS_MODEL", "Qwen/Qwen3.5-35B-A3B"),
    base_url=os.getenv("MAS_BASE_URL", "http://gpu-aic-mv-02-st-p5-node-1:8000/v1"),
    api_key=os.getenv("MAS_API_KEY", "dummy_api_key"),
)


# --------------------------------------------------------------------------
# Tunable limits
# --------------------------------------------------------------------------
TEMPERATURE = float(os.getenv("MAS_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAS_MAX_TOKENS", "4096"))
TOP_K = int(os.getenv("MAS_TOP_K", "20"))
ENABLE_THINKING = os.getenv("MAS_ENABLE_THINKING", "0") == "1"

# How many tasks run concurrently, and how many LLM calls may be in flight.
# One task fans out 5 tool-looping investigators + 5 compressions + 1 lead, so
# the default task concurrency is lower than math_mas's.
MAX_CONCURRENT_TASKS = int(os.getenv("MAS_MAX_CONCURRENT_TASKS", "8"))
LLM_CONCURRENCY = int(os.getenv("MAS_LLM_CONCURRENCY", "60"))

# Retry/backoff for LLM calls.
MAX_RETRIES = int(os.getenv("MAS_MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("MAS_RETRY_DELAY", "0.5"))
RETRY_MAX_DELAY = float(os.getenv("MAS_RETRY_MAX_DELAY", "30"))

# Pass each investigator's compressed briefing (instead of its full report) to
# the lead DBA. Mirrors MASPO's short-context behaviour.
USE_COMPRESSED_CONTEXT = os.getenv("MAS_USE_COMPRESSED_CONTEXT", "1") == "1"

# --------------------------------------------------------------------------
# Tool-calling (the investigators' query_db ReAct loop)
# --------------------------------------------------------------------------
# Master switch. When off, investigators answer from the task text alone (no
# evidence) — useful only for ablations. run_inference.py --no-tools flips this.
TOOLS_ENABLED = os.getenv("MAS_TOOLS_ENABLED", "1") == "1"
# Max ReAct rounds per investigator before a forced, tool-free final answer.
TOOL_MAX_ROUNDS = int(os.getenv("MAS_TOOL_MAX_ROUNDS", "5"))
# Context budget for the tool loop: cap on a single tool result fed back to the
# model, and on the total assembled input. Snapshot table dumps can be tens of
# KB, so both matter. (Token targets; converted to chars conservatively.)
MAX_TOOL_RESULT_TOKENS = int(os.getenv("MAS_MAX_TOOL_RESULT_TOKENS", "8000"))
MAX_INPUT_TOKENS = int(os.getenv("MAS_MAX_INPUT_TOKENS", "128000"))
# Cap on a single tool result kept in the SAVED trajectory (chars). The full
# dump is reproducible from the snapshot file, so trajectories store a prefix.
TRAJECTORY_TOOL_RESULT_CHARS = int(os.getenv("MAS_TRAJECTORY_TOOL_RESULT_CHARS", "2000"))


# --------------------------------------------------------------------------
# Prompt configuration (mas_prompt_cfg.yaml)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_prompt_cfg() -> dict:
    """Parse mas_prompt_cfg.yaml once and cache it."""
    with open(PROMPT_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def agent_prompt(name: str) -> tuple[str, str]:
    """Return `(role, task)` for an agent. `role` is frozen, `task` is editable."""
    cfg = load_prompt_cfg()["agents"][name]
    return cfg["role"].strip(), cfg["task"]


def tool_prompt(name: str) -> str:
    return load_prompt_cfg()["tools"][name]
