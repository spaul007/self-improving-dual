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
MATH500_PATH = Path(os.getenv("MAS_MATH500_PATH", DATA_DIR / "math-500" / "math_500.jsonl"))

RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SCORED_DIR = RESULTS_DIR / "scored"

# Keys in the math-500 jsonl.
PROBLEM_KEY = "problem"
ANSWER_KEY = "answer"


# --------------------------------------------------------------------------
# LLM endpoint / model configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str = "dummy_api_key"


# The MAS uses a single "main" model for both agents and the compression tool.
MAIN_LLM = LLMConfig(
    model=os.getenv("MAS_MODEL", "Qwen/Qwen3.5-35B-A3B"),
    base_url=os.getenv("MAS_BASE_URL", "http://gpu-aic-mv-02-st-p5-node-1:8000/v1"),
    api_key=os.getenv("MAS_API_KEY", "dummy_api_key"),
)


# --------------------------------------------------------------------------
# Tunable limits
# --------------------------------------------------------------------------
TEMPERATURE = float(os.getenv("MAS_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAS_MAX_TOKENS", "8192"))
TOP_K = int(os.getenv("MAS_TOP_K", "20"))
ENABLE_THINKING = os.getenv("MAS_ENABLE_THINKING", "0") == "1"

# How many tasks run concurrently, and how many LLM calls may be in flight.
MAX_CONCURRENT_TASKS = int(os.getenv("MAS_MAX_CONCURRENT_TASKS", "16"))
LLM_CONCURRENCY = int(os.getenv("MAS_LLM_CONCURRENCY", "60"))

# Retry/backoff for LLM calls.
MAX_RETRIES = int(os.getenv("MAS_MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("MAS_RETRY_DELAY", "0.5"))
RETRY_MAX_DELAY = float(os.getenv("MAS_RETRY_MAX_DELAY", "30"))

# Pass the predictor's compressed summary (instead of its full solution) to the
# reflector. Mirrors MASPO's short-context behaviour.
USE_COMPRESSED_CONTEXT = os.getenv("MAS_USE_COMPRESSED_CONTEXT", "1") == "1"


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
