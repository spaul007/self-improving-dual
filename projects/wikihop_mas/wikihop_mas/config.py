"""Central configuration: env vars, model/endpoint defaults, tunable limits.

Single source of truth — no endpoint, model name, path or limit should be
hard-coded anywhere else in the repo. Mirrors math_mas/config.py's convention.
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
WIKIHOP_DATA_DIR = DATA_DIR / "2wikimultihopqa"
WIKIHOP_DEV_PATH = Path(os.getenv("MAS_WIKIHOP_DEV_PATH", WIKIHOP_DATA_DIR / "dev.jsonl"))

RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SCORED_DIR = RESULTS_DIR / "scored"

# Keys in the 2WikiMultihopQA jsonl (post parquet->jsonl conversion).
ID_KEY = "_id"
QUESTION_KEY = "question"
ANSWER_KEY = "answer"
TYPE_KEY = "type"
CONTEXT_KEY = "context"
SUPPORTING_FACTS_KEY = "supporting_facts"
EVIDENCES_KEY = "evidences"


# --------------------------------------------------------------------------
# LLM endpoint / model configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str = "dummy_api_key"


# The MAS uses a single "main" model for every agent and tool call.
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

# How many questions run concurrently in run_many (plain ThreadPoolExecutor, no asyncio).
MAX_CONCURRENT_TASKS = int(os.getenv("MAS_MAX_CONCURRENT_TASKS", "8"))

# Retry/backoff for LLM calls (synchronous, time.sleep-based).
MAX_RETRIES = int(os.getenv("MAS_MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("MAS_RETRY_DELAY", "0.5"))
RETRY_MAX_DELAY = float(os.getenv("MAS_RETRY_MAX_DELAY", "30"))

# --------------------------------------------------------------------------
# 2WikiMultihopQA topology dispatch
# --------------------------------------------------------------------------
INDEPENDENT_TYPES = {"comparison", "bridge_comparison"}   # order-agnostic hops
DEPENDENT_TYPES = {"inference", "compositional"}          # chained hops

# --------------------------------------------------------------------------
# Tool-call round caps (bounded LLM<->tool loops)
# --------------------------------------------------------------------------
RETRIEVER_MAX_ROUNDS = int(os.getenv("MAS_RETRIEVER_MAX_ROUNDS", "3"))
CONCLUDER_MAX_ROUNDS = int(os.getenv("MAS_CONCLUDER_MAX_ROUNDS", "3"))

# BM25 (tools/immutable/search_context.py)
BM25_K1 = float(os.getenv("MAS_BM25_K1", "1.5"))
BM25_B = float(os.getenv("MAS_BM25_B", "0.75"))
SEARCH_DEFAULT_K = int(os.getenv("MAS_SEARCH_DEFAULT_K", "5"))
SEARCH_MAX_K = int(os.getenv("MAS_SEARCH_MAX_K", "10"))

# Grounding retry (per-hop bound; see mas_workflow._run_concluder_with_bounded_retry)
MAX_HOP_RETRIES = int(os.getenv("MAS_MAX_HOP_RETRIES", "1"))
QUOTE_FUZZY_MATCH_THRESHOLD = float(os.getenv("MAS_QUOTE_FUZZY_THRESHOLD", "0.85"))

# Ablation / debug toggle: trust the dataset's gold `type` instead of the
# Decomposer's own classification (mirrors math_mas's USE_COMPRESSED_CONTEXT pattern).
ORACLE_TYPE = os.getenv("MAS_WIKIHOP_ORACLE_TYPE", "0") == "1"


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
