"""Env vars, model/endpoint defaults, and tunable limits for the Shopping MAS."""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_RAW = REPO_ROOT / "results" / "raw"
RESULTS_SCORED = REPO_ROOT / "results" / "scored"


def level_database_dir(level: int) -> Path:
    """Pristine extracted benchmark database for a level (read-only master)."""
    return DATA_DIR / f"database_level{level}"


def level_query_meta(level: int) -> Path:
    return DATA_DIR / f"level_{level}_query_meta.json"


@dataclass
class ServerConfig:
    """OpenAI-compatible endpoint (vLLM). Must support native tool calling."""
    served_model_name: str = os.environ.get("MAS_MODEL", "Qwen/Qwen3.5-35B-A3B")
    # Renamed from the vendor's own MAS_URL/MAS_KEY to match this framework's
    # established convention (math_mas/db_mas/wikihop_mas all use
    # MAS_BASE_URL/MAS_API_KEY) -- see integrating.md Step 0.
    url: str = os.environ.get("MAS_BASE_URL", "http://gpu-aic-mv-02-st-p5-node-1:8000/v1")
    key: str = os.environ.get("MAS_API_KEY", "EMPTY")


@dataclass
class MASConfig:
    server: ServerConfig = field(default_factory=ServerConfig)

    # Benchmark level (1: cheapest, 2: + budget range, 3: + coupons).
    level: int = int(os.environ.get("MAS_LEVEL", 1))

    # Sampling. max_tokens must be generous: the scout's tool rounds and the
    # optimizer's worked arithmetic are long, and a truncated reply costs a
    # retry (see llm_client's finish_reason == "length" handling).
    temperature: float = float(os.environ.get("MAS_TEMPERATURE", 0.2))
    max_tokens: int = int(os.environ.get("MAS_MAX_TOKENS", 16384))

    # Per-agent tool-dispatch loop: max tool-call rounds before tools are
    # withheld and the agent must answer.
    max_tool_rounds: int = int(os.environ.get("MAS_MAX_TOOL_ROUNDS", 12))

    # Hard budget on LLM calls per case across all agents. The benchmark
    # allows 400 per sample; this system averages ~65.
    max_llm_calls: int = int(os.environ.get("MAS_MAX_LLM_CALLS", 240))

    # Repair loop: (optimize -> execute) reruns for products that failed at
    # execution time (e.g. out of stock).
    max_repair_iterations: int = 2

    # Candidates kept per line item (cheapest-first, as ordered by the scout)
    # that are passed to the optimizer.
    top_k_candidates: int = int(os.environ.get("MAS_TOP_K", 8))

    # Robustness limits
    json_retries: int = 4      # re-asks when an agent returns unparseable JSON
    request_retries: int = 4   # HTTP-level retries (with exponential backoff)
    backoff_seconds: float = 2.0

    # Batch evaluation: cases processed in parallel threads (each case has
    # its own database copy).
    workers: int = int(os.environ.get("MAS_WORKERS", 4))

    # Within one case, per-line-item scouts run concurrently (they are
    # independent and use read-only catalog tools). Total in-flight requests
    # are roughly workers x scout_workers, so keep the product near the
    # server's healthy concurrency.
    scout_workers: int = int(os.environ.get("MAS_SCOUT_WORKERS", 4))
