"""Focused re-test: does excluding DeepInfra + requiring bf16 (no
quantization) actually reduce/eliminate the OpenRouter status="failed"
bug? Re-runs just the 10 cases that were shown (openrouter_failure_report.md
section 3g) to have succeeded originally (score >=0.5625) but hit a
TERMINAL failure in the unmodified full-120-case benchmark
(round013_full_eval_120/) -- much cheaper than repeating the full 120-case
pass, and a direct apples-to-apples test since these are exactly the
cases known to have regressed.

Sets LLM_PROVIDER_PREFERENCE (a JSON object, forwarded verbatim as
OpenRouter's `provider` field -- see platform_core/llm_wrapper.py's
_env_default_provider) to {"ignore": ["DeepInfra"], "quantizations":
["bf16"], "allow_fallbacks": false} before building the framework, so
every call in this process inherits it.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_round013_provider_fix_check.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

os.environ["LLM_PROVIDER_PREFERENCE"] = json.dumps({
    "ignore": ["DeepInfra"],
    "quantizations": ["bf16"],
    "allow_fallbacks": False,
})

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from analyze_llm_call_failures import _iter_trace_files, _analyze_trace_file

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml"
SOURCE_TASK_AGENT = (
    REPO_ROOT / "runs"
    / "20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180"
    / "round_013" / "task_agent"
)
OUT_DIR = (REPO_ROOT / "round013_provider_fix_check").resolve()

REGRESSED_CASES = ["25", "40", "36", "34", "92", "42", "65", "59", "103", "105"]
ORIGINAL_SCORES = {
    "25": 0.9375, "40": 0.9375, "36": 0.8125, "34": 0.75, "92": 0.75,
    "42": 0.75, "65": 0.75, "59": 0.75, "103": 0.625, "105": 0.5625,
}

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)
shutil.copytree(
    SOURCE_TASK_AGENT, OUT_DIR / "task_agent",
    ignore=shutil.ignore_patterns("__pycache__"),
)
(OUT_DIR / "logs").mkdir(exist_ok=True)

print(f"Re-running the {len(REGRESSED_CASES)} previously-regressed cases "
      f"with provider preference: {os.environ['LLM_PROVIDER_PREFERENCE']}", flush=True)

t0 = time.time()
result = fw.evaluator.run(OUT_DIR, fw.benchmark_dir, case_ids=REGRESSED_CASES)
elapsed = time.time() - t0

print(f"\n{'case_id':<10} {'original':>10} {'new (no provider fix)':>22} {'new (WITH fix)':>16}")
for c in sorted(result.per_case, key=lambda c: int(c.case_id)):
    orig = ORIGINAL_SCORES.get(c.case_id)
    print(f"{c.case_id:<10} {orig:>10.4f} {'0.0-0.125 (regressed)':>22} {c.score:>16.4f}")

new_mean = sum(c.score for c in result.per_case) / len(result.per_case)
orig_mean = sum(ORIGINAL_SCORES.values()) / len(ORIGINAL_SCORES)
print(f"\noriginal mean (before regression): {orig_mean:.4f}")
print(f"new mean (WITH provider fix):      {new_mean:.4f}")
print(f"elapsed: {elapsed:.0f}s")

(OUT_DIR / "result.json").write_text(
    json.dumps(
        {
            "provider_preference": json.loads(os.environ["LLM_PROVIDER_PREFERENCE"]),
            "composite_score": new_mean,
            "original_mean": orig_mean,
            "elapsed_s": elapsed,
            "per_case": [
                {"case_id": c.case_id, "score": c.score, "passed": c.passed, "error": c.error}
                for c in result.per_case
            ],
        },
        indent=2,
    ),
    encoding="utf-8",
)

totals = {"n_llm_calls": 0, "n_status_failed_retries": 0, "n_terminal_failed_responses": 0}
error_codes: dict[str, int] = {}
for trace_path in _iter_trace_files(OUT_DIR):
    r = _analyze_trace_file(trace_path)
    totals["n_llm_calls"] += r["n_llm_calls"]
    totals["n_status_failed_retries"] += r["n_status_failed_retries"]
    totals["n_terminal_failed_responses"] += r["n_terminal_failed_responses"]
    for code, n in r["error_codes"].items():
        error_codes[code] = error_codes.get(code, 0) + n

print(f"\n=== failure health (with provider fix) ===")
print(f"total calls: {totals['n_llm_calls']}")
print(f"status=failed retries: {totals['n_status_failed_retries']}")
print(f"terminal failures: {totals['n_terminal_failed_responses']}")
print(f"error codes: {error_codes}")
print(f"\nOutput saved under: {OUT_DIR}")
