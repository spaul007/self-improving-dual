"""3x independent full-120-case benchmark of round_013's task_agent, WITH
the OpenRouter provider fix applied (LLM_PROVIDER_PREFERENCE =
{"ignore": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks":
false} -- see platform_core/llm_wrapper.py's new `provider` param /
_env_default_provider), at production parallelism (32, from the config,
unmodified).

Goal: confirm at full scale + full concurrency (not just the earlier
10-case, parallelism-1 spot check) that excluding DeepInfra and requiring
bf16 keeps the status="failed" failure rate very low, and get an
unbiased 3-pass mean score for round_013 now that the infra bug isn't
corrupting it.

Same methodology as eval_seed_baseline_full_benchmark_3x.py (3 separate
full 120-case passes, each against a fresh copy, mean of the 3 composite
scores) -- plus a failure-health report per pass via
analyze_llm_call_failures.py's logic.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_round013_provider_fix_3x_120.py
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
N_REPEATS = 3
OUT_ROOT = REPO_ROOT / "round013_provider_fix_3x_120"

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"
print(f"Evaluator parallelism (unmodified, from config): {fw.evaluator.parallelism}", flush=True)
print(f"LLM_PROVIDER_PREFERENCE: {os.environ['LLM_PROVIDER_PREFERENCE']}", flush=True)

all_cases = list(fw.train_case_ids) + list(fw.eval_case_ids)
print(f"Evaluating round_013 on {len(all_cases)} cases, x{N_REPEATS} independent passes", flush=True)

OUT_ROOT.mkdir(exist_ok=True)

pass_scores = []
pass_failure_health = []
for i in range(1, N_REPEATS + 1):
    pass_dir = (OUT_ROOT / f"pass_{i}").resolve()
    if pass_dir.exists():
        shutil.rmtree(pass_dir)
    pass_dir.mkdir(parents=True)
    shutil.copytree(
        SOURCE_TASK_AGENT, pass_dir / "task_agent",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (pass_dir / "logs").mkdir(exist_ok=True)

    t0 = time.time()
    result = fw.evaluator.run(pass_dir, fw.benchmark_dir, case_ids=all_cases)
    elapsed = time.time() - t0

    per_case = [
        {
            "case_id": c.case_id, "passed": c.passed, "score": c.score,
            "error": c.error, "details": c.details,
        }
        for c in result.per_case
    ]
    (OUT_ROOT / f"pass_{i}.json").write_text(
        json.dumps(
            {
                "pass": i, "composite_score": result.score,
                "passed": result.passed, "failed": result.failed,
                "crashed": result.crashed, "wall_time_s": result.wall_time_s,
                "elapsed_s": elapsed, "n_cases": len(per_case),
                "per_case": per_case,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pass_scores.append(result.score)

    # Failure health for this pass (trace.jsonl lives directly under
    # pass_dir/logs, not a round_*/logs subdir -- point analyze at it
    # directly rather than relying on _iter_trace_files' round_* glob).
    fh = _analyze_trace_file(pass_dir / "logs" / "trace.jsonl")
    pass_failure_health.append(fh)

    print(
        f"pass {i}/{N_REPEATS}: composite_score={result.score:.4f} "
        f"passed={result.passed} failed={result.failed} crashed={result.crashed} "
        f"elapsed={elapsed:.0f}s | calls={fh['n_llm_calls']} "
        f"status_failed_retries={fh['n_status_failed_retries']} "
        f"terminal_failures={fh['n_terminal_failed_responses']} "
        f"error_codes={fh['error_codes']}",
        flush=True,
    )
    shutil.rmtree(pass_dir, ignore_errors=True)

mean_score = sum(pass_scores) / len(pass_scores)
total_calls = sum(fh["n_llm_calls"] for fh in pass_failure_health)
total_retries = sum(fh["n_status_failed_retries"] for fh in pass_failure_health)
total_terminal = sum(fh["n_terminal_failed_responses"] for fh in pass_failure_health)
all_error_codes: dict[str, int] = {}
for fh in pass_failure_health:
    for code, n in fh["error_codes"].items():
        all_error_codes[code] = all_error_codes.get(code, 0) + n

(OUT_ROOT / "summary.json").write_text(
    json.dumps(
        {
            "node_id": 13,
            "source_run": "20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180",
            "provider_preference": json.loads(os.environ["LLM_PROVIDER_PREFERENCE"]),
            "parallelism": fw.evaluator.parallelism,
            "n_repeats": N_REPEATS,
            "n_cases_per_pass": len(all_cases),
            "pass_composite_scores": pass_scores,
            "mean_of_pass_composite_scores": mean_score,
            "total_llm_calls": total_calls,
            "total_status_failed_retries": total_retries,
            "total_terminal_failures": total_terminal,
            "incidence_rate_pct": (
                100.0 * (total_retries + total_terminal) / total_calls if total_calls else None
            ),
            "error_codes": all_error_codes,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\n=== SUMMARY ===")
print(f"per-pass composite scores: {pass_scores}")
print(f"mean of the {N_REPEATS} pass composite scores: {mean_score:.4f}")
print(f"total LLM calls across all passes: {total_calls}")
print(f"total status=failed retries: {total_retries}")
print(f"total terminal failures: {total_terminal}")
print(f"incidence rate: {100.0 * (total_retries + total_terminal) / total_calls:.2f}%" if total_calls else "n/a")
print(f"error codes: {all_error_codes}")
print(f"outputs saved under: {OUT_ROOT}")
