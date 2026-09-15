"""3x independent full-120-case benchmark of the OFF run's current best
node, seed_node0 (runs/20260913_214038_..._no_backbone_selection_X100Y180/
round_018), using the current production config (which already carries
the ignore-DeepInfra-only provider fix -- no bf16 requirement, see
configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml).

seed_node0 is block=individual_subagent -- the OFF run never touches
mas_llm_backbone.yaml (llm_backbone_selection is excluded from its
active_blocks), so every agent stays on the pipeline default
(google/gemma-4-31b-it). Its live HGM mean_utility was 0.5742 over 96
accumulated evals (round-robin repeat sampling); this script gets an
independent, unbiased 3-pass mean composite score on the FULL 120-case
benchmark (not a random 32/96-case subset), same methodology as
eval_node11_3x_120.py, plus a failure-health report per pass so the
reliability of this specific node's evaluation is tracked alongside its
score.

Usage: set -a && source /groups/AIC-MV/v.kulkarni1/.env && set +a && python3 eval_node18_3x_120.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.llm_failure_health import analyze_trace_file, incidence_rate_pct, DEFAULT_INCIDENCE_THRESHOLD_PCT

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml"
SOURCE_TASK_AGENT = (
    REPO_ROOT / "runs"
    / "20260913_214038_travel_mas_refactored_gemma_no_backbone_selection_X100Y180"
    / "round_000" / "task_agent"
)
N_REPEATS = 3
OUT_ROOT = REPO_ROOT / "seed_node0_full_eval_3x"

assert os.environ.get("OPENAI_API_KEY"), (
    "OPENAI_API_KEY is not set in this process's environment -- source "
    "/groups/AIC-MV/v.kulkarni1/.env with `set -a` first (plain `source` "
    "alone does not export it to child processes)."
)

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"
print(f"Evaluator parallelism (unmodified, from config): {fw.evaluator.parallelism}", flush=True)
print(f"LLM_PROVIDER_PREFERENCE: {os.environ.get('LLM_PROVIDER_PREFERENCE')}", flush=True)
print(f"OPENAI_API_KEY present: {bool(os.environ.get('OPENAI_API_KEY'))} (len={len(os.environ.get('OPENAI_API_KEY', ''))})", flush=True)

all_cases = list(fw.train_case_ids) + list(fw.eval_case_ids)
print(f"Evaluating seed_node0 on {len(all_cases)} cases, x{N_REPEATS} independent passes", flush=True)

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

    fh = analyze_trace_file(pass_dir / "logs" / "trace.jsonl")
    pass_failure_health.append(fh)
    rate = incidence_rate_pct(fh) if fh.get("n_llm_responses") else 0.0

    print(
        f"pass {i}/{N_REPEATS}: composite_score={result.score:.4f} "
        f"passed={result.passed} failed={result.failed} crashed={result.crashed} "
        f"elapsed={elapsed:.0f}s | calls={fh['n_llm_calls']} "
        f"status_failed_retries={fh['n_status_failed_retries']} "
        f"terminal_failures={fh['n_terminal_failed_responses']} "
        f"failure_rate={rate:.2f}% error_codes={fh['error_codes']}",
        flush=True,
    )
    if rate > DEFAULT_INCIDENCE_THRESHOLD_PCT:
        print(f"  ⚠️  pass {i} failure rate {rate:.2f}% exceeds {DEFAULT_INCIDENCE_THRESHOLD_PCT}% threshold", flush=True)
    shutil.rmtree(pass_dir, ignore_errors=True)

mean_score = sum(pass_scores) / len(pass_scores)
total_calls = sum(fh["n_llm_calls"] for fh in pass_failure_health)
total_retries = sum(fh["n_status_failed_retries"] for fh in pass_failure_health)
total_terminal = sum(fh["n_terminal_failed_responses"] for fh in pass_failure_health)
all_error_codes: dict[str, int] = {}
for fh in pass_failure_health:
    for code, n in fh["error_codes"].items():
        all_error_codes[code] = all_error_codes.get(code, 0) + n
overall_rate = 100.0 * (total_retries + total_terminal) / total_calls if total_calls else None

(OUT_ROOT / "summary.json").write_text(
    json.dumps(
        {
            "node_id": 0,
            "source_run": "20260913_214038_travel_mas_refactored_gemma_no_backbone_selection_X100Y180",
            "provider_preference": json.loads(os.environ["LLM_PROVIDER_PREFERENCE"]) if os.environ.get("LLM_PROVIDER_PREFERENCE") else None,
            "parallelism": fw.evaluator.parallelism,
            "n_repeats": N_REPEATS,
            "n_cases_per_pass": len(all_cases),
            "pass_composite_scores": pass_scores,
            "mean_of_pass_composite_scores": mean_score,
            "total_llm_calls": total_calls,
            "total_status_failed_retries": total_retries,
            "total_terminal_failures": total_terminal,
            "incidence_rate_pct": overall_rate,
            "error_codes": all_error_codes,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\n=== SUMMARY (seed_node0, full 120-case benchmark, 3x) ===")
print(f"per-pass composite scores: {pass_scores}")
print(f"mean of the {N_REPEATS} pass composite scores: {mean_score:.4f}")
print(f"total LLM calls across all passes: {total_calls}")
print(f"total status=failed retries: {total_retries}")
print(f"total terminal failures: {total_terminal}")
print(f"incidence rate: {overall_rate:.2f}%" if overall_rate is not None else "n/a")
print(f"error codes: {all_error_codes}")
print(f"outputs saved under: {OUT_ROOT}")
