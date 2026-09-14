"""Independent reliability check: does dropping the `quantizations:
["bf16"]` restriction -- keeping only `ignore: ["DeepInfra"]` -- still
keep the OpenRouter status="failed" failure rate low? Triggered by a
real regression found live: the current bf16-required provider
preference makes ALL FOUR llm_backbone_selection catalog slugs
(qwen/qwen3.6-27b, qwen/qwen3.8-27b, google/gemini-3.6-flash,
google/gemini-3.8-flash) 404 with "No endpoints found for the request
with quantization: bf16" -- confirmed live, one-off calls to each. A
one-off call succeeding isn't the same as sustained reliability under
real load, hence this 3x independent pass check before touching the
production configs.

3 independent passes, 32 cases each (round_013's own task_agent, same
node used for every earlier provider-fix validation this session, for
apples-to-apples comparability against round013_provider_fix_3x_120/'s
bf16-included results), parallelism 32 (production setting).

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_deepinfra_only_3x_32.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.llm_failure_health import analyze_trace_file

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml"
SOURCE_TASK_AGENT = (
    REPO_ROOT / "runs"
    / "20260912_001239_travel_mas_refactored_gemma_no_backbone_selection_X100Y180"
    / "round_013" / "task_agent"
)
N_REPEATS = 3
N_CASES = 32
OUT_ROOT = REPO_ROOT / "deepinfra_only_3x_32"

cfg = cfg_mod.load(str(CONFIG))
# Override the config's own provider (currently {ignore: DeepInfra,
# quantizations: bf16}, from the live production configs) directly on
# the loaded object, BEFORE apply_all -- setting os.environ first and
# calling apply_all after would get silently clobbered right back to
# the config's own value, since apply_task_agent_env unconditionally
# overwrites LLM_PROVIDER_PREFERENCE whenever cfg.task_agent.provider is
# truthy (confirmed live: an earlier run of this exact script showed
# the OLD bf16-including value in its own printed env var, caught and
# fixed before any real API calls were made).
cfg.task_agent.provider = {"ignore": ["DeepInfra"], "allow_fallbacks": False}
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"
print(f"Evaluator parallelism: {fw.evaluator.parallelism}", flush=True)
print(f"LLM_PROVIDER_PREFERENCE: {os.environ['LLM_PROVIDER_PREFERENCE']}", flush=True)

all_cases = sorted(list(fw.train_case_ids) + list(fw.eval_case_ids), key=int)
fixed_32 = all_cases[:N_CASES]
print(f"Fixed {N_CASES}-case batch (same for every pass): {fixed_32}", flush=True)

OUT_ROOT.mkdir(exist_ok=True)

pass_scores = []
pass_health = []
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
    result = fw.evaluator.run(pass_dir, fw.benchmark_dir, case_ids=fixed_32)
    elapsed = time.time() - t0

    health = analyze_trace_file(pass_dir / "logs" / "trace.jsonl")
    pass_scores.append(result.score)
    pass_health.append(health)

    per_case = [
        {"case_id": c.case_id, "score": c.score, "passed": c.passed, "error": c.error}
        for c in result.per_case
    ]
    (OUT_ROOT / f"pass_{i}.json").write_text(
        json.dumps(
            {"pass": i, "composite_score": result.score, "elapsed_s": elapsed,
             "health": health, "per_case": per_case},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"pass {i}/{N_REPEATS}: composite_score={result.score:.4f} elapsed={elapsed:.0f}s | "
        f"calls={health['n_llm_calls']} status_failed_retries={health['n_status_failed_retries']} "
        f"terminal_failures={health['n_terminal_failed_responses']} "
        f"error_codes={health['error_codes']}",
        flush=True,
    )
    shutil.rmtree(pass_dir, ignore_errors=True)

mean_score = sum(pass_scores) / len(pass_scores)
total_calls = sum(h["n_llm_calls"] for h in pass_health)
total_retries = sum(h["n_status_failed_retries"] for h in pass_health)
total_terminal = sum(h["n_terminal_failed_responses"] for h in pass_health)
all_error_codes: dict[str, int] = {}
for h in pass_health:
    for code, n in h["error_codes"].items():
        all_error_codes[code] = all_error_codes.get(code, 0) + n

(OUT_ROOT / "summary.json").write_text(
    json.dumps(
        {
            "provider_preference": json.loads(os.environ["LLM_PROVIDER_PREFERENCE"]),
            "n_repeats": N_REPEATS,
            "n_cases_per_pass": N_CASES,
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
print(f"\n=== SUMMARY (ignore DeepInfra only, no bf16 requirement) ===")
print(f"per-pass composite scores: {pass_scores}")
print(f"mean: {mean_score:.4f}")
print(f"total calls: {total_calls}  retries: {total_retries}  terminal failures: {total_terminal}")
if total_calls:
    print(f"incidence rate: {100.0 * (total_retries + total_terminal) / total_calls:.2f}%")
print(f"error codes: {all_error_codes}")
print(f"outputs saved under: {OUT_ROOT}")
