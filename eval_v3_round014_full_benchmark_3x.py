"""Standalone full-benchmark audit of a SPECIFIC HGM node's finalized
task_agent state -- here, round_014 of the
20260910_040445_travel_mas_refactored_full_scale_block_tagged_no_summarizer_X100Y180
run (the "harder catalog, no 122B" production run): a clean,
correctly-scoped llm_backbone_selection edit assigning
qwen/qwen3.6-35b-a3b to the sightseeing role via OpenRouter, no other
files touched.

Same methodology as eval_seed_baseline_full_benchmark_3x.py (3 independent
full 120-case passes, mean of the 3 composite scores) and matches the
existing full_eval_3x finalize convention already used elsewhere in this
codebase (see runs/.../round_071/full_eval_3x/summary.json from a prior
session) -- just invoked standalone against an in-progress run's node
instead of through HGMManager._run_top_k_full_eval's finalize step.

Never touches the live run's own round_014 directory -- each pass runs
against its own fresh copy.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_v3_round014_full_benchmark_3x.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_full_scale_block_tagged_no_summarizer_X100Y180.yaml"
SOURCE_TASK_AGENT = (
    REPO_ROOT / "runs"
    / "20260910_040445_travel_mas_refactored_full_scale_block_tagged_no_summarizer_X100Y180"
    / "round_014" / "task_agent"
)
N_REPEATS = 3
OUT_ROOT = REPO_ROOT / "v3_round014_full_eval_3x"

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

assert SOURCE_TASK_AGENT.exists(), f"missing: {SOURCE_TASK_AGENT}"

all_cases = list(fw.train_case_ids) + list(fw.eval_case_ids)
print(f"Evaluating v3/round_014 on {len(all_cases)} cases, x{N_REPEATS} passes", flush=True)

OUT_ROOT.mkdir(exist_ok=True)

pass_scores = []
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
    print(
        f"pass {i}/{N_REPEATS}: composite_score={result.score:.4f} "
        f"passed={result.passed} failed={result.failed} crashed={result.crashed} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )
    shutil.rmtree(pass_dir, ignore_errors=True)

mean_score = sum(pass_scores) / len(pass_scores)
(OUT_ROOT / "summary.json").write_text(
    json.dumps(
        {
            "node_id": 14,
            "source_run": "20260910_040445_travel_mas_refactored_full_scale_block_tagged_no_summarizer_X100Y180",
            "n_repeats": N_REPEATS,
            "n_cases_per_pass": len(all_cases),
            "pass_composite_scores": pass_scores,
            "mean_of_pass_composite_scores": mean_score,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\n=== SUMMARY ===")
print(f"per-pass composite scores: {pass_scores}")
print(f"mean of the {N_REPEATS} pass composite scores: {mean_score:.4f}")
print(f"outputs saved under: {OUT_ROOT}")
