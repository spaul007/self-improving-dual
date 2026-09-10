"""Standalone full-benchmark audit of the PRISTINE, unmodified seed with
EVERY sub-agent's backbone pointed at google/gemma-4-31b-it via OpenRouter
(see configs/eval_baseline_gemma4_31b_X100Y180.yaml) instead of the local
vLLM Qwen3.5-35B-A3B default: run all 120 cases (60 train + 60 held-out
eval), three independent times, saving full per-case details -- same
methodology as eval_seed_baseline_full_benchmark_3x.py, so the two mean
composite scores are directly comparable.

No HGM loop is ever invoked -- this is a pure baseline eval. Never
touches projects/travel_mas_refactored/seed/ itself -- each pass runs
against its own fresh copy of the pristine seed.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 eval_gemma4_31b_baseline_full_benchmark_3x.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/eval_baseline_gemma4_31b_X100Y180.yaml"
N_REPEATS = 3
OUT_ROOT = REPO_ROOT / "gemma4_31b_baseline_full_eval_3x"

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

all_cases = list(fw.train_case_ids) + list(fw.eval_case_ids)
print(f"Evaluating PRISTINE SEED (Gemma-4-31B backbone) on {len(all_cases)} cases, x{N_REPEATS} passes", flush=True)

OUT_ROOT.mkdir(exist_ok=True)

pass_scores = []
for i in range(1, N_REPEATS + 1):
    pass_dir = (OUT_ROOT / f"pass_{i}").resolve()
    if pass_dir.exists():
        shutil.rmtree(pass_dir)
    pass_dir.mkdir(parents=True)
    shutil.copytree(
        fw.seed_dir, pass_dir / "task_agent",
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
            "backbone": "google/gemma-4-31b-it",
            "per_pass_composite_scores": pass_scores,
            "mean_composite_score": mean_score,
            "n_cases_per_pass": len(all_cases),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\n=== SUMMARY ===")
print(f"per-pass composite scores: {pass_scores}")
print(f"mean of the {N_REPEATS} pass composite scores: {mean_score:.4f}")
print(f"outputs saved under: {OUT_ROOT}")
