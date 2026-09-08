"""Standalone full-benchmark audit of node 71 (best node from the crashed
hgm_travel_full_scale_block_tagged_no_summarizer_X100Y180 run): evaluate it
on all 120 cases (60 train + 60 held-out eval), three independent times,
and report the mean. Uses the same evaluator.run(round_dir, benchmark_dir,
case_ids=...) interface HGMManager._run_top_k_full_eval already uses for
its own top-k finalist audits -- just run manually since the search
process crashed before finalization ever got a chance to touch node 71.

Each pass runs against its own copy of node 71's workspace (never the
original round_071, which stays untouched) so three passes can't cross-
contaminate each other's logs. All three passes' full per-case results,
plus a summary with the mean, are written to
runs/<crashed_run>/round_071/full_eval_3x/.

Usage: python3 scripts_eval_node71_3x.py
"""
from __future__ import annotations

import glob
import json
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_full_scale_block_tagged_no_summarizer_X100Y180.yaml"
N_REPEATS = 3

cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

run_dir = Path(sorted(glob.glob(str(REPO_ROOT / "runs/*block_tagged_no_summarizer*")))[-1])
node71_dir = run_dir / "round_071"
assert (node71_dir / "task_agent").exists(), f"node 71 workspace missing under {node71_dir}"

all_cases = list(fw.train_case_ids) + list(fw.eval_case_ids)
print(f"Evaluating node 71 on {len(all_cases)} cases, x{N_REPEATS} passes", flush=True)

out_root = node71_dir / "full_eval_3x"
out_root.mkdir(exist_ok=True)

pass_scores = []
all_case_scores = []
for i in range(1, N_REPEATS + 1):
    pass_dir = out_root / f"pass_{i}"
    if pass_dir.exists():
        shutil.rmtree(pass_dir)
    pass_dir.mkdir(parents=True)
    shutil.copytree(node71_dir / "task_agent", pass_dir / "task_agent")
    (pass_dir / "logs").mkdir(exist_ok=True)

    t0 = time.time()
    result = fw.evaluator.run(pass_dir, fw.benchmark_dir, case_ids=all_cases)
    elapsed = time.time() - t0

    per_case = [
        {"case_id": c.case_id, "passed": c.passed, "score": c.score, "error": c.error}
        for c in result.per_case
    ]
    (out_root / f"pass_{i}.json").write_text(
        json.dumps(
            {
                "pass": i,
                "composite_score": result.score,
                "passed": result.passed,
                "failed": result.failed,
                "crashed": result.crashed,
                "wall_time_s": result.wall_time_s,
                "elapsed_s": elapsed,
                "n_cases": len(per_case),
                "per_case": per_case,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pass_scores.append(result.score)
    all_case_scores.extend(c.score for c in result.per_case)
    print(
        f"pass {i}/{N_REPEATS}: composite_score={result.score:.4f} "
        f"passed={result.passed} failed={result.failed} "
        f"crashed={result.crashed} elapsed={elapsed:.0f}s",
        flush=True,
    )
    shutil.rmtree(pass_dir, ignore_errors=True)

mean_of_pass_scores = sum(pass_scores) / len(pass_scores)
mean_of_all_case_scores = sum(all_case_scores) / len(all_case_scores)

summary = {
    "node_id": 71,
    "n_repeats": N_REPEATS,
    "n_cases_per_pass": len(all_cases),
    "pass_composite_scores": pass_scores,
    "mean_of_pass_composite_scores": mean_of_pass_scores,
    "mean_of_all_360_case_scores": mean_of_all_case_scores,
}
(out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(flush=True)
print("=== SUMMARY ===")
print("per-pass composite scores:", [round(s, 4) for s in pass_scores])
print(f"mean of the 3 pass composite scores: {mean_of_pass_scores:.4f}")
print(f"mean across all {len(all_case_scores)} individual case scores: {mean_of_all_case_scores:.4f}")
print(f"outputs saved under: {out_root}")
