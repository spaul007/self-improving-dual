#!/usr/bin/env bash
# Full-benchmark (120-case) re-evaluation of the LCB-selected node of each
# finished 2000-eval optimization run, 2 repeats per run, sequential.
#
#   tmux new-session -d -s lcb_full_2000 scripts/lcb_full_eval_2000.sh
#
# Each run is evaluated with ITS OWN config.snapshot.yaml, so the task agent is
# the one the optimization used (Qwen3.5-122B-A10B, reasoning medium, node-5)
# and the scorer's plan->JSON conversion targets the same endpoint. Only the
# evaluator concurrency is overridden (--parallelism 40).
#
# Selection: --select lcb --at-budget 2000 picks the latest snapshot at or below
# 2000 evals (the finalize snapshot) and recomputes the framework's LCB rule
# offline; verified 2026-09-22 to reproduce each run's own final pick
# (nodes 23 / 6 / 39 / 2).
#
# Output per run: <run>/snapshots/eval_at_budget_2000.json (mean + std over the
# 2 repeats), budget_curve_lcb_full_benchmark.csv, and isolated per-repeat logs
# under <run>/snapshots/eval_runs/lcb_full_benchmark/budget_2000/run_{1,2}/.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh
conda activate hgm-dual
source /groups/AIC-MV/sudipta.paul/code/random/api.sh
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*"; }

RUNS=(
  runs/20260919_013406_hgm_travel_2000_qwen122b_dsv4pro_agentic_editmem
  runs/20260919_013418_hgm_travel_2000_qwen122b_dsv4pro_agentic_no_editmem
  runs/20260919_013412_hgm_travel_2000_qwen122b_qwen122b_agentic_editmem
  runs/20260919_193443_hgm_travel_2000_qwen122b_qwen122b_agentic_no_editmem
)

for r in "${RUNS[@]}"; do
  log "START $r"
  PYTHONPATH=. python3 snapshot_eval.py \
    --config "$r/config.snapshot.yaml" \
    --experiment-dir "$r" \
    --at-budget 2000 \
    --select lcb \
    --parallelism 40 \
    --repeats 2
  log "DONE  $r (exit $?)"
done
log "all four runs evaluated"
for r in "${RUNS[@]}"; do
  python3 - "$r" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "snapshots" / "eval_at_budget_2000.json"
if p.is_file():
    d = json.load(open(p))
    print(f"{pathlib.Path(sys.argv[1]).name[16:60]:<46} node {d.get('node_id')} "
          f"mean {d.get('composite_score')} std {d.get('composite_score_std')} "
          f"runs {[r.get('composite_score') for r in d.get('runs', [])]}")
EOF
done
