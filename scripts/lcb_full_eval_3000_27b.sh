#!/usr/bin/env bash
# Full-benchmark (120-case) re-evaluation of the LCB-selected node at budget
# 3000 for the four Qwen3.8-27B task-agent runs. 2 repeats each, sequential,
# parallelism 16, every run measured through the SAME task-agent settings and
# the load-distributing 27B endpoint (node-3:8010) via
# configs/eval_travel_qwen38_27b_lb_nothink.yaml — thinking off, temp 0.2,
# max_output_tokens 16384, plan->JSON conversion on the 122B node-5 server.
#
#   tmux new-session -d -s lcb_full_3000_27b scripts/lcb_full_eval_3000_27b.sh
#
# Order: the two Qwen3.5-122B meta-agent runs first, then the DeepSeek V4 Pro
# pair. The last run was still optimizing when this was written, so the script
# waits (up to WAIT_MAX_H) for its run_summary.md so the comparison is at a
# real 3000 evals; if the wait expires it evaluates the latest snapshot anyway
# and says so.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh
conda activate hgm-dual
source /groups/AIC-MV/sudipta.paul/code/random/api.sh
CFG=configs/eval_travel_qwen38_27b_lb_nothink.yaml
WAIT_MAX_H=${WAIT_MAX_H:-14}
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*"; }

RUNS=(
  runs/20260924_051953_hgm_travel_3000_qwen38_27b_qwen122b_agentic_editmem
  runs/20260923_233900_hgm_travel_3000_qwen38_27b_qwen122b_agentic_no_editmem
  runs/20260924_212724_hgm_travel_3000_qwen38_27b_dsv4pro_agentic_editmem
  runs/20260924_212730_hgm_travel_3000_qwen38_27b_dsv4pro_agentic_no_editmem
)

for r in "${RUNS[@]}"; do
  if [ ! -f "$r/run_summary.md" ]; then
    log "WAIT $r is still optimizing; waiting up to ${WAIT_MAX_H}h for run_summary.md"
    deadline=$(( $(date +%s) + WAIT_MAX_H * 3600 ))
    while [ ! -f "$r/run_summary.md" ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 300; done
    if [ -f "$r/run_summary.md" ]; then log "WAIT over: $r finished"; else
      log "WAIT expired: evaluating $r at its latest snapshot (< 3000 evals)"; fi
  fi
  log "START $r"
  PYTHONPATH=. python3 snapshot_eval.py \
    --config "$CFG" \
    --experiment-dir "$r" \
    --at-budget 3000 \
    --select lcb \
    --parallelism 16 \
    --repeats 2
  log "DONE  $r (exit $?)"
done
log "all four runs evaluated"
for r in "${RUNS[@]}"; do
  python3 - "$r" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "snapshots" / "eval_at_budget_3000.json"
if p.is_file():
    d = json.load(open(p))
    print(f"{pathlib.Path(sys.argv[1]).name[16:62]:<48} node {d.get('node_id')} "
          f"mean {d.get('composite_score')} std {d.get('composite_score_std')}")
EOF
done
