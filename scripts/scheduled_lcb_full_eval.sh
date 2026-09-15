#!/usr/bin/env bash
# Wait until a wall-clock time, then re-evaluate a run's LCB-selected node on
# the FULL benchmark with repeats, via snapshot_eval.py (latest tree snapshot
# at launch time -> HGMTree.lcb_select, same rule as the manager's final pick).
#
#   scripts/scheduled_lcb_full_eval.sh <experiment_dir> <eval_config> "<start time>" [repeats] [parallelism]
#   e.g. scripts/scheduled_lcb_full_eval.sh \
#          runs/20260913_090207_hgm_travel_1000_dsv4pro_agentic_no_editmem \
#          configs/eval_travel_dsv4pro_agentic_node.yaml "2026-09-14 07:00" 2 40
#
# <start time> is interpreted in America/Los_Angeles. Outputs land under
# <experiment_dir>/snapshots/: eval_at_budget_99999999.json (mean +- std over
# repeats) and eval_runs/lcb_full_benchmark/budget_99999999/run_<i>/logs/.
set -u
EXP="$1"; CFG="$2"; WHEN="$3"; REPEATS="${4:-2}"; PAR="${5:-40}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

TARGET=$(TZ=America/Los_Angeles date -d "$WHEN" +%s) || { echo "bad start time: $WHEN"; exit 1; }
NOW=$(date +%s)
if [ "$TARGET" -gt "$NOW" ]; then
  echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] waiting $(( (TARGET - NOW) / 60 )) min until $(TZ=America/Los_Angeles date -d "@$TARGET" '+%F %T %Z') for $EXP"
  sleep $(( TARGET - NOW ))
fi

# shellcheck disable=SC1091
source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh && conda activate hgm-dual
# shellcheck disable=SC1091
source /groups/AIC-MV/sudipta.paul/code/random/api.sh   # OpenRouter_API_KEY
export PYTHONPATH="$REPO"

echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] LCB pick from the latest snapshot:"
python3 snapshot_eval.py --experiment-dir "$EXP" --at-budget 99999999 --select lcb --list
echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] starting full-benchmark eval: repeats=$REPEATS parallelism=$PAR config=$CFG"
python3 snapshot_eval.py --config "$CFG" --experiment-dir "$EXP" \
  --at-budget 99999999 --select lcb --repeats "$REPEATS" --parallelism "$PAR"
rc=$?
echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] snapshot_eval exit code $rc"
exit $rc
