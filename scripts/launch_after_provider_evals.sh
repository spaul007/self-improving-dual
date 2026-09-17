#!/usr/bin/env bash
# Wait until the three provider A/B evals (tmux sessions prov_streamlake,
# prov_alibaba, prov_baidu) have exited, then start the 2026-09-17 pair of
# 1000-eval runs, each in its own tmux session:
#   hgm1000_no_editmem : configs/hgm_travel_1000_qwen122b_dsv4pro_agentic_no_editmem.yaml
#   hgm1000_editmem    : configs/hgm_travel_1000_qwen122b_dsv4pro_agentic_editmem.yaml
# Run this itself inside tmux:  tmux new-session -d -s launch_pair scripts/launch_after_provider_evals.sh
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*"; }

log "waiting for prov_streamlake / prov_alibaba / prov_baidu to finish"
while tmux has-session -t prov_streamlake 2>/dev/null || tmux has-session -t prov_alibaba 2>/dev/null || tmux has-session -t prov_baidu 2>/dev/null; do
  sleep 30
done
log "provider evals done"

launch() {  # name config
  local name="$1" cfg="$2"
  tmux new-session -d -s "$name" -c "$REPO" \
    "source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh && conda activate hgm-dual \
     && source /groups/AIC-MV/sudipta.paul/code/random/api.sh \
     && PYTHONPATH=. python3 main_loop.py --config $cfg 2>&1 | tee runs/console_${name}.log"
  log "started tmux session $name ($cfg)"
}
launch hgm1000_no_editmem configs/hgm_travel_1000_qwen122b_dsv4pro_agentic_no_editmem.yaml
sleep 5   # distinct run-dir timestamps
launch hgm1000_editmem    configs/hgm_travel_1000_qwen122b_dsv4pro_agentic_editmem.yaml
tmux ls | grep -E "hgm1000_(no_)?editmem"
