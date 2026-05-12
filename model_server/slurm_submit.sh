#!/usr/bin/env bash
# Submit a long-running vLLM model server as a SLURM job. Reads the
# per-model YAML config to pull out GPU count / partition / walltime,
# then defers to the repo-wide slurm/submit.sh wrapper so the same
# cgroup tear-down behaviour applies (server processes are killed
# when the SLURM job ends).
#
# Usage:
#   model_server/slurm_submit.sh model_server/configs/gpt_oss_120b.yaml
#
# Optional environment overrides:
#   SLURM_PARTITION   default gpu-aic-mv-01
#   SLURM_MEM         default 256G
#   SLURM_CPUS        default 32
#   SLURM_LOG_DIR     default /groups/AIC-MV/n.tzou/server_logs
#
# The model YAML's `gres:` and `slurm_time:` fields override the
# defaults from slurm/submit.sh.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_server/configs/*.yaml>" >&2
  exit 2
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[slurm_submit.sh] config not found: $CONFIG_PATH" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pull gres + time + hf_repo out of the YAML to set per-model SLURM args.
read -r GRES_VAL TIME_VAL HF_REPO_VAL < <(python3 - <<PY
import yaml
with open("$CONFIG_PATH") as fh:
    cfg = yaml.safe_load(fh) or {}
print(cfg.get("gres", "gpu:h100:8"),
      cfg.get("slurm_time", "24:00:00"),
      cfg.get("hf_repo", "unknown"))
PY
)

export SLURM_PARTITION="${SLURM_PARTITION:-gpu-aic-mv-01}"
export SLURM_GRES="$GRES_VAL"
export SLURM_TIME="$TIME_VAL"
export SLURM_CPUS="${SLURM_CPUS:-32}"
export SLURM_MEM="${SLURM_MEM:-256G}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/n.tzou/server_logs}"
export SLURM_JOB_NAME="model-server-$(basename "$CONFIG_PATH" .yaml)"

mkdir -p "$SLURM_LOG_DIR"

CMD="bash $REPO_ROOT/model_server/launch.sh $CONFIG_PATH"
exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
