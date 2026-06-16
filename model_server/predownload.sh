#!/usr/bin/env bash
# Pre-download model weights into the shared HF cache via a CPU-only
# SLURM job. Runs `huggingface-cli download <hf_repo>` from the venv
# named in the model YAML; no GPU is allocated.
#
# Usage:
#   bash model_server/predownload.sh model_server/configs/gpt_oss_120b.yaml
#
# Idempotent — re-running against a fully cached model is a no-op.
# Logs land at $SLURM_LOG_DIR/predl_<jobid>.{out,err}.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_server/configs/*.yaml>" >&2
  exit 2
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[predownload.sh] config not found: $CONFIG_PATH" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pull hf_repo + venv out of the YAML.
read -r HF_REPO_VAL VENV_VAL < <(python3 - <<PY
import yaml
with open("$CONFIG_PATH") as fh:
    cfg = yaml.safe_load(fh) or {}
print(cfg["hf_repo"], cfg["venv"])
PY
)

export SLURM_PARTITION="${SLURM_PARTITION:-cpu-prepro-queue-02}"
# CPU job: no GPU. Override SLURM_GRES=none when targeting a GPU
# partition (e.g. if cpu-prepro-queue-02 is DOWN — see CLAUDE.md note).
export SLURM_GRES="${SLURM_GRES:-}"
export SLURM_TIME="${SLURM_TIME:-06:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-4}"
export SLURM_MEM="${SLURM_MEM:-16G}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/n.tzou/server_logs}"
# Job-name prefix so it's easy to filter in `squeue -u <user>`.
export SLURM_JOB_NAME="predl-$(basename "$CONFIG_PATH" .yaml)"

mkdir -p "$SLURM_LOG_DIR"

# The wrapped command:
#   1. activates the model's server venv (created previously by
#      launch.sh on its first invocation, or here on first download)
#   2. exports HF_HOME so the cache lands on group storage
#   3. ensures huggingface_hub[cli] is installed
#   4. runs the download
HF_HOME_DIR="${HF_HOME:-/groups/AIC-MV/n.tzou/hf_cache}"
mkdir -p "$HF_HOME_DIR"

CMD=$(cat <<EOF
if [[ ! -d "$VENV_VAL" ]]; then
  echo "[predownload-job] creating venv at $VENV_VAL"
  # vLLM (installed later by launch.sh in the same venv) requires
  # Python <3.13. Use python3.12 from the parallelcluster pyenv
  # (on PATH in SLURM jobs); system python3.10 lacks ensurepip.
  python3.12 -m venv "$VENV_VAL"
fi
# shellcheck disable=SC1091
source "$VENV_VAL/bin/activate"
pip install --quiet --upgrade pip
# huggingface_hub >=1.0 ships the new \`hf\` CLI directly; the
# legacy \`huggingface-cli\` was deprecated and is a no-op in 1.x.
pip install --quiet huggingface_hub
export HF_HOME="$HF_HOME_DIR"
echo "[predownload-job] downloading $HF_REPO_VAL into \$HF_HOME"
hf download "$HF_REPO_VAL"
echo "[predownload-job] done"
EOF
)

exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
