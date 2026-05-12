#!/usr/bin/env bash
# Inside-the-SLURM-job entrypoint for the model server. Takes one
# positional arg: a path to a per-model YAML config (see configs/).
#
# Steps:
#   1. Parse the YAML (via python3 -c) into shell vars.
#   2. Ensure the server venv exists; create + pip-install vLLM if not.
#   3. Pre-download the HF weights into a shared cache so the SLURM job
#      doesn't block on a cold first request.
#   4. Write /groups/AIC-MV/n.tzou/model_server/endpoint.json describing
#      host/port/model/job-id so client jobs can discover the server.
#   5. Install an EXIT/TERM trap that deletes the discovery file.
#   6. exec `vllm serve …` with flags from the YAML.
#
# Usage (typically driven by slurm_submit.sh, but works standalone too):
#   bash launch.sh model_server/configs/gpt_oss_120b.yaml

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_server/configs/*.yaml>" >&2
  exit 2
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[launch.sh] config not found: $CONFIG_PATH" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="${MODEL_SERVER_ROOT:-/groups/AIC-MV/n.tzou/model_server}"
HF_HOME_DIR="${HF_HOME:-/groups/AIC-MV/n.tzou/hf_cache}"

mkdir -p "$SERVER_ROOT" "$HF_HOME_DIR"

# --- Parse YAML into shell vars (python3 is part of every node image). ---
# We export each top-level scalar as $CFG_<KEY_UPPER>. Lists land in
# CFG_PIP_INSTALL_ARGS / CFG_EXTRA_VLLM_ARGS as space-separated tokens
# (each token shell-quoted so spaces are preserved inside elements).
eval "$(python3 - <<PY
import os, shlex, sys, yaml
with open("$CONFIG_PATH") as fh:
    cfg = yaml.safe_load(fh) or {}
scalars = ["hf_repo", "port", "tensor_parallel_size", "max_model_len",
           "gres", "slurm_time", "venv", "discovery_name"]
for k in scalars:
    v = cfg.get(k)
    if v is not None:
        print(f"CFG_{k.upper()}={shlex.quote(str(v))}")
for k in ("pip_install_args", "extra_vllm_args"):
    items = cfg.get(k) or []
    quoted = " ".join(shlex.quote(str(x)) for x in items)
    print(f'CFG_{k.upper()}=({quoted})')
# Default discovery_name: basename of config without .yaml extension.
# Falls back to the basename so concurrent servers don't clobber one
# another's endpoint files.
if "discovery_name" not in cfg:
    base = os.path.basename("$CONFIG_PATH")
    if base.endswith(".yaml"):
        base = base[:-5]
    elif base.endswith(".yml"):
        base = base[:-4]
    print(f"CFG_DISCOVERY_NAME={shlex.quote(base)}")
PY
)"

: "${CFG_HF_REPO:?missing hf_repo in $CONFIG_PATH}"
: "${CFG_PORT:?missing port in $CONFIG_PATH}"
: "${CFG_TENSOR_PARALLEL_SIZE:?missing tensor_parallel_size in $CONFIG_PATH}"
: "${CFG_VENV:?missing venv in $CONFIG_PATH}"
: "${CFG_DISCOVERY_NAME:?internal: discovery_name should always resolve}"
CFG_MAX_MODEL_LEN="${CFG_MAX_MODEL_LEN:-}"

DISCOVERY_FILE="$SERVER_ROOT/endpoint_${CFG_DISCOVERY_NAME}.json"

# --- Venv: create if missing, then activate. ---
if [[ ! -d "$CFG_VENV" ]]; then
  echo "[launch.sh] creating venv at $CFG_VENV"
  python3 -m venv "$CFG_VENV"
  # shellcheck disable=SC1091
  source "$CFG_VENV/bin/activate"
  pip install --upgrade pip
  if [[ ${#CFG_PIP_INSTALL_ARGS[@]} -gt 0 ]]; then
    pip install "${CFG_PIP_INSTALL_ARGS[@]}"
  fi
  pip install "huggingface_hub[cli]"
else
  # shellcheck disable=SC1091
  source "$CFG_VENV/bin/activate"
fi

# --- Pre-download weights. ---
export HF_HOME="$HF_HOME_DIR"
echo "[launch.sh] pre-downloading $CFG_HF_REPO into $HF_HOME"
huggingface-cli download "$CFG_HF_REPO" \
  --cache-dir "$HF_HOME/hub" \
  --quiet || {
  echo "[launch.sh] huggingface-cli download failed; vLLM will retry on first request" >&2
}

# --- Write discovery file. ---
HOST="$(hostname -s)"
PORT="$CFG_PORT"
JOB_ID="${SLURM_JOB_ID:-local-$$}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PID="$$"

python3 - <<PY > "$DISCOVERY_FILE"
import json
print(json.dumps({
    "host": "$HOST",
    "port": int("$PORT"),
    "model": "$CFG_HF_REPO",
    "slurm_job_id": "$JOB_ID",
    "started_at": "$STARTED_AT",
    "pid": int("$PID"),
}, indent=2))
PY
echo "[launch.sh] wrote $DISCOVERY_FILE"

cleanup() {
  echo "[launch.sh] removing $DISCOVERY_FILE"
  rm -f "$DISCOVERY_FILE"
}
trap cleanup EXIT TERM INT

# --- Build vLLM command. ---
VLLM_ARGS=(
  "$CFG_HF_REPO"
  --host 0.0.0.0
  --port "$CFG_PORT"
  --tensor-parallel-size "$CFG_TENSOR_PARALLEL_SIZE"
)
if [[ -n "$CFG_MAX_MODEL_LEN" ]]; then
  VLLM_ARGS+=(--max-model-len "$CFG_MAX_MODEL_LEN")
fi
if [[ ${#CFG_EXTRA_VLLM_ARGS[@]} -gt 0 ]]; then
  VLLM_ARGS+=("${CFG_EXTRA_VLLM_ARGS[@]}")
fi

echo "[launch.sh] exec vllm serve ${VLLM_ARGS[*]}"
exec vllm serve "${VLLM_ARGS[@]}"
