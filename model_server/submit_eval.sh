#!/usr/bin/env bash
# Submit a meta-agent main_loop.py run as a SLURM job, optionally
# pointing it at a locally-hosted vLLM server via LLM_BASE_URL.
#
# Usage:
#   # Hit OpenAI (no LLM_BASE_URL):
#   bash model_server/submit_eval.sh configs/eval_openai_medium.yaml
#
#   # Hit a local server (LLM_BASE_URL resolved from per-model discovery):
#   bash model_server/submit_eval.sh configs/eval_local_gpt_oss_120b.yaml gpt_oss_120b
#
# Honours the same SLURM_* env-var knobs as slurm/run.sh / slurm/submit.sh.
# Logs land at $SLURM_LOG_DIR/<jobid>.{out,err} (default
# /groups/AIC-MV/n.tzou/server_logs/).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [<discovery_name>]" >&2
  exit 2
fi

CONFIG_REL="$1"
DISCOVERY_NAME="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SH="${OPENAI_API_SH:-/users/n.tzou/api.sh}"

if [[ ! -f "$REPO_ROOT/$CONFIG_REL" && ! -f "$CONFIG_REL" ]]; then
  echo "[submit_eval.sh] config not found: $CONFIG_REL" >&2
  exit 2
fi

# Resolve LLM_BASE_URL from per-model discovery file when requested.
if [[ -n "$DISCOVERY_NAME" ]]; then
  LLM_BASE_URL="$(python3 "$REPO_ROOT/model_server/health.py" \
    --name "$DISCOVERY_NAME" --print-base-url)" || {
    echo "[submit_eval.sh] could not resolve URL for --name $DISCOVERY_NAME" >&2
    echo "  (server may not be running; check: python3 model_server/health.py --name $DISCOVERY_NAME)" >&2
    exit 3
  }
  if [[ -z "$LLM_BASE_URL" ]]; then
    echo "[submit_eval.sh] empty base URL for --name $DISCOVERY_NAME" >&2
    exit 3
  fi
  echo "[submit_eval.sh] routing $CONFIG_REL at $LLM_BASE_URL"
else
  echo "[submit_eval.sh] no discovery name given → using OpenAI (LLM_BASE_URL unset)"
fi

export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/n.tzou/server_logs}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-eval-$(basename "$CONFIG_REL" .yaml)}"
# Long enough for 3 rounds × ~20 cases / parallelism=8 ≈ 1-2 h plus the
# editor/manager LLM calls. 6 h leaves headroom for slow remote models.
export SLURM_TIME="${SLURM_TIME:-06:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-16}"
export SLURM_MEM="${SLURM_MEM:-32G}"
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-aic-mv-01}"
export SLURM_GRES="${SLURM_GRES:-none}"  # CPU-only meta-agent workload

mkdir -p "$SLURM_LOG_DIR"

# Compose the wrapped command. We export LLM_BASE_URL + OPENAI_API_KEY
# inline so the SLURM job inherits them; the api.sh export sets the
# real OpenAI key when going through OpenAI's endpoint.
if [[ -n "${LLM_BASE_URL:-}" ]]; then
  ENV_PREAMBLE="export LLM_BASE_URL=$(printf %q "$LLM_BASE_URL"); export OPENAI_API_KEY=EMPTY;"
else
  ENV_PREAMBLE="if [ -f \"$API_SH\" ]; then source \"$API_SH\"; fi;"
fi

CMD="$ENV_PREAMBLE PYTHONPATH=. python3 main_loop.py --config $CONFIG_REL"

exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
