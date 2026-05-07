#!/usr/bin/env bash
# Submit a meta-agent experiment (main_loop.py) as a SLURM job. Sources the
# OpenAI API key from /users/n.tzou/api.sh inside the job so the compute
# node has it. Workload runs under `srun`, so every Python subprocess the
# evaluator spawns is in the job-step cgroup and is killed on job end.
#
# Usage:
#   slurm/run.sh configs/travel.yaml
#   slurm/run.sh configs/default.yaml
#
# Env-var knobs same as slurm/submit.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SH="${OPENAI_API_SH:-/users/n.tzou/api.sh}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  exit 2
fi

CONFIG_REL="$1"

# Resolve relative paths against the repo root (the job runs --chdir=$REPO_ROOT).
if [[ "$CONFIG_REL" = /* ]]; then
  CONFIG_PATH="$CONFIG_REL"
else
  CONFIG_PATH="$CONFIG_REL"
fi

if [[ ! -f "$REPO_ROOT/$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
  echo "[run.sh] config not found: $CONFIG_PATH" >&2
  exit 2
fi

export SLURM_JOB_NAME="${SLURM_JOB_NAME:-meta-agent-$(basename "$CONFIG_PATH" .yaml)}"

# The wrapped command sources the API key inside the SLURM job so it is
# available on the compute node, then invokes main_loop.py.
CMD="if [ -f \"$API_SH\" ]; then source \"$API_SH\"; fi; PYTHONPATH=. python3 main_loop.py --config $CONFIG_PATH"

exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
