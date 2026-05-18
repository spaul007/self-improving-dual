#!/usr/bin/env bash
# Submit a Huxley-Gödel-Machine (HGM) optimization run to SLURM.
#
# Thin wrapper over slurm/run.sh that pre-sets resource defaults sized for
# an HGM run (eval_budget=400 + the finalist re-evaluation pass): a B=400
# travel run takes ~9-10 h, so the wall-time default is 14 h; shopping and
# math finish much sooner.
#
# Usage:
#   slurm/run_hgm.sh travel        # -> configs/hgm_travel.yaml
#   slurm/run_hgm.sh shopping      # -> configs/hgm_shopping.yaml
#   slurm/run_hgm.sh math          # -> configs/hgm_math.yaml  (quick smoke)
#
# Any env var below can be overridden on the command line, e.g.:
#   SLURM_TIME=06:00:00 slurm/run_hgm.sh shopping
#
# Output goes to the group filesystem (SLURM_LOG_DIR / META_AGENT_RUNS_ROOT
# below) — the local /users disk is small. After submitting, tail with:
#   tail -f /groups/AIC-MV/n.tzou/meta-agent/slurm/<jobid>.out
# The run ends with "HGM done: ... best = node K", "held-out eval ... Y",
# and the Experiment dir path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT="${1:-}"
if [[ -z "$PROJECT" ]]; then
  echo "usage: $0 <travel|shopping|math>" >&2
  exit 2
fi

CONFIG="configs/hgm_${PROJECT}.yaml"
if [[ ! -f "$REPO_ROOT/$CONFIG" ]]; then
  echo "[run_hgm.sh] no such config: $CONFIG" >&2
  echo "             expected one of: travel, shopping, math" >&2
  exit 2
fi

# gpu-aic-mv-01 with GRES=none is the known-good CPU-capacity partition on
# this cluster (the nominal cpu-prepro-queue-02 is frequently DOWN). HGM is
# CPU-bound — the heavy work is OpenAI API calls — so no GPU is requested.
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-aic-mv-01}"
export SLURM_GRES="${SLURM_GRES:-none}"
export SLURM_TIME="${SLURM_TIME:-14:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-16}"
export SLURM_MEM="${SLURM_MEM:-32G}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-hgm-${PROJECT}}"

# HGM runs are the largest output (per-node task_agent copies + traces +
# per-case JSON). Send run folders and SLURM job logs to the group
# filesystem so they don't fill the small local /users disk. SLURM_LOG_DIR
# is read by submit.sh; META_AGENT_RUNS_ROOT is read by config.py (the
# `runs_root` default) and reaches the job via sbatch --export=ALL.
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/n.tzou/meta-agent/slurm}"
export META_AGENT_RUNS_ROOT="${META_AGENT_RUNS_ROOT:-/groups/AIC-MV/n.tzou/meta-agent/runs}"

echo "[run_hgm.sh] submitting $CONFIG"
echo "[run_hgm.sh]   partition=$SLURM_PARTITION gres=$SLURM_GRES time=$SLURM_TIME cpus=$SLURM_CPUS mem=$SLURM_MEM"
echo "[run_hgm.sh]   run folders -> $META_AGENT_RUNS_ROOT ; job logs -> $SLURM_LOG_DIR"
exec "$REPO_ROOT/slurm/run.sh" "$CONFIG"
