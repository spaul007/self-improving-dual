#!/usr/bin/env bash
# Submit the 4000-budget HGM-dual travel run driven by the local vLLM
# Qwen/Qwen3.6-35B-A3B server (gpu-aic-mv-01-st-p5-node-6) to SLURM.
#
# Sibling of slurm/run_hgm_dual_qwen122b.sh — same mechanics, different
# config (35B model, evaluator.parallelism=16). Wraps slurm/submit.sh (srun
# + cgroup teardown), activates the `hgm-dual` conda env on the compute node,
# and needs no OpenAI key (all LLM calls route to the local server, which
# ignores auth).
#
# Usage:
#   bash slurm/run_hgm_dual_qwen35b.sh
#   SLURM_TIME=36:00:00 bash slurm/run_hgm_dual_qwen35b.sh   # override a knob
#
# After submitting, tail the printed job log and watch for APIConnectionError
# storms (parallelism 16 on one server is aggressive — lower it in the config
# and resubmit if the server buckles) and the final "HGM done:" line.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/hgm_dual_travel_4000_qwen35b_node6.yaml"
if [[ ! -f "$REPO_ROOT/$CONFIG" ]]; then
  echo "[run_hgm_dual_qwen35b.sh] no such config: $CONFIG" >&2
  exit 2
fi

CONDA_SH="${CONDA_SH:-/users/sudipta.paul/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-hgm-dual}"

export SLURM_PARTITION="${SLURM_PARTITION:-gpu-aic-mv-01}"
export SLURM_GRES="${SLURM_GRES:-none}"
export SLURM_TIME="${SLURM_TIME:-24:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-16}"
export SLURM_MEM="${SLURM_MEM:-32G}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-hgm-dual-qwen35b}"

export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/sudipta.paul/meta-agent/slurm}"
export META_AGENT_RUNS_ROOT="${META_AGENT_RUNS_ROOT:-/groups/AIC-MV/sudipta.paul/meta-agent/runs}"
mkdir -p "$SLURM_LOG_DIR" "$META_AGENT_RUNS_ROOT"

CMD="source \"$CONDA_SH\"; conda activate \"$CONDA_ENV\"; PYTHONPATH=. python3 main_loop.py --config $CONFIG"

echo "[run_hgm_dual_qwen35b.sh] submitting $CONFIG"
echo "[run_hgm_dual_qwen35b.sh]   conda env=$CONDA_ENV"
echo "[run_hgm_dual_qwen35b.sh]   partition=$SLURM_PARTITION gres=$SLURM_GRES time=$SLURM_TIME cpus=$SLURM_CPUS mem=$SLURM_MEM"
echo "[run_hgm_dual_qwen35b.sh]   run folders -> $META_AGENT_RUNS_ROOT ; job logs -> $SLURM_LOG_DIR"
exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
