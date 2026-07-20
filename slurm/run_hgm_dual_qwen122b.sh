#!/usr/bin/env bash
# Submit the 4000-budget HGM-dual travel run driven by the local vLLM
# Qwen/Qwen3.5-122B-A10B server (gpu-aic-mv-01-st-p5-node-6) to SLURM.
#
# Thin wrapper over slurm/submit.sh (which wraps the workload in `srun` so
# the job-step cgroup kills every evaluator subprocess on job end). Unlike
# slurm/run.sh this does NOT source the OpenAI key — the config routes every
# LLM call (task agent, editor, summarizer, plan->JSON conversion) at the
# local server, which ignores auth. It DOES activate the `hgm-dual` conda env
# on the compute node, because the deps (numpy/pyyaml/pydantic/openai) live
# there, not in base.
#
# Usage:
#   bash slurm/run_hgm_dual_qwen122b.sh
#
# Override any knob inline, e.g.:
#   SLURM_TIME=36:00:00 bash slurm/run_hgm_dual_qwen122b.sh
#
# After submitting, tail the job log printed below:
#   tail -f $SLURM_LOG_DIR/<jobid>.out
# Watch for "APIConnectionError" storms (server overloaded -> lower
# evaluator.parallelism in the config) and the final
# "HGM done: ... best = node K" line + the Experiment dir path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/hgm_dual_travel_4000_qwen122b_node6.yaml"
if [[ ! -f "$REPO_ROOT/$CONFIG" ]]; then
  echo "[run_hgm_dual_qwen122b.sh] no such config: $CONFIG" >&2
  exit 2
fi

# Conda env holding the repo deps. Override CONDA_ENV / CONDA_SH if yours differ.
CONDA_SH="${CONDA_SH:-/users/sudipta.paul/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-hgm-dual}"

# gpu-aic-mv-01 with GRES=none is the known-good CPU-capacity partition on
# this cluster (cpu-prepro-queue-02 is frequently DOWN). The meta-agent is
# CPU-bound here — the heavy lifting happens on the remote GPU server — so no
# local GPU is requested.
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-aic-mv-01}"
export SLURM_GRES="${SLURM_GRES:-none}"
# A 4000-budget local-Qwen run is long. 24 h is a starting point; raise it if
# the job hits wall-time before "HGM done" (there is no mid-run resume).
export SLURM_TIME="${SLURM_TIME:-24:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-16}"
export SLURM_MEM="${SLURM_MEM:-32G}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-hgm-dual-qwen122b}"

# Run folders + job logs are large — send both to group storage, not the
# small local /users disk. SLURM_LOG_DIR is read by submit.sh;
# META_AGENT_RUNS_ROOT is read by config.py (the runs_root default) and
# reaches the job via sbatch --export=ALL.
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-/groups/AIC-MV/sudipta.paul/meta-agent/slurm}"
export META_AGENT_RUNS_ROOT="${META_AGENT_RUNS_ROOT:-/groups/AIC-MV/sudipta.paul/meta-agent/runs}"
mkdir -p "$SLURM_LOG_DIR" "$META_AGENT_RUNS_ROOT"

# Command run on the compute node: activate the conda env, then main_loop.py.
CMD="source \"$CONDA_SH\"; conda activate \"$CONDA_ENV\"; PYTHONPATH=. python3 main_loop.py --config $CONFIG"

echo "[run_hgm_dual_qwen122b.sh] submitting $CONFIG"
echo "[run_hgm_dual_qwen122b.sh]   conda env=$CONDA_ENV"
echo "[run_hgm_dual_qwen122b.sh]   partition=$SLURM_PARTITION gres=$SLURM_GRES time=$SLURM_TIME cpus=$SLURM_CPUS mem=$SLURM_MEM"
echo "[run_hgm_dual_qwen122b.sh]   run folders -> $META_AGENT_RUNS_ROOT ; job logs -> $SLURM_LOG_DIR"
exec "$REPO_ROOT/slurm/submit.sh" "$CMD"
