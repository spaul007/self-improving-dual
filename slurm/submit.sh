#!/usr/bin/env bash
# Generic sbatch wrapper. Submits an arbitrary command as a SLURM job and
# wraps the workload in `srun` so that SLURM's cgroup tracks every
# descendant process. When the job ends or is cancelled (scancel,
# walltime, OOM, login-session drop), the cgroup tear-down kills every
# subprocess the workload spawned. This is the property we need: the
# meta-agent's per-case Python evaluator subprocesses cannot become
# orphaned zombies on the compute node.
#
# Usage:
#   slurm/submit.sh "<shell command to run>"
#
# Optional environment overrides (set before invoking):
#   SLURM_PARTITION   default cpu-prepro-queue-02
#   SLURM_TIME        default 04:00:00
#   SLURM_CPUS        default 4
#   SLURM_MEM         default 16G
#   SLURM_JOB_NAME    default meta-agent
#   SLURM_LOG_DIR     default $REPO_ROOT/runs/slurm
#
# Example:
#   slurm/submit.sh "PYTHONPATH=. python3 -m unittest tests.test_smoke"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command>" >&2
  exit 2
fi

CMD="$*"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PART="${SLURM_PARTITION:-cpu-prepro-queue-02}"
TIME="${SLURM_TIME:-04:00:00}"
CPUS="${SLURM_CPUS:-4}"
MEM="${SLURM_MEM:-16G}"
JOB_NAME="${SLURM_JOB_NAME:-meta-agent}"
LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/runs/slurm}"
# Optional --gres value. Set to "none" when targeting a GPU partition for a
# CPU-only workload, so the scheduler doesn't allocate a GPU we won't use.
GRES="${SLURM_GRES:-}"

mkdir -p "$LOG_DIR"

# Write the job body to a temp script with a bash shebang and submit that
# instead of using --wrap. sbatch --wrap runs through /bin/sh (dash on
# Ubuntu) which doesn't support `set -o pipefail`, but we want the strict
# settings.
#
# We execute the workload via `srun bash -c "$CMD"` so the cgroup tracks
# the bash and every descendant. If we just inlined $CMD into the script
# body it would run as a child of the slurm batch shell, but srun gives
# us a job-step cgroup that is more aggressive about killing leftovers.
JOB_SCRIPT=$(mktemp -t meta-agent-slurm.XXXXXX.sh)
trap 'rm -f "$JOB_SCRIPT"' EXIT
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo 'echo "[submit.sh] host=$(hostname) jobid=$SLURM_JOB_ID"'
  printf 'CMD=%q\n' "$CMD"
  echo 'echo "[submit.sh] cmd=$CMD"'
  echo 'srun --kill-on-bad-exit=1 bash -c "$CMD"'
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

SBATCH_ARGS=(
  --job-name="$JOB_NAME"
  --partition="$PART"
  --time="$TIME"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --nodes=1
  --ntasks=1
  --output="$LOG_DIR/%j.out"
  --error="$LOG_DIR/%j.err"
  --chdir="$REPO_ROOT"
)
if [[ -n "$GRES" ]]; then
  SBATCH_ARGS+=("--gres=$GRES")
fi

sbatch "${SBATCH_ARGS[@]}" "$JOB_SCRIPT"
