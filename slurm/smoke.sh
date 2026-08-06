#!/usr/bin/env bash
# Submit the unit smoke suite as a SLURM job. No API key needed.
#
# Usage:
#   slurm/smoke.sh                # runs the default tests.test_smoke
#   slurm/smoke.sh tests.test_travel_smoke
#
# Same env-var knobs as slurm/submit.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-tests.test_smoke}"

# Smoke tests are quick; shrink the default time/memory.
export SLURM_TIME="${SLURM_TIME:-00:30:00}"
export SLURM_CPUS="${SLURM_CPUS:-2}"
export SLURM_MEM="${SLURM_MEM:-4G}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-meta-agent-smoke}"

exec "$REPO_ROOT/slurm/submit.sh" "PYTHONPATH=. python3 -m unittest -v $SUITE"
