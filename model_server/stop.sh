#!/usr/bin/env bash
# Stop the running model server: read the SLURM job id out of the
# discovery file, scancel it, then verify the discovery file is gone.
# Falls back to force-removing the file if the in-job EXIT trap didn't
# fire (shouldn't happen, but be defensive).
#
# Usage:
#   bash model_server/stop.sh

set -euo pipefail

SERVER_ROOT="${MODEL_SERVER_ROOT:-/groups/AIC-MV/n.tzou/model_server}"
DISCOVERY_FILE="$SERVER_ROOT/endpoint.json"

if [[ ! -f "$DISCOVERY_FILE" ]]; then
  echo "[stop.sh] no discovery file at $DISCOVERY_FILE — nothing to stop" >&2
  exit 1
fi

JOB_ID="$(python3 -c '
import json, sys
with open(sys.argv[1]) as fh:
    print(json.load(fh)["slurm_job_id"])
' "$DISCOVERY_FILE")"

echo "[stop.sh] cancelling SLURM job $JOB_ID"
scancel "$JOB_ID"

# Give the in-job EXIT trap up to 10s to remove the discovery file.
for _ in $(seq 1 10); do
  if [[ ! -f "$DISCOVERY_FILE" ]]; then
    echo "[stop.sh] discovery file removed by job trap"
    exit 0
  fi
  sleep 1
done

echo "[stop.sh] WARNING: discovery file still present after scancel; force-removing" >&2
rm -f "$DISCOVERY_FILE"
