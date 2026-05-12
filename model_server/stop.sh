#!/usr/bin/env bash
# Stop the running model server: read the SLURM job id out of the
# discovery file, scancel it, then verify the discovery file is gone.
# Falls back to force-removing the file if the in-job EXIT trap didn't
# fire (shouldn't happen, but be defensive).
#
# Usage:
#   bash model_server/stop.sh                # single-slot endpoint.json
#   bash model_server/stop.sh --name <name>  # per-model endpoint_<name>.json

set -euo pipefail

SERVER_ROOT="${MODEL_SERVER_ROOT:-/groups/AIC-MV/n.tzou/model_server}"
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --name=*)
      NAME="${1#--name=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--name <discovery-name>]"
      exit 0
      ;;
    *)
      echo "[stop.sh] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$NAME" ]]; then
  DISCOVERY_FILE="$SERVER_ROOT/endpoint_${NAME}.json"
else
  DISCOVERY_FILE="$SERVER_ROOT/endpoint.json"
fi

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
