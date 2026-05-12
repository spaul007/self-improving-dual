#!/usr/bin/env bash
# Same idea as simple_client.py but using nothing but bash + curl.
# Useful when you want to confirm a server is alive from a shell or
# when integrating from a non-Python codebase.
#
# Usage:
#   # Auto-discover the URL via health.py and ask one question:
#   bash curl_example.sh "Why is the sky blue?"
#
#   # Or pass the base URL explicitly:
#   BASE_URL=http://node-7:8000/v1 bash curl_example.sh "Why is the sky blue?"

set -euo pipefail

PROMPT="${1:-In one sentence: why is the sky blue?}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HEALTH_PY="$REPO_ROOT/model_server/health.py"

if [[ -z "${BASE_URL:-}" ]]; then
  if [[ -x "$HEALTH_PY" || -f "$HEALTH_PY" ]]; then
    BASE_URL="$(python3 "$HEALTH_PY" --print-base-url || true)"
  fi
fi

if [[ -z "${BASE_URL:-}" ]]; then
  echo "Could not discover a server URL. Set BASE_URL or start the server." >&2
  exit 2
fi

# Pick a model id from the discovery file when MODEL isn't set.
DISCOVERY="${MODEL_SERVER_DISCOVERY_PATH:-/groups/AIC-MV/n.tzou/model_server/endpoint.json}"
if [[ -z "${MODEL:-}" && -f "$DISCOVERY" ]]; then
  MODEL="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1])).get("model", "local"))
' "$DISCOVERY")"
fi
MODEL="${MODEL:-local}"

echo "# base_url:  $BASE_URL"
echo "# model:     $MODEL"
echo "# endpoint:  /v1/chat/completions"
echo "# prompt:    $PROMPT"
echo

curl -sS "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d "$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": 512,
}))
' "$MODEL" "$PROMPT")" \
  | python3 -m json.tool
