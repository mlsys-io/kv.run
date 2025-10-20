#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <count> <rate_per_sec>" >&2
  echo "Example: $0 10 0.5  # submit 10 tasks at 0.5 requests/sec" >&2
  exit 1
fi

COUNT="$1"
RATE="$2"

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -le 0 ]]; then
  echo "Count must be a positive integer; got '$COUNT'" >&2
  exit 1
fi

if ! python3 - "$RATE" <<'PY' >/dev/null 2>&1
import sys
rate = float(sys.argv[1])
sys.exit(0 if rate > 0 else 1)
PY
then
  echo "Rate must be a positive number; got '$RATE'" >&2
  exit 1
fi

SLEEP_INTERVAL="$(python3 - "$RATE" <<'PY'
import sys
rate = float(sys.argv[1])
print(1.0 / rate)
PY
)"

HOST_URL="${HOST_URL:-http://localhost:8000}"
TASK_ENDPOINT="${TASK_ENDPOINT:-/api/v1/tasks}"
TOKEN="${TOKEN:-dev-token}"

mapfile -t TASK_FILES < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
TOTAL=${#TASK_FILES[@]}

if [[ "$TOTAL" -eq 0 ]]; then
  echo "No YAML task files found in $SCRIPT_DIR" >&2
  exit 1
fi

echo "Randomly submitting $COUNT task(s) to ${HOST_URL}${TASK_ENDPOINT} at ~${RATE}/sec"

HAS_JQ=0
if command -v jq >/dev/null 2>&1; then
  HAS_JQ=1
fi

for ((i = 0; i < COUNT; i++)); do
  index=$((RANDOM % TOTAL))
  FILE="${TASK_FILES[$index]}"
  BASENAME="$(basename "$FILE")"
  echo "[$((i + 1))/$COUNT] Submitting $BASENAME"

  if [[ "$HAS_JQ" -eq 1 ]]; then
    curl -sS -X POST "${HOST_URL}${TASK_ENDPOINT}" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: text/yaml" \
      --data-binary @"$FILE" | jq '.'
  else
    curl -sS -X POST "${HOST_URL}${TASK_ENDPOINT}" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: text/yaml" \
      --data-binary @"$FILE"
    echo
  fi

  if [[ $((i + 1)) -lt $COUNT ]]; then
    sleep "$SLEEP_INTERVAL"
  fi
done
