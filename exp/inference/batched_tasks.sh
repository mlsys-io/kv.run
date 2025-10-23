#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <count>" >&2
  echo "Example: $0 5" >&2
  exit 1
fi

COUNT="$1"
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -le 0 ]]; then
  echo "Count must be a positive integer; got '$COUNT'" >&2
  exit 1
fi

HOST_URL="${HOST_URL:-http://localhost:8000}"
TASK_ENDPOINT="${TASK_ENDPOINT:-/api/v1/tasks}"
TOKEN="${TOKEN:-dev-token}"

mapfile -t TASK_FILES < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
TOTAL=${#TASK_FILES[@]}

if [[ "$TOTAL" -eq 0 ]]; then
  echo "No YAML task files found in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ "$COUNT" -gt "$TOTAL" ]]; then
  echo "Requested $COUNT tasks but only $TOTAL available; submitting $TOTAL instead." >&2
  COUNT="$TOTAL"
fi

declare -a SELECTED_TASKS=()
if [[ "$COUNT" -ge "$TOTAL" ]]; then
  SELECTED_TASKS=("${TASK_FILES[@]}")
else
  mapfile -t SELECTED_TASKS < <(shuf -e -- "${TASK_FILES[@]}" | head -n "$COUNT")
fi

SELECT_COUNT=${#SELECTED_TASKS[@]}

echo "Submitting $SELECT_COUNT task(s) to ${HOST_URL}${TASK_ENDPOINT}"

HAS_JQ=0
if command -v jq >/dev/null 2>&1; then
  HAS_JQ=1
fi

for ((i = 0; i < SELECT_COUNT; i++)); do
  FILE="${SELECTED_TASKS[$i]}"
  BASENAME="$(basename "$FILE")"
  echo "[$((i + 1))/$SELECT_COUNT] Submitting $BASENAME"

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
done
