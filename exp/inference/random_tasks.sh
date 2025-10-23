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

echo "Submitting $COUNT random task(s) (with replacement) to ${HOST_URL}${TASK_ENDPOINT}"

HAS_JQ=0
if command -v jq >/dev/null 2>&1; then
  HAS_JQ=1
fi

for ((i = 1; i <= COUNT; i++)); do
  # 随机选择一个 YAML 文件（可重复）
  RAND_INDEX=$((RANDOM % TOTAL))
  FILE="${TASK_FILES[$RAND_INDEX]}"
  BASENAME="$(basename "$FILE")"

  echo "[$i/$COUNT] Submitting $BASENAME"

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
