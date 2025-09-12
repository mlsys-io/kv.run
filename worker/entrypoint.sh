#!/usr/bin/env bash
set -euo pipefail

# Validate required env vars
if [[ -z "${REDIS_URL:-}" ]]; then
  echo "ERROR: REDIS_URL is required" >&2
  exit 1
fi

# Auto-generate worker ID if not provided
export WORKER_ID="${WORKER_ID:-}"

# Print key environment values
echo "Starting worker..."
echo "REDIS_URL=$REDIS_URL"
echo "TASK_TOPICS=${TASK_TOPICS:-unset}"
echo "RESULTS_DIR=${RESULTS_DIR:-./results}"
echo "HEARTBEAT_INTERVAL_SEC=${HEARTBEAT_INTERVAL_SEC:-30}"

# Run worker
exec python /app/main.py
