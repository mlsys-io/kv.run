#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Redis defaults (override via env)
# -----------------------------
: "${REDIS_HOST:=redis}"
: "${REDIS_PORT:=6379}"
: "${REDIS_DB:=0}"

# Build REDIS_URL from parts if not provided
if [[ -z "${REDIS_URL:-}" ]]; then
  export REDIS_URL="redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
fi

# -----------------------------
# Optional worker ID (leave empty to let app decide)
# -----------------------------
export WORKER_ID="${WORKER_ID:-}"

# -----------------------------
# Print key environment values
# -----------------------------
echo "Starting worker..."
echo "REDIS_URL=${REDIS_URL}"
echo "TASK_TOPIC=${TASK_TOPIC:-unset}"
echo "RESULTS_DIR=${RESULTS_DIR:-./results}"
echo "HEARTBEAT_INTERVAL_SEC=${HEARTBEAT_INTERVAL_SEC:-30}"

# -----------------------------
# Run worker
# -----------------------------
exec python /app/main.py
