#!/usr/bin/env bash
set -euo pipefail

ORCH_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8090}"
TOKEN="${ORCHESTRATOR_TOKEN:-}"

CMD=(run --scenario echo-local --orchestrator-url "$ORCH_URL" --fetch-result)
if [[ -n "$TOKEN" ]]; then
  CMD+=(--token "$TOKEN")
fi

python scripts/worker_validate.py "${CMD[@]}"
