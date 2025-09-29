#!/usr/bin/env bash
# Submit the two-stage SFT pipeline.
set -eu

: "${TOKEN:=dev-token}"
ORCH_URL="${ORCH_URL:-http://localhost:8000}"

echo "Submitting two-stage SFT task to ${ORCH_URL}" >&2
curl -sS -X POST "${ORCH_URL}/api/v1/tasks" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/sft_llama_two_stage.yaml | jq
