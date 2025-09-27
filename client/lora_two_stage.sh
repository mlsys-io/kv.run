#!/usr/bin/env bash
# Submit the two-stage LoRA SFT pipeline.
set -eu

: "${TOKEN:=dev-token}"
ORCH_URL="${ORCH_URL:-http://localhost:8080}"

echo "Submitting two-stage LoRA SFT task to ${ORCH_URL}" >&2
curl -sS -X POST "${ORCH_URL}/api/v1/tasks" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/lora_sft_llama_two_stage.yaml | jq
