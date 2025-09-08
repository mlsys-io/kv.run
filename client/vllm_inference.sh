# Orchestrator must be running with REDIS_URL configured
# Worker must have REDIS_URL and ORCHESTRATOR_URL set and be subscribed to tasks.inference

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/inference_vllm_mistral.yaml | jq
