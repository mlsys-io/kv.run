# example command to create a LoRA SFT training task
export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/lora_sft_llama.yaml | jq
