export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/sft_llama_3b.yaml | jq
