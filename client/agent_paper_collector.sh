#!/bin/bash

# Academic Paper Collector Agent Client for MLOC
# Usage: ./client/paper_collector.sh

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/agent_paper_collector.yaml | jq