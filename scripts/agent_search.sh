#!/bin/bash

# Agent Task Client for MLOC
# Usage: ./client/agent.sh

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/agent_query_search.yaml | jq