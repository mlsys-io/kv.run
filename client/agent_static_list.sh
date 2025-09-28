#!/bin/bash

# Agent Static List Client for MLOC
# Tests agent execution with predefined task lists
# Usage: ./client/agent_static_list.sh

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/agent_static_list.yaml | jq