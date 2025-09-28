#!/bin/bash

# Agent Dataset Batch Client for MLOC
# Tests agent execution with HuggingFace dataset integration
# Usage: ./client/agent_dataset_batch.sh

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/agent_dataset_batch.yaml | jq