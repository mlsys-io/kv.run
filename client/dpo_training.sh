#!/bin/bash
# DPO Training Script
# Orchestrator must be running with REDIS_URL configured
# Worker must have REDIS_URL and be subscribed to tasks.dpo

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/dpo_training_mistral.yaml | jq