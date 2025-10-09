#!/bin/bash
# PPO Training Task Submission Script
#
# Prerequisites:
# - Orchestrator must be running with REDIS_URL configured
# - Worker must have REDIS_URL set and be subscribed to tasks.ppo
# - Worker must have PPO dependencies installed (trl, transformers, torch, datasets)

export TOKEN="dev-token"

echo "Submitting PPO training task..."

curl -sS -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/ppo_training_mistral.yaml | jq

echo ""
echo "Task submitted! Check the worker logs for training progress."
echo "Results will be saved to RESULTS_DIR/<task_id>/responses.json"