#!/bin/bash
# DAG inference example using graph-templated synthesis
# Requires orchestrator on :8000 and workers subscribed to tasks

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/dag_inference_example.yaml | jq
