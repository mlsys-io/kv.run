#!/bin/bash
# RAG (Qdrant) Query Script
# Orchestrator must be running and worker subscribed to tasks.rag

export TOKEN="dev-token"
curl -sS -X POST "http://localhost:8080/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/rag_query_qdrant.yaml | jq


