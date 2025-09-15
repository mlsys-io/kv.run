#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation - Retrieval Only) Executor

This executor queries a Qdrant collection using server-side embeddings.
Given a user query, it returns the top matching payload paragraphs.

Spec schema (YAML):
  apiVersion: mloc/v1
  kind: RetrievalTask
  metadata:
    name: my-rag
  spec:
    taskType: rag
    resources: { ... }
    qdrant:
      url: "https://<host>:6333"
      api_key: "<key>"          # optional if not needed
      collection: "demo_collection"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"
    search:
      top_k: 5
    query: "What is Qdrant?"
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from worker.executors.base_executor import Executor, ExecutionError


logger = logging.getLogger("worker.rag")


class RAGExecutor(Executor):
    name = "rag"

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        start_ts = time.time()
        spec = (task or {}).get("spec") or {}

        qdrant_cfg = spec.get("qdrant", {})
        url = qdrant_cfg.get("url")
        api_key = qdrant_cfg.get("api_key")
        collection = qdrant_cfg.get("collection")

        embedding_cfg = spec.get("embedding", {})
        model_name = embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        search_cfg = spec.get("search", {})
        top_k = int(search_cfg.get("top_k", 5))

        query_text = spec.get("query")

        # Basic validation
        if not url:
            raise ExecutionError("Missing spec.qdrant.url")
        if not collection:
            raise ExecutionError("Missing spec.qdrant.collection")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ExecutionError("Missing or empty spec.query")

        logger.info("Connecting Qdrant url=%s collection=%s", url, collection)
        client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)

        try:
            logger.info("Querying top_k=%d using model=%s", top_k, model_name)
            res = client.query_points(
                collection_name=collection,
                query=models.Document(text=query_text, model=model_name),
                limit=top_k,
            )
            points = getattr(res, "points", []) or []
        except Exception as e:
            logger.exception("Qdrant query failed: %s", e)
            raise ExecutionError(f"Qdrant query failed: {e}")

        items: List[Dict[str, Any]] = []
        for p in points:
            items.append({
                "id": getattr(p, "id", None),
                "score": getattr(p, "score", None),
                "payload": getattr(p, "payload", None),
            })

        # Compose response
        out: Dict[str, Any] = {
            "ok": True,
            "executor": self.name,
            "qdrant": {"collection": collection, "url": url},
            "query": query_text,
            "items": items,
            "usage": {
                "latency_sec": round(time.time() - start_ts, 4),
                "num_results": len(items),
            },
        }

        # Persist outputs
        self.save_json(out_dir / "responses.json", out)
        logger.info("RAG query completed results=%d", len(items))
        return out


