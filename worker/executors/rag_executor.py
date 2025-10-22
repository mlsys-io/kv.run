#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation - Retrieval Only) Executor

This executor queries a Qdrant collection using server-side embeddings.
Supports single or multiple queries.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from .base_executor import Executor, ExecutionError
from .graph_templates import build_prompts_from_graph_template
from datasets import load_dataset
from urllib.parse import urlparse


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

        # -------- Prepare queries (dataset | list | graph_template | single query) --------
        queries: List[str] = []
        data_cfg = spec.get("data") or {}
        dtype = data_cfg.get("type") if isinstance(data_cfg, dict) else None
        if dtype == "dataset":
            data_url = data_cfg.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'.")
            name = data_cfg.get("name", None)
            split = data_cfg.get("split", "train")
            shuffle = bool(data_cfg.get("shuffle", False))
            column = data_cfg.get("column", "text")

            trust_remote_code = data_cfg.get("trust_remote_code")
            revision = data_cfg.get("revision")
            dataset_kwargs = {
                "name": name,
                "split": split,
                "revision": revision,
            }
            if trust_remote_code is not None:
                dataset_kwargs["trust_remote_code"] = bool(trust_remote_code)
            dataset = load_dataset(
                data_url,
                **{k: v for k, v in dataset_kwargs.items() if v is not None},
            )
            if shuffle:
                seed = int(data_cfg.get("seed", 42))
                buffer_size = data_cfg.get("buffer_size", None)
                dataset = dataset.shuffle(seed=seed) if buffer_size is None else dataset.shuffle(seed=seed, buffer_size=int(buffer_size))

            if column not in dataset.column_names:
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. Available: {dataset.column_names}"
                )
            queries = [str(x) for x in dataset[column]]
        elif dtype == "list":
            items = data_cfg.get("items", [])
            if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
                raise ExecutionError("spec.data.items must be a list of strings for type == 'list'.")
            queries = [s for s in items]
        elif dtype == "graph_template":
            # Build queries from upstream results using the graph template
            queries = build_prompts_from_graph_template(data_cfg, spec)
        else:
            # Backward compatibility: spec.query as a single string
            query_text = spec.get("query")
            if isinstance(query_text, str) and query_text.strip():
                queries = [query_text]
            else:
                raise ExecutionError("Missing input queries: provide spec.query or spec.data")

        # Basic validation
        if not url:
            raise ExecutionError("Missing spec.qdrant.url")
        if not collection:
            raise ExecutionError("Missing spec.qdrant.collection")
        if not queries:
            raise ExecutionError("No queries prepared. Check spec.query or spec.data configuration.")

        logger.info("Connecting Qdrant url=%s collection=%s", url, collection)
        client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)

        results_per_query: List[Dict[str, Any]] = []
        total_items = 0
        for i, q in enumerate(queries):
            try:
                logger.info("Querying top_k=%d using model=%s", top_k, model_name)
                res = client.query_points(
                    collection_name=collection,
                    query=models.Document(text=str(q), model=model_name),
                    limit=top_k,
                )
                points = getattr(res, "points", []) or []
            except Exception as e:
                has_api_key = bool(api_key)
                scheme = host = ""
                try:
                    parsed = urlparse(url or "")
                    scheme = parsed.scheme
                    host = parsed.netloc
                except Exception:
                    pass
                err_msg = f"Qdrant query failed: {e}"
                ctx_msg = f"url={url} (scheme={scheme}, host={host}), collection={collection}, has_api_key={has_api_key}"
                logger.error("%s; context: %s; exception_type=%s", err_msg, ctx_msg, type(e).__name__)
                print(f"[RAGExecutor] {err_msg}; context: {ctx_msg}; exception_type={type(e).__name__}")
                raise ExecutionError(f"{err_msg}. {ctx_msg}")

            items: List[Dict[str, Any]] = []
            for p in points:
                items.append({
                    "id": getattr(p, "id", None),
                    "score": getattr(p, "score", None),
                    "payload": getattr(p, "payload", None),
                })
            total_items += len(items)
            results_per_query.append({
                "index": i,
                "query": str(q),
                "items": items,
            })

        # Compose response
        out: Dict[str, Any] = {
            "ok": True,
            "executor": self.name,
            "qdrant": {"collection": collection, "url": url},
            "embedding": {"model": model_name},
            "search": {"top_k": top_k},
            "queries": results_per_query,
            "usage": {
                "latency_sec": round(time.time() - start_ts, 4),
                "num_queries": len(queries),
                "total_results": total_items,
            },
        }

        # Persist outputs
        self.save_json(out_dir / "responses.json", out)
        logger.info("RAG query completed queries=%d total_results=%d", len(queries), total_items)
        return out
