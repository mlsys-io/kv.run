# worker/runner.py
"""Pub/Sub runner that executes assigned tasks using pluggable executors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Runner:
    def __init__(self, lifecycle, rds, topic: str, results_dir: Path, executors: Dict[str, Any], default_executor: Any, logger: Any):
        self.lifecycle = lifecycle
        self.redis = rds
        self.topic = topic
        self.results_dir = results_dir
        self.executors = executors
        self.logger = logger
        self.default_executor = default_executor

    def _write_results(self, task_id: str, result: Optional[Dict[str, Any]]):
        if result is None:
            return
        out_dir = self.results_dir / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def start(self):
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(self.topic)
        for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue

            raw = msg.get("data")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if data.get("assigned_worker") != self.lifecycle.rworker.worker_id:
                continue

            task_id = str(data.get("task_id"))
            task = data.get("task") or {}
            task_type = (task.get("spec") or {}).get("taskType")

            self.lifecycle.set_running(task_id)
            try:
                if task_type == "inference":
                    enforce_cpu = bool((task.get("spec") or {}).get("enforce_cpu", False))
                    if enforce_cpu:
                        executor = self.default_executor
                    else:
                        executor = self.executors.get("vllm", self.default_executor)
                else:
                    executor = self.executors.get(task_type, self.default_executor)

                out = None
                if executor:
                    out = executor.run(task, self.results_dir / task_id)
                self._write_results(task_id, out)
                self.lifecycle.set_succeeded(task_id)
                self.logger.info("Task %s completed successfully", task_id)
            except Exception as e:
                self.lifecycle.set_failed(task_id, str(e))
                self.logger.error("Task %s failed: %s", task_id, e)
            finally:
                self.lifecycle.set_idle(task_id)

