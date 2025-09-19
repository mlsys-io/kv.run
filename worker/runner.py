# worker/runner.py
"""Pub/Sub runner that executes assigned tasks using pluggable executors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class Runner:
    def __init__(self, lifecycle, rds, topic: str, results_dir: Path, executors: Dict[str, Any], default_executor: Any, logger: Any):
        self.lifecycle = lifecycle
        self.redis = rds
        self.topic = topic
        self.results_dir = results_dir
        self.executors = executors
        self.logger = logger
        self.default_executor = default_executor

    def _resolve_output_dir(self, task_id: str, task: Dict[str, Any]) -> Path:
        """Pick the destination directory for task outputs.

        Priority: spec.output.destination.path -> default RESULTS_DIR.
        Relative paths under spec are resolved against the configured RESULTS_DIR
        so users can safely provide per-task subfolders.
        """
        spec = (task or {}).get("spec") or {}
        output_cfg = spec.get("output") or {}
        dest = (output_cfg.get("destination") or {})
        dest_type = str(dest.get("type") or "local").lower()
        dest_path = dest.get("path")

        if dest_type == "local" and dest_path:
            chosen = Path(dest_path)
            if not chosen.is_absolute():
                chosen = self.results_dir / chosen
        else:
            chosen = self.results_dir / task_id

        # Ensure each task still gets an isolated directory
        if chosen.name != task_id:
            chosen = chosen / task_id
        return chosen

    def _write_results(self, task_id: str, task: Dict[str, Any], out_dir: Path, result: Optional[Dict[str, Any]]):
        if result is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        self._maybe_emit_http(task_id, task, result)

    def _maybe_emit_http(self, task_id: str, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Send task results to an HTTP endpoint when requested by the spec."""
        spec = (task or {}).get("spec") or {}
        output_cfg = spec.get("output") or {}
        destination = output_cfg.get("destination") or {}

        dest_type = str(destination.get("type") or "local").lower()
        if dest_type != "http":
            return

        url = destination.get("url")
        if not url:
            raise RuntimeError("spec.output.destination.url is required when type is 'http'")

        method = str(destination.get("method") or "POST").upper()
        headers = destination.get("headers") or {}
        timeout = float(destination.get("timeoutSec") or 15)

        rworker = getattr(self.lifecycle, "rworker", None)
        payload = {
            "task_id": task_id,
            "result": result,
            "worker_id": getattr(rworker, "worker_id", None),
        }

        try:
            response = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to deliver task {task_id} result to {url}: {exc}") from exc

        if response.status_code >= 400:
            snippet = response.text[:200]
            raise RuntimeError(
                f"HTTP delivery for task {task_id} returned status {response.status_code}: {snippet}"
            )

        self.logger.info("Task %s result delivered to %s (%s)", task_id, url, response.status_code)

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

            out_dir = self._resolve_output_dir(task_id, task)
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
                    out = executor.run(task, out_dir)
                self._write_results(task_id, task, out_dir, out)
                self.lifecycle.set_succeeded(task_id)
                self.logger.info("Task %s completed successfully", task_id)
            except Exception as e:
                self.lifecycle.set_failed(task_id, str(e))
                self.logger.error("Task %s failed: %s", task_id, e)
            finally:
                self.lifecycle.set_idle(task_id)
