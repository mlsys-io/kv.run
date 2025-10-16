from __future__ import annotations

import copy
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis

from .task_models import TaskStatus
from .task_runtime import TaskRuntime
from .results import result_file_path
from .utils import now_iso
from .task_metadata import extract_model_dataset_names
from .worker_registry import (
    Worker,
    idle_satisfying_pool,
    update_worker_status,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker

_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


class StageReferenceNotReady(Exception):
    """Raised when a task references a stage whose artifacts are not yet available."""


class Dispatcher:
    """Handles FCFS task dispatching via Redis pub/sub."""

    def __init__(
        self,
        runtime: TaskRuntime,
        redis_client: redis.Redis,
        results_dir: Path,
        *,
        logger,
        worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
    ) -> None:
        self._runtime = runtime
        self._redis = redis_client
        self._logger = logger
        self._results_dir = Path(results_dir)
        self._worker_selection_strategy = worker_selection_strategy

    def dispatch_once(self, task_id: str) -> bool:
        """Dispatch a single task if possible; requeue when no worker."""
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        task_payload = copy.deepcopy(record.parsed)
        model_names, dataset_names = extract_model_dataset_names(task_payload)
        pool = idle_satisfying_pool(self._redis, task_payload)
        if not pool:
            self._logger.debug("No idle worker available for %s; requeueing", task_id)
            self._runtime.mark_pending(task_id)
            self._runtime.requeue(task_id)
            return False

        preferred_pool = self._cached_worker_candidates(pool, model_names, dataset_names)
        candidate_pool = preferred_pool or pool
        if preferred_pool:
            self._logger.debug(
                "Preferring cached worker candidates for %s (models=%s datasets=%s)",
                task_id,
                model_names,
                dataset_names,
            )

        worker = select_worker(candidate_pool, self._worker_selection_strategy, logger=self._logger)
        if not worker and preferred_pool:
            worker = select_worker(pool, self._worker_selection_strategy, logger=self._logger)
        if not worker:
            self._logger.debug("No suitable worker selected for %s; requeueing", task_id)
            self._runtime.mark_pending(task_id)
            self._runtime.requeue(task_id)
            return False

        message = {
            "task_id": task_id,
            "task": task_payload,
            "task_type": record.task_type,
            "assigned_worker": worker.worker_id,
            "dispatched_at": now_iso(),
            "parent_task_id": None,
        }

        try:
            rendered_task = self._resolve_stage_references(task_id, task_payload, record)
        except StageReferenceNotReady as exc:
            self._logger.debug("Task %s waiting on stage artifacts: %s", task_id, exc)
            self._runtime.requeue(task_id)
            return False
        except Exception as exc:
            self._logger.error("Failed to resolve stage references for %s: %s", task_id, exc)
            self._runtime.mark_failed(task_id, None, {"error": str(exc)}, now_iso(), error=str(exc))
            return True

        message["task"] = rendered_task

        try:
            receivers = int(self._redis.publish("tasks", json.dumps(message, ensure_ascii=False)))
        except Exception as exc:
            self._logger.warning("Failed to publish task %s: %s", task_id, exc)
            self._runtime.mark_pending(task_id)
            self._runtime.requeue(task_id, front=True)
            return False

        if receivers <= 0:
            self._logger.info("No subscribers on tasks channel; delaying task %s", task_id)
            self._runtime.mark_pending(task_id)
            self._runtime.requeue(task_id, front=True)
            return False

        self._runtime.mark_dispatched(task_id, worker.worker_id)
        try:
            update_worker_status(self._redis, worker.worker_id, "RUNNING")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Failed to update worker %s status: %s", worker.worker_id, exc)
        return True

    def dispatch_loop(self, stop_event, poll_interval: float = 1.0) -> None:
        """Continuously dispatch ready tasks until stop_event is set."""
        while not stop_event.is_set():
            task_id = self._runtime.next_ready(stop_event, timeout=poll_interval)
            if not task_id:
                continue
            try:
                success = self.dispatch_once(task_id)
                if not success:
                    time.sleep(0.5)
            except Exception as exc:  # noqa: broad-except
                self._logger.exception("Dispatch loop error for %s: %s", task_id, exc)
                self._runtime.mark_pending(task_id)
                self._runtime.requeue(task_id, front=True)
                time.sleep(1.0)

    def _cached_worker_candidates(
        self,
        pool: List[Worker],
        model_names: List[str],
        dataset_names: List[str],
    ) -> List[Worker]:
        if not pool or (not model_names and not dataset_names):
            return []

        def _score(worker: Worker) -> int:
            score = 0
            cached_models = {m.lower() for m in (worker.cached_models or []) if isinstance(m, str)}
            cached_datasets = {d.lower() for d in (worker.cached_datasets or []) if isinstance(d, str)}
            if model_names:
                score += sum(1 for name in model_names if name.lower() in cached_models)
            if dataset_names:
                score += sum(1 for name in dataset_names if name.lower() in cached_datasets)
            return score

        scored: List[Tuple[Worker, int]] = []
        for worker in pool:
            value = _score(worker)
            if value > 0:
                scored.append((worker, value))

        if not scored:
            return []

        max_score = max(score for _, score in scored)
        return [worker for worker, score in scored if score == max_score]

    # ------------------------------------------------------------------ #
    # Stage reference handling
    # ------------------------------------------------------------------ #

    def _resolve_stage_references(self, task_id: str, payload: Dict[str, Any], record) -> Dict[str, Any]:
        """
        Replace ${stage.result.foo} references with values from completed stage results
        and attach upstream result payloads for graph-aware executors.
        """
        if not isinstance(payload, dict):
            return payload

        context = self._build_stage_context(record)
        has_placeholders = self._contains_placeholder(payload)

        resolved = payload
        if has_placeholders:
            if not context:
                return payload

            def _transform(value):
                if isinstance(value, dict):
                    return {k: _transform(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_transform(item) for item in value]
                if isinstance(value, str):
                    exact = _PLACEHOLDER_PATTERN.fullmatch(value)
                    if exact:
                        return self._resolve_reference(exact.group(1), context)
                    return _PLACEHOLDER_PATTERN.sub(
                        lambda m: str(self._resolve_reference(m.group(1), context)),
                        value,
                    )
                return value

            resolved = _transform(payload)
        else:
            # Ensure we copy payload only when we need to attach upstream context
            resolved = copy.deepcopy(payload)

        upstream_results = self._collect_upstream_results(context, task_id) if context else {}
        if upstream_results:
            spec = resolved.setdefault("spec", {})
            existing = spec.get("_upstreamResults")
            merged = {}
            if isinstance(existing, dict):
                merged.update(copy.deepcopy(existing))
            merged.update(upstream_results)
            spec["_upstreamResults"] = merged

        return resolved

    def _contains_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(_PLACEHOLDER_PATTERN.search(value))
        if isinstance(value, dict):
            return any(self._contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    def _build_stage_context(self, record) -> Dict[str, Any]:
        """Collect completed stage records keyed by stage name for the same submission."""
        if record.graph_node_name is None and not self._contains_placeholder(record.parsed):
            return {}

        base_yaml = record.raw_yaml
        context: Dict[str, Any] = {}
        for other in self._runtime.tasks.values():
            if other.raw_yaml != base_yaml:
                continue
            if not other.graph_node_name:
                continue
            context[other.graph_node_name] = other
        return context

    def _resolve_reference(self, expr: str, context: Dict[str, Any]) -> Any:
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty stage reference")
        if "." not in expr:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_name, path = expr.split(".", 1)
        stage_name = stage_name.strip()
        if not stage_name:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_record = context.get(stage_name)
        if not stage_record:
            raise ValueError(f"Unknown stage reference '{stage_name}'")
        if path == "task_id":
            return stage_record.task_id
        if stage_record.status != "DONE":
            raise StageReferenceNotReady(f"Stage '{stage_name}' has not completed")
        data = self._load_stage_result(stage_record.task_id)
        value = self._dig_path(data, path.split("."))
        if value is None:
            raise ValueError(f"Missing value for reference '{expr}'")
        return value

    def _collect_upstream_results(self, context: Dict[str, Any], current_task_id: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name, record in context.items():
            if not record or record.task_id == current_task_id:
                continue
            if record.status != TaskStatus.DONE:
                continue
            try:
                data = self._load_stage_result(record.task_id)
            except StageReferenceNotReady as exc:
                raise exc
            except Exception as exc:
                self._logger.debug(
                    "Failed to load upstream result for %s (%s): %s",
                    name,
                    record.task_id,
                    exc,
                )
                continue
            results[name] = data
        return results

    def _load_stage_result(self, stage_task_id: str) -> Dict[str, Any]:
        path = result_file_path(self._results_dir, stage_task_id)
        if not path.exists():
            raise StageReferenceNotReady(f"Result for task {stage_task_id} not found at {path}")
        content = json.loads(path.read_text(encoding="utf-8"))
        return content

    def _dig_path(self, data: Any, parts: List[str]) -> Any:
        current = data
        for part in parts:
            part = part.strip()
            if part == "":
                continue
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
                continue
            if isinstance(current, list):
                try:
                    idx = int(part)
                except ValueError as exc:
                    raise ValueError(f"List index must be integer in reference path, got '{part}'") from exc
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            return None
        return current
