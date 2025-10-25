from __future__ import annotations

import copy
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import redis

from .event_schema import TaskEvent
from .task_models import TaskStatus
from .task_runtime import TaskRuntime
from .results import result_file_path
from .utils import now_iso
from .task_metadata import extract_model_dataset_names
from .worker_registry import (
    Worker,
    get_worker_from_redis,
    idle_satisfying_pool,
    is_stale_by_redis,
    update_worker_status,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker

if TYPE_CHECKING:
    from .elastic import ElasticCoordinator

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
        enable_context_reuse: bool = True,
        enable_task_merge: bool = True,
        task_merge_max_batch_size: int = 4,
        elastic_coordinator: Optional["ElasticCoordinator"] = None,
        reuse_cache_ttl_sec: int = 3600,
        lambda_config: Optional[Dict[str, float]] = None,
        selection_jitter_epsilon: float = 1e-3,
        enable_stage_weight_stickiness: bool = False,
        metrics_recorder=None,
    ) -> None:
        self._runtime = runtime
        self._redis = redis_client
        self._logger = logger
        self._results_dir = Path(results_dir)
        self._worker_selection_strategy = worker_selection_strategy
        self._context_reuse_enabled = enable_context_reuse
        self._task_merge_enabled = enable_task_merge
        self._task_merge_max_batch_size = max(1, task_merge_max_batch_size)
        self._elastic_coordinator = elastic_coordinator
        self._cache_ttl_sec = max(0, int(reuse_cache_ttl_sec))
        self._lambda_config = lambda_config or {}
        self._selection_jitter = max(0.0, float(selection_jitter_epsilon))
        self._stage_weight_stickiness_enabled = bool(enable_stage_weight_stickiness)
        self._metrics = metrics_recorder
        self._weight_reference_hints: Tuple[str, ...] = (
            "checkpoint",
            "weight",
            "weights",
            "model",
            "adapter",
            "lora",
            "load",
            "artifact",
        )

    def dispatch_once(self, task_id: str) -> bool:
        """Dispatch a single task if possible; requeue when no worker."""
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        task_payload = copy.deepcopy(record.parsed)
        merged_children: List[str] = []
        worker = None
        selection_info: Dict[str, Any] = {}

        model_names, dataset_names = extract_model_dataset_names(task_payload)
        task_category = (record.category or "other").lower() if hasattr(record, "category") else "other"
        task_age = None
        if getattr(record, "last_queue_ts", None) is not None:
            task_age = max(0.0, time.time() - record.last_queue_ts)
        disabled_workers = None
        if self._elastic_coordinator and self._elastic_coordinator.enabled:
            disabled_workers = self._elastic_coordinator.disabled_workers()
        pool = idle_satisfying_pool(self._redis, task_payload, disabled_workers=disabled_workers)
        if not pool:
            self._logger.debug("No idle worker available for %s; requeueing", task_id)
            self._requeue_task(task_id, reason="no_idle_worker", count_retry=False)
            return False

        sticky_worker_id: Optional[str] = None
        if self._stage_weight_stickiness_enabled:
            sticky_worker_id = self._preferred_stage_worker(record, task_payload)
            if sticky_worker_id:
                sticky_candidate = next((item for item in pool if item.worker_id == sticky_worker_id), None)
                if sticky_candidate:
                    worker = sticky_candidate
                    selection_info = {
                        "strategy": "stage_weight_affinity",
                        "sticky_worker": sticky_worker_id,
                        "candidate_pool": len(pool),
                        "chosen_metrics": {"score": 1.0},
                    }
                    self._logger.debug(
                        "Using stage-affinity worker %s for %s",
                        sticky_worker_id,
                        task_id,
                    )
                else:
                    if self._should_wait_for_sticky_worker(record, sticky_worker_id):
                        self._logger.debug(
                            "Preferred stage worker %s busy for %s; waiting for availability",
                            sticky_worker_id,
                            task_id,
                        )
                        self._requeue_task(task_id, reason="sticky_worker_busy")
                        return False
                    else:
                        self._logger.debug(
                            "Preferred stage worker %s unavailable for %s; falling back to normal selection",
                            sticky_worker_id,
                            task_id,
                        )

        preferred_pool: List[Worker] = []
        if worker is None and self._context_reuse_enabled:
            preferred_pool = self._cached_worker_candidates(pool, model_names, dataset_names)
        if worker is None:
            candidate_pool = preferred_pool or pool
            if preferred_pool:
                self._logger.debug(
                    "Preferring cached worker candidates for %s (models=%s datasets=%s)",
                    task_id,
                    model_names,
                    dataset_names,
                )

            worker, selection_info = select_worker(
                candidate_pool,
                self._worker_selection_strategy,
                logger=self._logger,
                task_category=task_category,
                lambda_overrides=self._lambda_config,
                task_id=task_id,
                jitter_epsilon=self._selection_jitter,
                task_age=task_age,
            )
            if not worker and preferred_pool:
                worker, selection_info = select_worker(
                    pool,
                    self._worker_selection_strategy,
                    logger=self._logger,
                    task_category=task_category,
                    lambda_overrides=self._lambda_config,
                    task_id=task_id,
                    jitter_epsilon=self._selection_jitter,
                    task_age=task_age,
                )
        if not worker:
            self._logger.debug("No suitable worker selected for %s; requeueing", task_id)
            self._requeue_task(task_id, reason="no_selection")
            return False

        if self._task_merge_enabled and self._task_merge_max_batch_size > 1:
            merged_children = self._runtime.plan_merge(task_id, self._task_merge_max_batch_size)
            if merged_children:
                self._logger.debug(
                    "Coalesced task %s with siblings %s",
                    task_id,
                    ", ".join(merged_children),
                )

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
            self._requeue_task(task_id, reason="stage_reference_pending", count_retry=False)
            return False
        except Exception as exc:
            self._logger.error("Failed to resolve stage references for %s: %s", task_id, exc)
            self._runtime.release_merge(task_id)
            self._fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        rendered_children: Optional[List[Dict[str, Any]]] = None
        if record.merged_children:
            rendered_children = []
            for child_entry in record.merged_children:
                child_id = str(child_entry.get("task_id") or "").strip()
                if not child_id:
                    rendered_children.append(copy.deepcopy(child_entry))
                    continue
                child_record = self._runtime.get_record(child_id)
                if not child_record:
                    rendered_children.append(copy.deepcopy(child_entry))
                    continue
                child_payload = copy.deepcopy(child_entry)
                child_payload.pop("task_id", None)
                if not child_payload.get("spec"):
                    rendered_children.append(copy.deepcopy(child_entry))
                    continue
                try:
                    resolved_child = self._resolve_stage_references(child_id, child_payload, child_record)
                except StageReferenceNotReady as exc:
                    self._logger.debug(
                        "Merged child %s waiting on stage artifacts: %s",
                        child_id,
                        exc,
                    )
                    self._runtime.release_merge(task_id)
                    self._requeue_task(task_id, reason="stage_reference_pending", count_retry=False)
                    return False
                except Exception as exc:
                    self._logger.error(
                        "Failed to resolve stage references for merged child %s: %s",
                        child_id,
                        exc,
                    )
                    self._runtime.release_merge(task_id)
                    self._fail_task(
                        task_id,
                        f"Failed to resolve merged child {child_id}: {exc}",
                        payload={"error": str(exc)},
                    )
                    return True
                rendered_entry: Dict[str, Any] = {"task_id": child_id}
                rendered_entry.update(resolved_child)
                rendered_children.append(rendered_entry)

        message["task"] = rendered_task
        if rendered_children is not None:
            message["merged_children"] = rendered_children
        elif record.merged_children:
            message["merged_children"] = copy.deepcopy(record.merged_children)

        try:
            receivers = int(self._redis.publish("tasks", json.dumps(message, ensure_ascii=False)))
        except Exception as exc:
            self._logger.warning("Failed to publish task %s: %s", task_id, exc)
            self._requeue_task(task_id, reason="publish_failed", front=True)
            return False

        if receivers <= 0:
            self._logger.info("No subscribers on tasks channel; delaying task %s", task_id)
            self._requeue_task(task_id, reason="no_subscribers", front=True)
            return False

        self._runtime.mark_dispatched(task_id, worker.worker_id)
        if merged_children:
            try:
                self._logger.info(
                    "[TaskMerge] parent=%s merged_children=%d -> %s",
                    task_id,
                    len(merged_children),
                    ", ".join(merged_children),
                )
            except Exception:
                pass
        try:
            update_worker_status(self._redis, worker.worker_id, "RUNNING")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Failed to update worker %s status: %s", worker.worker_id, exc)
        try:
            chosen_score = selection_info.get("chosen_metrics", {}).get("score")
            score_display = f"{chosen_score:.4f}" if isinstance(chosen_score, (int, float)) else "n/a"
            self._logger.info(
                "Dispatch %s -> %s (strategy=%s reuse=%d/%d score=%s age=%.2fs)",
                task_id,
                worker.worker_id,
                selection_info.get("strategy"),
                len(preferred_pool),
                len(pool),
                score_display,
                task_age if task_age is not None else -1.0,
            )
        except Exception:
            pass
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
                self._requeue_task(task_id, reason="dispatch_exception", front=True)
                time.sleep(1.0)

    def _requeue_task(
        self,
        task_id: str,
        *,
        reason: str,
        front: bool = False,
        release_merge: bool = True,
        mark_pending: bool = True,
        count_retry: bool = True,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if release_merge:
            self._runtime.release_merge(task_id)

        if mark_pending:
            self._runtime.mark_pending(task_id, increment_retry=count_retry)

        record = self._runtime.get_record(task_id)
        if count_retry and record:
            attempts = record.attempts
            max_attempts = record.max_attempts
            if max_attempts is not None and max_attempts >= 0 and attempts >= max_attempts:
                self._fail_task(
                    task_id,
                    "max_attempts_exceeded",
                    payload={
                        "reason": reason,
                        "attempts": attempts,
                        "max_attempts": max_attempts,
                    },
                    worker_id=record.last_failed_worker or record.assigned_worker,
                )
                return

        self._runtime.requeue(task_id, front=front)
        if count_retry:
            payload = {"reason": reason}
            if extra_payload:
                payload.update(extra_payload)
            self._emit_task_event("TASK_REQUEUED", task_id, payload=payload)

    def _fail_task(
        self,
        task_id: str,
        error_message: str,
        *,
        worker_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        failure_payload = payload.copy() if isinstance(payload, dict) else {}
        if error_message and "error" not in failure_payload:
            failure_payload["error"] = error_message
        impacted, merged_children = self._runtime.mark_failed(
            task_id,
            worker_id,
            failure_payload,
            now_iso(),
            error=error_message,
        )
        self._emit_task_event(
            "TASK_FAILED",
            task_id,
            worker_id=worker_id,
            payload=failure_payload,
            error=error_message,
        )
        for dep_id, reason in impacted:
            dependent_payload = {"dependency_failure": task_id, "error": reason}
            self._emit_task_event(
                "TASK_FAILED",
                dep_id,
                payload=dependent_payload,
                error=reason,
            )
        for child_id in merged_children:
            child_payload = failure_payload.copy()
            child_payload["parent_task_id"] = task_id
            child_payload["dependency_failure"] = task_id
            child_payload["is_child_task"] = True
            self._emit_task_event(
                "TASK_FAILED",
                child_id,
                payload=child_payload,
                error=error_message or "parent_failed",
                is_child=True,
            )

    def _emit_task_event(
        self,
        event_type: str,
        task_id: str,
        *,
        worker_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        is_child: bool = False,
    ) -> None:
        if not self._metrics:
            return
        event = TaskEvent(
            type=event_type,
            task_id=task_id,
            worker_id=worker_id,
            payload=payload or {},
            error=error,
            ts=now_iso(),
        )
        self._metrics.record_task_event(event, is_child=is_child)
        if event_type == "TASK_FAILED":
            self._metrics.finalize_task_failure(task_id)

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
        ttl = self._cache_ttl_sec
        now_ts = time.time()
        for worker in pool:
            if ttl:
                ts_raw = worker.cache_updated_ts
                if not ts_raw:
                    continue
                try:
                    parsed = datetime.fromisoformat(ts_raw.rstrip("Z").replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                age = now_ts - parsed.timestamp()
                if age > ttl:
                    continue
            value = _score(worker)
            if value > 0:
                scored.append((worker, value))

        if not scored:
            return []

        max_score = max(score for _, score in scored)
        return [worker for worker, score in scored if score == max_score]

    def _preferred_stage_worker(self, record, task_payload: Dict[str, Any]) -> Optional[str]:
        if not record or not record.graph_node_name:
            return None
        task_type = (record.task_type or "").strip().lower()
        if task_type.startswith("lora"):
            return None
        payload = task_payload or {}
        if not isinstance(payload, dict):
            return None
        try:
            info = self._runtime.describe_task(record.task_id) or {}
        except Exception:
            info = {}
        depends_on = info.get("depends_on") or []
        if not depends_on:
            return None
        for dep_id in depends_on:
            dep_record = self._runtime.get_record(dep_id)
            if not dep_record or not dep_record.assigned_worker:
                continue
            stage_refs = [dep_record.graph_node_name, dep_id]
            for ref in stage_refs:
                if not ref:
                    continue
                if self._spec_has_weight_reference(payload, ref):
                    return dep_record.assigned_worker
        return None

    def _spec_has_weight_reference(self, payload: Dict[str, Any], stage_identifier: str) -> bool:
        token = f"${{{stage_identifier}.result"
        return self._search_weight_reference(payload, token, tuple())

    def _search_weight_reference(self, value: Any, token: str, path: Tuple[str, ...]) -> bool:
        if isinstance(value, dict):
            for key, sub in value.items():
                next_path = path + (str(key).lower(),)
                if self._search_weight_reference(sub, token, next_path):
                    return True
        elif isinstance(value, list):
            for item in value:
                if self._search_weight_reference(item, token, path):
                    return True
        elif isinstance(value, str):
            if token in value:
                lowered = value.lower()
                if any(hint in path for hint in self._weight_reference_hints) or any(
                    hint in lowered for hint in self._weight_reference_hints
                ):
                    return True
        return False

    def _should_wait_for_sticky_worker(self, record, worker_id: str) -> bool:
        if not record or not worker_id:
            return False
        task_type = (record.task_type or "").strip().lower()
        if not task_type:
            return False
        if task_type.startswith("lora"):
            return False
        category = (record.category or "").strip().lower()
        if category != "training":
            return False
        try:
            worker = get_worker_from_redis(self._redis, worker_id)
        except Exception:
            worker = None
        if not worker:
            return False
        try:
            if is_stale_by_redis(self._redis, worker_id):
                return False
        except Exception:
            return False
        status = (worker.status or "").upper()
        if status == "IDLE":
            return False
        return True

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
        submission_id = getattr(record, "submission_id", None)
        context: Dict[str, Any] = {}
        for other in self._runtime.tasks.values():
            if other.raw_yaml != base_yaml:
                continue
            if submission_id and getattr(other, "submission_id", None) != submission_id:
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
