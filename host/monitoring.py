from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import redis

from .elastic import ElasticCoordinator
from .event_schema import TaskEvent, WorkerEvent, parse_event
from .manifest_utils import sync_manifest
from .metrics import MetricsRecorder
from .results import result_file_path
from .task_metadata import extract_model_dataset_names
from .task_runtime import TaskRuntime
from .utils import log_worker_event
from .worker_registry import record_worker_cache, update_worker_status
from .watchdog import WorkerWatchdog


class TaskWorkerMonitor:
    """Background listeners for task and worker event streams."""

    def __init__(
        self,
        *,
        redis_url: str,
        redis_client: Any,
        stop_event: threading.Event,
        logger,
        runtime: TaskRuntime,
        dispatcher: Any,
        metrics_recorder: MetricsRecorder,
        elastic_coordinator: ElasticCoordinator,
        watchdog: WorkerWatchdog,
        results_dir: Path,
    ) -> None:
        self._redis_url = redis_url
        self._redis_client = redis_client
        self._stop_event = stop_event
        self._logger = logger
        self._runtime = runtime
        self._dispatcher = dispatcher
        self._metrics = metrics_recorder
        self._elastic = elastic_coordinator
        self._watchdog = watchdog
        self._results_dir = Path(results_dir)

        self._pending_result_clones: Dict[str, List[str]] = {}
        self._pending_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> List[threading.Thread]:
        """Spawn task/worker listener threads."""
        threads = [
            threading.Thread(target=self._tasks_events_loop, name="tasks-events", daemon=True),
            threading.Thread(target=self._workers_events_loop, name="workers-events", daemon=True),
        ]
        for thread in threads:
            thread.start()
        return threads

    def mirror_task_results(self, parent_task_id: str, child_ids: List[str]) -> None:
        if not child_ids:
            return
        parent_dir = result_file_path(self._results_dir, parent_task_id).parent
        if not parent_dir.exists():
            with self._pending_lock:
                pending = self._pending_result_clones.setdefault(parent_task_id, [])
                for child in child_ids:
                    if child not in pending:
                        pending.append(child)
            self._logger.debug("Deferring result mirroring for %s (waiting for artifacts)", parent_task_id)
            return

        for child_id in child_ids:
            if child_id == parent_task_id:
                continue
            dst_dir = result_file_path(self._results_dir, child_id).parent
            if dst_dir.exists() and (dst_dir / "responses.json").exists():
                continue
            try:
                if dst_dir.exists():
                    shutil.rmtree(dst_dir, ignore_errors=True)
                shutil.copytree(parent_dir, dst_dir)
                record = self._runtime.get_record(child_id)
                expected_artifacts: List[str] = []
                if record:
                    expected_artifacts = (
                        ((record.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
                    )
                sync_manifest(dst_dir, child_id, expected_artifacts)
            except Exception as exc:  # pragma: no cover
                self._logger.debug("Failed to mirror results from %s to %s: %s", parent_task_id, child_id, exc)

        with self._pending_lock:
            self._pending_result_clones.pop(parent_task_id, None)

    def pop_pending_clones(self, task_id: str) -> List[str]:
        with self._pending_lock:
            return self._pending_result_clones.pop(task_id, [])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _tasks_events_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sub_rds = redis.from_url(self._redis_url, decode_responses=True)
                pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe("tasks.events")
                self._logger.info("Subscribed to tasks.events")
                for msg in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    if msg.get("type") != "message":
                        continue
                    raw = msg.get("data")
                    try:
                        data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                    except Exception as exc:
                        self._logger.debug("Invalid tasks.events payload: %s", exc)
                        continue
                    try:
                        event = parse_event(data)
                    except Exception as exc:
                        self._logger.debug("Failed to parse task event: %s (%s)", data, exc)
                        continue
                    if isinstance(event, TaskEvent):
                        self._handle_task_event(event)
                try:
                    pubsub.close()
                except Exception:
                    pass
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning("tasks.events listener error: %s; reconnecting...", exc)
                time.sleep(2.0)

    def _handle_task_event(self, event: TaskEvent) -> None:
        payload = event.payload or {}
        event_type = event.type
        if event_type == "TASK_STARTED":
            self._metrics.record_task_event(event)
            self._runtime.mark_started(event.task_id, event.worker_id, payload, event.ts)
            return

        if event_type == "TASK_SUCCEEDED":
            self._metrics.record_task_event(event)
            _, merged_children = self._runtime.mark_succeeded(
                event.task_id, event.worker_id, payload, event.ts
            )
            try:
                queueing, dispatched, pending, done, total = self._runtime.task_status_counts()
                summary = (
                    f"QUEUEING {queueing}, DISPATCHED {dispatched}, PENDING {pending}, "
                    f"DONE {done}, TOTAL {total}"
                )
            except Exception:
                summary = (
                    "QUEUEING UNKNOWN, DISPATCHED UNKNOWN, PENDING UNKNOWN, "
                    "DONE UNKNOWN, TOTAL UNKNOWN"
                )
            self._logger.info("Task %s completed; %s", event.task_id, summary)
            if merged_children:
                for child_id in merged_children:
                    child_payload = dict(payload)
                    child_payload["parent_task_id"] = event.task_id
                    child_payload["is_child_task"] = True
                    child_event = TaskEvent(
                        type="TASK_SUCCEEDED",
                        task_id=child_id,
                        worker_id=event.worker_id,
                        payload=child_payload,
                        ts=event.ts,
                    )
                    self._metrics.record_task_event(child_event, is_child=True)
            if event.worker_id:
                try:
                    record = self._runtime.get_record(event.task_id)
                    if record:
                        models, datasets = extract_model_dataset_names(record.parsed)
                        if models or datasets:
                            record_worker_cache(
                                self._redis_client,
                                event.worker_id,
                                models=models,
                                datasets=datasets,
                            )
                except Exception as exc:
                    self._logger.debug("Failed to update cache metadata for worker %s: %s", event.worker_id, exc)
                try:
                    update_worker_status(self._redis_client, event.worker_id, "IDLE")
                except Exception:
                    pass
            if merged_children:
                self.mirror_task_results(event.task_id, merged_children)
            return

        if event_type == "TASK_FAILED":
            record = self._runtime.get_record(event.task_id)
            attempts = record.attempts if record else 0
            max_attempts = record.max_attempts if record else None
            can_retry = record and (max_attempts is None or max_attempts < 0 or attempts < max_attempts)
            if can_retry:
                limit_display = "∞" if max_attempts is None or max_attempts < 0 else max_attempts
                if event.worker_id and record:
                    record.last_failed_worker = event.worker_id
                self._logger.warning(
                    "Retrying task %s after worker failure (%d/%s)",
                    event.task_id,
                    attempts + 1,
                    limit_display,
                )
                self._dispatcher._requeue_task(  # noqa: SLF001
                    event.task_id,
                    reason="worker_failed",
                    front=True,
                    extra_payload={
                        "error": event.error,
                        "attempt": attempts + 1,
                        "max_attempts": max_attempts,
                    },
                )
                return

            self._metrics.record_task_event(event)
            impacted, merged_children = self._runtime.mark_failed(
                event.task_id,
                event.worker_id,
                payload,
                event.ts,
                error=event.error,
            )
            self._metrics.finalize_task_failure(event.task_id)
            for task_id, reason in impacted:
                derived = TaskEvent(
                    type="TASK_FAILED",
                    task_id=task_id,
                    error=reason,
                    payload={"dependency_failure": event.task_id},
                )
                self._metrics.record_task_event(derived)
                self._metrics.finalize_task_failure(task_id)
            for child_id in merged_children:
                child_payload = dict(payload)
                child_payload["parent_task_id"] = event.task_id
                child_payload["dependency_failure"] = event.task_id
                child_payload["is_child_task"] = True
                child_event = TaskEvent(
                    type="TASK_FAILED",
                    task_id=child_id,
                    worker_id=event.worker_id,
                    error=event.error or "parent_failed",
                    payload=child_payload,
                )
                self._metrics.record_task_event(child_event, is_child=True)
                self._metrics.finalize_task_failure(child_id)
            return

        self._logger.debug("Ignoring task event type=%s payload=%s", event_type, payload)

    def _workers_events_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sub_rds = redis.from_url(self._redis_url, decode_responses=True)
                pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe("workers.events")
                self._logger.info("Subscribed to workers.events")
                for msg in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    if msg.get("type") != "message":
                        continue
                    raw = msg.get("data")
                    try:
                        data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                    except Exception as exc:
                        self._logger.debug("Invalid workers.events payload: %s", exc)
                        continue
                    try:
                        event = parse_event(data)
                    except Exception as exc:
                        self._logger.debug("Failed to parse worker event: %s (%s)", data, exc)
                        continue
                    if not isinstance(event, WorkerEvent):
                        continue
                    log_worker_event(self._logger, event)
                    self._metrics.record_worker_event(event)
                    self._elastic.record_worker_event(event)
                    if event.type == "UNREGISTER":
                        worker_id = (event.worker_id or "").strip()
                        if worker_id:
                            if self._watchdog.enabled and self._watchdog.is_marked_dead(worker_id):
                                self._logger.info(
                                    "Skipping direct requeue for %s unregister; watchdog already emitted synthetic failures",
                                    worker_id,
                                )
                                self._watchdog.clear_dead_mark(worker_id)
                                continue
                            recovered = self._runtime.recover_tasks_for_worker(worker_id)
                            if recovered:
                                self._logger.info(
                                    "Requeued %d task(s) after worker %s unregistered: %s",
                                    len(recovered),
                                    worker_id,
                                    ", ".join(recovered),
                                )
                                for task_id in recovered:
                                    record = self._runtime.get_record(task_id)
                                    if record:
                                        record.last_failed_worker = worker_id
                                    self._dispatcher._requeue_task(  # noqa: SLF001
                                        task_id,
                                        reason="worker_unregistered",
                                        front=True,
                                        extra_payload={"worker": worker_id},
                                    )
                try:
                    pubsub.close()
                except Exception:
                    pass
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning("workers.events listener error: %s; reconnecting...", exc)
                time.sleep(2.0)
