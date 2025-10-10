from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from worker_cls import update_worker_status
from load_func import MEDIUM_LOAD_THRESHOLD
from task_store import TaskStore
from scheduler import (
    choose_strategy,
    estimate_queue_length,
    idle_satisfying_pool,
    is_data_parallel_enabled,
    make_shard_messages,
    select_workers_for_task,
)
from task import TaskRecord, TaskStatus, categorize_task_type
from parser import resolve_graph_templates
from utils import now_iso, safe_get


# Base requeue timing; small jitter reduces requeue stampedes.
REQUEUE_BASE_DELAY_SEC = max(0.0, float(os.getenv("TASK_REQUEUE_DELAY_SEC", "3")))
REQUEUE_MAX_DELAY_SEC = max(REQUEUE_BASE_DELAY_SEC, float(os.getenv("TASK_REQUEUE_MAX_DELAY_SEC", "30")))
REQUEUE_JITTER_FRAC = 0.15  # +/- 15% jitter to avoid synchronized retries


class DispatchManager:
    """
    Central dispatcher that integrates TaskStore's SLO-aware pool with
    worker selection and Redis-based publication.

    Responsibilities:
      - Configure TaskStore with dispatch callback.
      - Enqueue tasks and trigger pool scans.
      - Single-worker dispatch with backoff requeue on failure.
      - Data-parallel (sharded) dispatch and child TaskRecord creation.
    """

    def __init__(
        self,
        *,
        task_store: TaskStore,
        redis_client,
        tasks: Dict[str, TaskRecord],
        tasks_lock,
        parent_shards: Dict[str, Dict[str, Any]],
        child_to_parent: Dict[str, str],
        results_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self._task_store = task_store
        self._redis = redis_client
        self._tasks = tasks
        self._tasks_lock = tasks_lock
        self._parent_shards = parent_shards
        self._child_to_parent = child_to_parent
        self._results_dir = results_dir
        self._logger = logger

    # ------------------------------------------------------------------ #
    # TaskStore wiring
    # ------------------------------------------------------------------ #

    def configure_task_store(
        self,
        *,
        batch_size: int,
        slo_fraction: float,
        pending_status: str,
        done_status: str,
    ) -> None:
        """Wire this manager into the TaskStore so pool scans call back here."""
        self._task_store.configure_pool(
            batch_size=batch_size,
            slo_fraction=slo_fraction,
            redis_client=self._redis,
            dispatch_fn=self.try_dispatch_one,
            logger=self._logger,
            task_lookup=self._lookup_task,
            tasks_lock=self._tasks_lock,
            pending_status=pending_status,
            done_status=done_status,
        )

    # ------------------------------------------------------------------ #
    # Enqueue / Requeue
    # ------------------------------------------------------------------ #

    def enqueue_for_dispatch(
        self,
        task_id: str,
        task: Dict[str, Any],
        rec: TaskRecord,
        exclude_worker_id: Optional[str] = None,
    ) -> None:
        """Put a task into the pool and immediately trigger a scan."""
        self._task_store.enqueue_for_dispatch(task_id, task, rec, exclude_worker_id=exclude_worker_id)
        self._task_store.flush_pool()

    def _next_retry_timestamp(self, rec: TaskRecord) -> str:
        """
        Compute the next retry ETA with linear backoff and small jitter.
        attempt = max(1, retries+1)
        delay   = min(BASE * attempt, MAX) * (1 +/- jitter)
        """
        if REQUEUE_BASE_DELAY_SEC <= 0:
            return now_iso()
        attempt = max(1, int(getattr(rec, "retries", 0) or 0) + 1)
        delay = min(REQUEUE_BASE_DELAY_SEC * attempt, REQUEUE_MAX_DELAY_SEC)

        # Add small symmetric jitter to reduce synchronized requeues.
        if REQUEUE_JITTER_FRAC > 0:
            jitter = 1.0 + random.uniform(-REQUEUE_JITTER_FRAC, REQUEUE_JITTER_FRAC)
            delay = max(0.0, delay * jitter)

        eta = datetime.now(timezone.utc) + timedelta(seconds=delay)
        return eta.isoformat()

    def requeue_task(
        self,
        task_id: str,
        rec: Optional[TaskRecord],
        reason: str,
        *,
        task: Optional[Dict[str, Any]] = None,
        exclude_worker_id: Optional[str] = None,
    ) -> None:
        """
        Move a task back to PENDING with a next_retry_at, and re-enqueue it.
        Optionally exclude a worker (e.g., the one that just failed or had no subscribers).
        """
        if not rec:
            self._logger.warning("Cannot requeue %s: missing task record (%s)", task_id, reason)
            return

        payload = task or self._task_store.get_parsed(task_id) or rec.parsed
        if not payload:
            self._logger.warning("Cannot requeue %s: missing task payload (%s)", task_id, reason)
            return

        self._logger.info("Requeueing %s: %s", task_id, reason)
        with self._tasks_lock:
            rec.status = TaskStatus.PENDING
            rec.assigned_worker = None
            rec.error = reason
            rec.retries = int(getattr(rec, "retries", 0) or 0) + 1  # increment attempt counter
            rec.next_retry_at = self._next_retry_timestamp(rec)
            if exclude_worker_id:
                rec.last_failed_worker = exclude_worker_id
            rec.last_queue_ts = time.time()
            rec.started_ts = None
            rec.dispatched_ts = None
            rec.finished_ts = None

        self.enqueue_for_dispatch(task_id, payload, rec, exclude_worker_id=exclude_worker_id)

    # ------------------------------------------------------------------ #
    # Dispatch (single or sharded)
    # ------------------------------------------------------------------ #

    def try_dispatch_one(
        self,
        task_id: str,
        task: Dict[str, Any],
        rec: TaskRecord,
        exclude_worker_id: Optional[str] = None,
        preferred_worker_id: Optional[str] = None,
    ) -> None:
        """
        Attempt to dispatch one task:
          - Prefer data-parallel if enabled and capacity allows.
          - Else choose a single worker and publish the task.
          - If no eligible workers or publish failures, requeue with backoff.
        """
        # Already dispatched by another thread/path.
        with self._tasks_lock:
            if rec.status == TaskStatus.DISPATCHED:
                return

        # Build exclusion set (e.g., last failed worker).
        exclude_ids: Set[str] = set()
        if exclude_worker_id:
            exclude_ids.add(exclude_worker_id)

        # Try data-parallel path first.
        if self._try_dispatch_sharded(task_id, task, rec, exclude_ids=exclude_ids):
            return

        # Single-worker path.
        queued = estimate_queue_length(self._task_store)
        pool = idle_satisfying_pool(self._redis, task, exclude_ids=exclude_ids)

        # Stable prefer-ordering: keep others relative order intact.
        if preferred_worker_id:
            pool = sorted(pool, key=lambda w: (0 if w.worker_id == preferred_worker_id else 1, w.worker_id))

        if not pool:
            self._logger.info("No suitable IDLE worker for %s; deferring via task pool", task_id)
            self.requeue_task(
                task_id,
                rec,
                "No suitable IDLE worker; retry via task pool",
                task=task,
                exclude_worker_id=exclude_worker_id,
            )
            return

        strategy = choose_strategy(task, pool, queued)
        task_load = int(rec.load or 0)
        prefer_best = strategy.prefer_best or task_load >= MEDIUM_LOAD_THRESHOLD

        worker_list = select_workers_for_task(
            pool,
            shard_count=1,
            prefer_best=prefer_best,
            task_load=task_load,
        )
        if not worker_list:
            self._logger.info("Worker selection failed for %s; deferring via task pool", task_id)
            self.requeue_task(
                task_id,
                rec,
                "Worker selection failed; retry via task pool",
                task=task,
                exclude_worker_id=exclude_worker_id,
            )
            return

        worker = worker_list[0]
        topic = "tasks"

        # Resolve templates just-in-time to avoid stale rendering.
        task_payload = resolve_graph_templates(
            task, task_id, rec, self._task_store, self._tasks, self._results_dir, self._logger,
        )

        message = {
            "task_id": task_id,
            "task": task_payload,
            "task_type": safe_get(task_payload, "spec.taskType"),
            "assigned_worker": worker.worker_id,
            "dispatched_at": now_iso(),
            "parent_task_id": rec.parent_task_id,
        }

        try:
            receivers = self._publish_task(topic, message)
            if receivers <= 0:
                self._logger.warning(
                    "No subscribers listening on %s; requeueing %s via task pool",
                    topic, task_id,
                )
                self.requeue_task(
                    task_id,
                    rec,
                    "No active workers subscribed; retry via task pool",
                    task=task,
                    exclude_worker_id=worker.worker_id,
                )
                return

            update_worker_status(self._redis, worker.worker_id, "RUNNING")
            with self._tasks_lock:
                rec.status = TaskStatus.DISPATCHED
                rec.assigned_worker = worker.worker_id
                rec.topic = topic
                rec.next_retry_at = None
                rec.last_failed_worker = None

            self._task_store.record_assignment(task_payload, worker.worker_id)
            self._task_store.mark_released(task_id)
            with self._tasks_lock:
                rec.dispatched_ts = time.time()

        except Exception as exc:
            self._logger.warning("Publish failed for %s; requeueing via task pool", task_id)
            self.requeue_task(
                task_id,
                rec,
                f"Publish failed: {exc}",
                task=task,
                exclude_worker_id=worker.worker_id,
            )

    def _publish_task(self, topic: str, message: Dict[str, Any]) -> int:
        """Publish the task to Redis topic and return the subscriber count."""
        payload = json.dumps(message, ensure_ascii=False)
        receivers = self._redis.publish(topic, payload)
        self._logger.info("Published to topic=%s receivers=%d", topic, receivers)
        return int(receivers or 0)

    def _try_dispatch_sharded(
        self,
        parent_task_id: str,
        base_task: Dict[str, Any],
        parent_rec: TaskRecord,
        exclude_ids: Optional[Set[str]] = None,
    ) -> bool:
        """
        Attempt data-parallel dispatch when enabled and strategy recommends it.
        On success, create child TaskRecords and initialize parent aggregation map.
        """
        if not is_data_parallel_enabled(base_task):
            return False

        queued = estimate_queue_length(self._task_store)
        pool = idle_satisfying_pool(self._redis, base_task, exclude_ids=exclude_ids)
        if not pool:
            return False

        strategy = choose_strategy(base_task, pool, queued)
        if getattr(strategy, "mode", None) != "data_parallel":
            return False

        task_load = int(parent_rec.load or 0)
        workers = select_workers_for_task(
            pool,
            shard_count=strategy.shard_count,
            prefer_best=strategy.prefer_best or task_load >= MEDIUM_LOAD_THRESHOLD,
            task_load=task_load,
        )
        if len(workers) <= 1:
            return False

        child_msgs = make_shard_messages(parent_task_id, base_task, workers)

        topic = "tasks"
        created_children: List[str] = []
        order_map: Dict[str, int] = {}

        for idx, (child_id, child_task, worker) in enumerate(child_msgs):
            # Avoid per-shard HTTP delivery: force local output for children.
            child_spec = dict(child_task.get("spec") or {})
            output_cfg = dict(child_spec.get("output") or {})
            destination = dict(output_cfg.get("destination") or {})
            if str(destination.get("type", "")).lower() == "http":
                destination = {"type": "local"}
                output_cfg["destination"] = destination
                child_spec["output"] = output_cfg
                child_task = dict(child_task)
                child_task["spec"] = child_spec

            message = {
                "task_id": child_id,
                "task": child_task,
                "task_type": safe_get(child_task, "spec.taskType"),
                "assigned_worker": worker.worker_id,
                "dispatched_at": now_iso(),
                "parent_task_id": parent_task_id,
                "shard_index": idx,
                "shard_total": len(child_msgs),
            }

            try:
                receivers = self._publish_task(topic, message)
                if receivers <= 0:
                    self._logger.warning(
                        "No subscribers for shard %s on topic %s; will retry later",
                        child_id, topic,
                    )
                    continue

                update_worker_status(self._redis, worker.worker_id, "RUNNING")

                child_type = safe_get(child_task, "spec.taskType")
                child_category = categorize_task_type(child_type)
                child_rec = TaskRecord(
                    task_id=child_id,
                    raw_yaml=parent_rec.raw_yaml,
                    parsed=child_task,
                    status=TaskStatus.DISPATCHED,
                    assigned_worker=worker.worker_id,
                    topic=topic,
                    parent_task_id=parent_task_id,
                    shard_index=idx,
                    shard_total=len(child_msgs),  # keep consistent with message
                    max_retries=parent_rec.max_retries,
                    load=parent_rec.load,
                    slo_seconds=parent_rec.slo_seconds,
                    task_type=child_type,
                    category=child_category,
                )
                with self._tasks_lock:
                    self._tasks[child_id] = child_rec
                    self._child_to_parent[child_id] = parent_task_id
                    child_rec.last_queue_ts = time.time()
                    child_rec.dispatched_ts = child_rec.last_queue_ts

                created_children.append(child_id)
                order_map[child_id] = idx

            except Exception as exc:
                self._logger.exception("Shard publish failed for %s: %s", child_id, exc)
                with self._tasks_lock:
                    self._tasks[child_id] = TaskRecord(
                        task_id=child_id,
                        raw_yaml=parent_rec.raw_yaml,
                        parsed=child_task,
                        status=TaskStatus.FAILED,
                        error=f"Publish failed: {exc}",
                        parent_task_id=parent_task_id,
                        shard_index=idx,
                        shard_total=len(child_msgs),
                        load=parent_rec.load,
                        slo_seconds=parent_rec.slo_seconds,
                    )

        if not created_children:
            return False

        with self._tasks_lock:
            self._parent_shards[parent_task_id] = {
                "total": len(created_children),
                "done": 0,
                "failed": 0,
                "children": set(created_children),
                "order": order_map,
                "results": {},
                "aggregated": False,
            }
            parent_rec.status = TaskStatus.DISPATCHED
            parent_rec.assigned_worker = "MULTI"
            parent_rec.topic = topic

        self._task_store.mark_released(parent_task_id)
        return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _lookup_task(self, task_id: str) -> Optional[TaskRecord]:
        """TaskStore callback to resolve a TaskRecord by id."""
        with self._tasks_lock:
            return self._tasks.get(task_id)
