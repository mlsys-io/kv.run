from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, Optional

import redis

from .task_runtime import TaskRuntime
from .utils import now_iso
from .worker_registry import (
    idle_satisfying_pool,
    sort_workers,
    update_worker_status,
)


class Dispatcher:
    """Handles FCFS task dispatching via Redis pub/sub."""

    def __init__(
        self,
        runtime: TaskRuntime,
        redis_client: redis.Redis,
        *,
        logger,
    ) -> None:
        self._runtime = runtime
        self._redis = redis_client
        self._logger = logger

    def dispatch_once(self, task_id: str) -> bool:
        """Dispatch a single task if possible; requeue when no worker."""
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        task_payload = copy.deepcopy(record.parsed)
        pool = idle_satisfying_pool(self._redis, task_payload)
        if not pool:
            self._logger.debug("No idle worker available for %s; requeueing", task_id)
            self._runtime.mark_pending(task_id)
            self._runtime.requeue(task_id)
            return False

        sorted_pool = sort_workers(pool) if pool else pool
        worker = sorted_pool[0]

        message = {
            "task_id": task_id,
            "task": task_payload,
            "task_type": record.task_type,
            "assigned_worker": worker.worker_id,
            "dispatched_at": now_iso(),
            "parent_task_id": None,
        }

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
