from __future__ import annotations

import copy
import json
from typing import List, Optional

from .dispatcher import Dispatcher, StageReferenceNotReady
from .utils import now_iso
from .worker_registry import list_workers_from_redis, update_worker_status
from .worker_selector import DEFAULT_WORKER_SELECTION


class StaticRoundRobinDispatcher(Dispatcher):
    """
    Baseline dispatcher that ignores capability matching and assigns jobs
    using a simple round-robin over idle workers.

    The implementation keeps a rotating list of idle workers (filtered only by
    status and staleness) and hands out tasks in that fixed order. It reuses the
    base dispatcher for stage resolution so graph-based YAMLs remain compatible.
    """

    def __init__(
        self,
        runtime,
        redis_client,
        results_dir,
        *,
        logger,
        worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
        metrics_recorder=None,
    ) -> None:
        super().__init__(
            runtime,
            redis_client,
            results_dir,
            logger=logger,
            worker_selection_strategy=worker_selection_strategy,
            metrics_recorder=metrics_recorder,
        )
        self._rr_workers: List[str] = []
        self._rr_index: int = 0

    def dispatch_once(self, task_id: str) -> bool:
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        worker_id = self._next_worker()
        if not worker_id:
            self._logger.debug("Static RR: no idle worker available for %s; requeueing", task_id)
            self._requeue_task(task_id, reason="no_idle_worker", release_merge=False)
            return False

        task_payload = copy.deepcopy(record.parsed)
        message = {
            "task_id": task_id,
            "task": task_payload,
            "task_type": record.task_type,
            "assigned_worker": worker_id,
            "dispatched_at": now_iso(),
            "parent_task_id": None,
        }

        try:
            rendered_task = self._resolve_stage_references(task_id, task_payload, record)
        except StageReferenceNotReady as exc:
            self._logger.debug("Static RR: task %s waiting on stage artifacts: %s", task_id, exc)
            self._requeue_task(
                task_id,
                reason="stage_reference_pending",
                release_merge=False,
                count_retry=False,
            )
            return False
        except Exception as exc:  # noqa: broad-except
            self._logger.error("Static RR: failed to resolve stage references for %s: %s", task_id, exc)
            self._fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        message["task"] = rendered_task

        try:
            receivers = int(self._redis.publish("tasks", json.dumps(message, ensure_ascii=False)))
        except Exception as exc:  # noqa: broad-except
            self._logger.warning("Static RR: failed to publish task %s: %s", task_id, exc)
            self._requeue_task(task_id, reason="publish_failed", front=True, release_merge=False)
            self._invalidate_worker(worker_id)
            return False

        if receivers <= 0:
            self._logger.info("Static RR: no subscribers on tasks channel; delaying task %s", task_id)
            self._requeue_task(task_id, reason="no_subscribers", front=True, release_merge=False)
            self._invalidate_worker(worker_id)
            return False

        self._runtime.mark_dispatched(task_id, worker_id)
        try:
            update_worker_status(self._redis, worker_id, "RUNNING")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Static RR: failed to update worker %s status: %s", worker_id, exc)
        self._invalidate_worker(worker_id)
        return True

    # ------------------------------------------------------------------ #
    # Round-robin helpers
    # ------------------------------------------------------------------ #

    def _next_worker(self) -> Optional[str]:
        self._refresh_worker_cache()
        if not self._rr_workers:
            return None
        worker_id = self._rr_workers[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(self._rr_workers)
        return worker_id

    def _refresh_worker_cache(self) -> None:
        available = self._available_worker_ids()
        if not available:
            self._rr_workers = []
            self._rr_index = 0
            return

        if not self._rr_workers:
            self._rr_workers = available
            self._rr_index %= len(self._rr_workers)
            return

        existing = self._rr_workers
        new_order: List[str] = [worker for worker in existing if worker in available]
        for worker in available:
            if worker not in new_order:
                new_order.append(worker)

        if new_order:
            self._rr_workers = new_order
            self._rr_index %= len(self._rr_workers)
        else:
            self._rr_workers = []
            self._rr_index = 0

    def _available_worker_ids(self) -> List[str]:
        workers = list_workers_from_redis(self._redis)
        return [
            worker["worker_id"]
            for worker in workers
            if worker.get("status") == "IDLE" and not worker.get("stale")
        ]

    def _invalidate_worker(self, worker_id: str) -> None:
        """
        Remove the worker from the local round-robin list once it has been used.
        This keeps the cache aligned with subsequent Redis refreshes even if the
        worker status update is delayed.
        """
        if worker_id not in self._rr_workers:
            return
        index = self._rr_workers.index(worker_id)
        self._rr_workers.pop(index)
        if not self._rr_workers:
            self._rr_index = 0
            return
        if index < self._rr_index:
            self._rr_index -= 1
        self._rr_index %= len(self._rr_workers)
