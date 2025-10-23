from __future__ import annotations

import copy
import json
from typing import Dict, Optional

from .dispatcher import Dispatcher, StageReferenceNotReady
from .utils import now_iso
from .worker_registry import (
    get_worker_from_redis,
    idle_satisfying_pool,
    is_stale_by_redis,
    update_worker_status,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker


class FixedPipelineDispatcher(Dispatcher):
    """
    Baseline dispatcher that locks each task type to a single worker.

    The first time a task type is observed, the dispatcher chooses one idle worker
    for it and remembers that assignment. Subsequent tasks of the same type are
    only dispatched to the same worker, effectively simulating rigid executor
    binding. Tasks for which the assigned worker is busy or unavailable are
    requeued until the desired worker returns to idle state.
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
        self._type_to_worker: Dict[str, str] = {}

    def dispatch_once(self, task_id: str) -> bool:
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        task_type = record.task_type or ""
        if not task_type:
            self._logger.debug("Fixed pipeline dispatcher skipping %s: task type missing", task_id)
            self._requeue_task(task_id, reason="missing_task_type", release_merge=False)
            return False

        worker_id = self._resolve_worker(task_type, record.parsed)
        if not worker_id:
            self._logger.debug("Fixed pipeline dispatcher delaying %s: no worker ready", task_id)
            self._requeue_task(
                task_id,
                reason="no_worker_ready",
                release_merge=False,
                count_retry=False,
            )
            return False

        task_payload = copy.deepcopy(record.parsed)
        message = {
            "task_id": task_id,
            "task": task_payload,
            "task_type": task_type,
            "assigned_worker": worker_id,
            "dispatched_at": now_iso(),
            "parent_task_id": None,
        }

        try:
            rendered_task = self._resolve_stage_references(task_id, task_payload, record)
        except StageReferenceNotReady as exc:
            self._logger.debug("Fixed pipeline dispatcher waiting for %s: %s", task_id, exc)
            self._requeue_task(
                task_id,
                reason="stage_reference_pending",
                release_merge=False,
                count_retry=False,
            )
            return False
        except Exception as exc:  # noqa: broad-except
            self._logger.error("Fixed pipeline dispatcher failed to resolve %s: %s", task_id, exc)
            self._fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        message["task"] = rendered_task

        try:
            receivers = int(self._redis.publish("tasks", json.dumps(message, ensure_ascii=False)))
        except Exception as exc:  # noqa: broad-except
            self._logger.warning("Fixed pipeline dispatcher failed to publish %s: %s", task_id, exc)
            self._requeue_task(task_id, reason="publish_failed", front=True, release_merge=False)
            return False

        if receivers <= 0:
            self._logger.info("Fixed pipeline dispatcher delaying %s: no subscribers", task_id)
            self._requeue_task(task_id, reason="no_subscribers", front=True, release_merge=False)
            return False

        self._runtime.mark_dispatched(task_id, worker_id)
        try:
            update_worker_status(self._redis, worker_id, "RUNNING")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Fixed pipeline dispatcher failed to update %s status: %s", worker_id, exc)
        return True

    # ------------------------------------------------------------------ #
    # Worker resolution helpers
    # ------------------------------------------------------------------ #

    def _resolve_worker(self, task_type: str, payload) -> Optional[str]:
        existing = self._type_to_worker.get(task_type)
        if existing:
            worker = get_worker_from_redis(self._redis, existing)
            if not worker:
                self._logger.info(
                    "Fixed pipeline dispatcher dropping binding for %s; worker %s missing",
                    task_type,
                    existing,
                )
                self._type_to_worker.pop(task_type, None)
                return self._resolve_worker(task_type, payload)
            if worker.status != "IDLE" or is_stale_by_redis(self._redis, worker.worker_id):
                return None
            return worker.worker_id

        pool = idle_satisfying_pool(self._redis, payload)
        if not pool:
            return None

        worker, _ = select_worker(pool, self._worker_selection_strategy, logger=self._logger)
        if not worker:
            return None
        self._type_to_worker[task_type] = worker.worker_id
        self._logger.info(
            "Fixed pipeline dispatcher bound task type %s to worker %s",
            task_type,
            worker.worker_id,
        )
        return worker.worker_id
