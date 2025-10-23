from __future__ import annotations

import copy
import json
from typing import Dict, Optional

from .dispatcher import Dispatcher, StageReferenceNotReady
from .task_models import TaskStatus
from .utils import now_iso
from .worker_registry import (
    get_worker_from_redis,
    idle_satisfying_pool,
    is_stale_by_redis,
    update_worker_status,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker


class StaticWorkerDispatcher(Dispatcher):
    """
    Baseline dispatcher that assigns an entire workflow submission to a single worker.

    Once a workflow is bound to a worker, no other workflow can use that worker until
    every task in the submission has reached a terminal status. This models
    traditional serverless deployments where a client's DAG is pinned to a cold start.
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
        self._workflow_to_worker: Dict[str, str] = {}
        self._worker_to_workflow: Dict[str, str] = {}

    def dispatch_once(self, task_id: str) -> bool:
        self._cleanup_completed_workflows()

        record = self._runtime.get_record(task_id)
        if not record:
            return True

        workflow_id = self._workflow_key(record)
        if not workflow_id:
            self._logger.debug("Static worker dispatcher skipping %s: missing submission id", task_id)
            self._requeue_task(task_id, reason="missing_workflow_id", release_merge=False)
            return False

        worker_id = self._resolve_worker(workflow_id, record.parsed)
        if not worker_id:
            self._logger.debug("Static worker dispatcher delaying %s: no worker available", task_id)
            self._requeue_task(
                task_id,
                reason="static_worker_unavailable",
                release_merge=False,
                count_retry=False,
            )
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
            self._logger.debug("Static worker dispatcher waiting for %s: %s", task_id, exc)
            self._requeue_task(
                task_id,
                reason="stage_reference_pending",
                release_merge=False,
                count_retry=False,
            )
            return False
        except Exception as exc:  # noqa: broad-except
            self._logger.error("Static worker dispatcher failed to resolve %s: %s", task_id, exc)
            self._fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        message["task"] = rendered_task

        try:
            receivers = int(self._redis.publish("tasks", json.dumps(message, ensure_ascii=False)))
        except Exception as exc:  # noqa: broad-except
            self._logger.warning("Static worker dispatcher failed to publish %s: %s", task_id, exc)
            self._requeue_task(task_id, reason="publish_failed", front=True, release_merge=False)
            self._invalidate_binding(workflow_id)
            return False

        if receivers <= 0:
            self._logger.info("Static worker dispatcher delaying %s: no subscribers", task_id)
            self._requeue_task(task_id, reason="no_subscribers", front=True, release_merge=False)
            return False

        self._runtime.mark_dispatched(task_id, worker_id)
        try:
            update_worker_status(self._redis, worker_id, "RUNNING")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Static worker dispatcher failed to update %s status: %s", worker_id, exc)
        return True

    # ------------------------------------------------------------------ #
    # Worker and workflow helpers
    # ------------------------------------------------------------------ #

    def _workflow_key(self, record) -> Optional[str]:
        submission_id = getattr(record, "submission_id", None)
        if submission_id:
            return submission_id
        if record.raw_yaml:
            return record.raw_yaml
        return None

    def _resolve_worker(self, workflow_id: str, payload) -> Optional[str]:
        existing_worker = self._workflow_to_worker.get(workflow_id)
        if existing_worker:
            if self._worker_to_workflow.get(existing_worker) != workflow_id:
                # Stale mapping; drop it and retry.
                self._invalidate_binding(workflow_id)
                return self._resolve_worker(workflow_id, payload)

            worker = get_worker_from_redis(self._redis, existing_worker)
            if not worker:
                self._logger.info(
                    "Static worker dispatcher dropping binding %s -> %s: worker missing",
                    workflow_id,
                    existing_worker,
                )
                self._invalidate_binding(workflow_id)
                return self._resolve_worker(workflow_id, payload)
            if worker.status != "IDLE" or is_stale_by_redis(self._redis, worker.worker_id):
                return None
            return worker.worker_id

        pool = [
            worker
            for worker in idle_satisfying_pool(self._redis, payload)
            if worker.worker_id not in self._worker_to_workflow
        ]
        if not pool:
            return None

        worker, _ = select_worker(pool, self._worker_selection_strategy, logger=self._logger)
        if not worker:
            return None

        self._workflow_to_worker[workflow_id] = worker.worker_id
        self._worker_to_workflow[worker.worker_id] = workflow_id
        self._logger.info(
            "Static worker dispatcher bound workflow %s to worker %s",
            workflow_id,
            worker.worker_id,
        )
        return worker.worker_id

    def _cleanup_completed_workflows(self) -> None:
        for workflow_id, worker_id in list(self._workflow_to_worker.items()):
            if self._is_workflow_active(workflow_id):
                continue
            self._logger.debug(
                "Static worker dispatcher releasing worker %s from workflow %s",
                worker_id,
                workflow_id,
            )
            self._workflow_to_worker.pop(workflow_id, None)
            mapped = self._worker_to_workflow.get(worker_id)
            if mapped == workflow_id:
                self._worker_to_workflow.pop(worker_id, None)

    def _invalidate_binding(self, workflow_id: str) -> None:
        worker_id = self._workflow_to_worker.pop(workflow_id, None)
        if not worker_id:
            return
        if self._worker_to_workflow.get(worker_id) == workflow_id:
            self._worker_to_workflow.pop(worker_id, None)

    def _is_workflow_active(self, workflow_id: str) -> bool:
        for record in self._runtime.tasks.values():
            if getattr(record, "submission_id", None) == workflow_id:
                if record.status not in {TaskStatus.DONE, TaskStatus.FAILED}:
                    return True
        return False
