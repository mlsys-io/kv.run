from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Optional, Set

from .dispatcher import Dispatcher, StageReferenceNotReady
from .utils import now_iso
from .worker_registry import (
    get_worker_from_redis,
    idle_satisfying_pool,
    is_stale_by_redis,
    list_workers_from_redis,
    update_worker_status,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker


class FixedPipelineDispatcher(Dispatcher):
    """
    Baseline dispatcher that pins each task type to a limited worker set.

    The first time a task type is observed, the dispatcher chooses an idle worker
    and remembers that assignment. It may provision additional workers for the
    same type, but never exceeds half of the currently known workers. Tasks wait
    whenever all bound workers are busy or unavailable, preserving the fixed
    pipeline flavour while capping concurrency per task type.
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
        binding_idle_timeout_sec: int = 0,
    ) -> None:
        super().__init__(
            runtime,
            redis_client,
            results_dir,
            logger=logger,
            worker_selection_strategy=worker_selection_strategy,
            metrics_recorder=metrics_recorder,
        )
        self._type_to_workers: Dict[str, Set[str]] = {}
        self._type_order: Dict[str, List[str]] = {}
        self._type_rr_index: Dict[str, int] = {}
        self._worker_to_type: Dict[str, str] = {}
        self._binding_idle_timeout_sec = max(0, int(binding_idle_timeout_sec))
        self._binding_last_used: Dict[str, float] = {}

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
        self._note_binding_usage(worker_id)
        return True

    # ------------------------------------------------------------------ #
    # Worker resolution helpers
    # ------------------------------------------------------------------ #

    def _resolve_worker(self, task_type: str, payload) -> Optional[str]:
        pool_valid = True
        try:
            pool = idle_satisfying_pool(self._redis, payload)
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Fixed pipeline dispatcher failed to build eligible pool for %s: %s", task_type, exc)
            pool = []
            pool_valid = False
        eligible_ids = {worker.worker_id for worker in pool} if pool_valid else None

        idle_bound = self._idle_bound_worker(task_type, eligible_ids)
        if idle_bound:
            return idle_bound

        max_workers_for_type = self._max_workers_for_type()
        bound_workers = self._type_to_workers.get(task_type, set())
        if len(bound_workers) >= max_workers_for_type:
            return None

        if not pool:
            return None

        filtered_pool = [
            worker
            for worker in pool
            if self._worker_to_type.get(worker.worker_id) in (None, task_type)
        ]
        if not filtered_pool:
            return None

        worker, _ = select_worker(filtered_pool, self._worker_selection_strategy, logger=self._logger)
        if not worker:
            return None

        if worker.worker_id not in bound_workers:
            self._register_binding(task_type, worker.worker_id)
            self._logger.info(
                "Fixed pipeline dispatcher bound task type %s to worker %s (active bindings=%d, limit=%d)",
                task_type,
                worker.worker_id,
                len(self._type_to_workers.get(task_type, set())),
                max_workers_for_type,
            )
        return worker.worker_id

    def _idle_bound_worker(self, task_type: str, eligible_ids: Optional[Set[str]]) -> Optional[str]:
        order = self._type_order.get(task_type, [])
        if not order:
            return None
        start_index = self._type_rr_index.get(task_type, 0)
        count = len(order)
        idx = start_index % count
        checked = 0
        while order and checked < count:
            worker_id = order[idx]
            worker = get_worker_from_redis(self._redis, worker_id)
            if not worker:
                self._logger.info(
                    "Fixed pipeline dispatcher dropping binding for %s; worker %s missing",
                    task_type,
                    worker_id,
                )
                self._unregister_binding(worker_id)
                order = self._type_order.get(task_type, [])
                count = len(order)
                if count == 0:
                    self._type_rr_index.pop(task_type, None)
                    return None
                idx %= count
                checked += 1
                continue
            if worker.status != "IDLE":
                idx = (idx + 1) % count
                checked += 1
                continue
            try:
                if is_stale_by_redis(self._redis, worker.worker_id):
                    idx = (idx + 1) % count
                    checked += 1
                    continue
            except Exception:  # noqa: broad-except
                idx = (idx + 1) % count
                checked += 1
                continue
            if self._binding_idle_timeout_sec > 0:
                last_used = self._binding_last_used.get(worker.worker_id)
                now = time.time()
                idle_duration = 0.0 if last_used is None else now - last_used
                if last_used is None or idle_duration >= self._binding_idle_timeout_sec:
                    self._logger.info(
                        "Fixed pipeline dispatcher releasing binding for %s; worker %s idle for %.0fs (limit=%ds)",
                        task_type,
                        worker.worker_id,
                        idle_duration,
                        self._binding_idle_timeout_sec,
                    )
                    self._unregister_binding(worker.worker_id)
                    order = self._type_order.get(task_type, [])
                    count = len(order)
                    if count == 0:
                        self._type_rr_index.pop(task_type, None)
                        return None
                    idx %= count
                    checked += 1
                    continue
            if eligible_ids is not None and worker.worker_id not in eligible_ids:
                self._logger.info(
                    "Fixed pipeline dispatcher dropping binding for %s; worker %s no longer satisfies resources",
                    task_type,
                    worker.worker_id,
                )
                self._unregister_binding(worker.worker_id)
                order = self._type_order.get(task_type, [])
                count = len(order)
                if count == 0:
                    self._type_rr_index.pop(task_type, None)
                    return None
                idx %= count
                checked += 1
                continue
            self._type_rr_index[task_type] = (idx + 1) % count
            return worker.worker_id
        if order:
            self._type_rr_index[task_type] = idx % len(order)
        else:
            self._type_rr_index.pop(task_type, None)
        return None

    def _note_binding_usage(self, worker_id: str) -> None:
        self._binding_last_used[worker_id] = time.time()

    def _register_binding(self, task_type: str, worker_id: str) -> None:
        workers = self._type_to_workers.setdefault(task_type, set())
        workers.add(worker_id)
        order = self._type_order.setdefault(task_type, [])
        if worker_id not in order:
            order.append(worker_id)
        self._type_rr_index.setdefault(task_type, 0)
        self._worker_to_type[worker_id] = task_type
        self._binding_last_used[worker_id] = time.time()

    def _unregister_binding(self, worker_id: str) -> None:
        task_type = self._worker_to_type.pop(worker_id, None)
        if not task_type:
            return
        workers = self._type_to_workers.get(task_type)
        if not workers:
            return
        workers.discard(worker_id)
        order = self._type_order.get(task_type)
        if order and worker_id in order:
            index = order.index(worker_id)
            order.pop(index)
            if not order:
                self._type_order.pop(task_type, None)
                self._type_rr_index.pop(task_type, None)
            else:
                self._type_rr_index[task_type] = min(self._type_rr_index.get(task_type, 0), len(order) - 1)
        if not workers:
            self._type_to_workers.pop(task_type, None)
        self._binding_last_used.pop(worker_id, None)

    def _max_workers_for_type(self) -> int:
        try:
            workers = list_workers_from_redis(self._redis)
        except Exception:  # noqa: broad-except
            workers = []
        active_count = sum(1 for worker in workers if not worker.get("stale"))
        total = active_count if active_count > 0 else len(workers)
        limit = total // 2
        if limit < 1:
            limit = 1
        return limit
