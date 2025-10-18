from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict, deque
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from .task_models import TaskRecord, TaskStatus, categorize_task_type
from .task_parser import parse_task_yaml
from .utils import parse_iso_ts


def _sanitize_merge_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    clone = copy.deepcopy(spec)
    if isinstance(clone.get("inference"), dict):
        inference_cfg = clone["inference"]
        inference_cfg.pop("system_prompt", None)
        inference_cfg.pop("systemPrompt", None)
    clone.pop("data", None)
    return clone


def _compute_merge_key(parsed: Dict[str, object]) -> Optional[str]:
    spec = (parsed.get("spec") if isinstance(parsed, dict) else None) or {}
    if not isinstance(spec, dict):
        return None
    task_type = str(spec.get("taskType") or "").strip().lower()
    if not task_type:
        return None
    if task_type not in {"inference", "rag"}:
        return None
    try:
        sanitized = _sanitize_merge_spec(spec)
        return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None


class TaskRuntime:
    """In-memory task registry with FIFO-ready queue and dependency tracking."""

    def __init__(self, logger) -> None:
        self._logger = logger
        self._tasks: Dict[str, TaskRecord] = {}
        self._original_deps: Dict[str, Set[str]] = {}
        self._pending_deps: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._ready_queue: deque[str] = deque()
        self._ready_index: Set[str] = set()
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._merge_key_by_task: Dict[str, Optional[str]] = {}
        self._merge_buckets: Dict[str, List[str]] = defaultdict(list)
        self._merge_children_map: Dict[str, List[str]] = defaultdict(list)
        self._merge_parent_map: Dict[str, str] = {}

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

    # ------------------------------------------------------------------ #
    # Registration & submission
    # ------------------------------------------------------------------ #

    def register_yaml(self, yaml_text: str) -> List[Dict[str, object]]:
        """
        Parse YAML into logical tasks, register records, and enqueue any
        dependency-free tasks. Returns lightweight descriptors for the caller.
        """
        specs = parse_task_yaml(yaml_text)
        results: List[Dict[str, object]] = []
        new_ready = False

        with self._cv:
            for entry in specs:
                task_id = entry["task_id"]
                parsed = copy.deepcopy(entry["parsed"])
                depends_on = list(entry.get("depends_on") or [])
                original = set(depends_on)
                pending = {dep for dep in depends_on if dep not in self._completed}

                task_type = ((parsed.get("spec") or {}).get("taskType") or None)
                category = categorize_task_type(task_type)

                record = TaskRecord(
                    task_id=task_id,
                    raw_yaml=yaml_text,
                    parsed=parsed,
                    graph_node_name=entry.get("graph_node_name"),
                    load=int(entry.get("load", 0) or 0),
                    task_type=task_type,
                    category=category,
                )
                record.last_queue_ts = record.submitted_ts
                merge_key = _compute_merge_key(parsed)
                record.merge_key = merge_key
                self._merge_key_by_task[task_id] = merge_key

                self._tasks[task_id] = record
                self._original_deps[task_id] = original
                self._pending_deps[task_id] = pending
                self._failed.discard(task_id)
                if record.status == TaskStatus.DONE:
                    self._completed.add(task_id)
                else:
                    self._completed.discard(task_id)

                for dep in original:
                    self._dependents[dep].add(task_id)

                if not pending and record.status == TaskStatus.PENDING:
                    if self._enqueue_ready_locked(task_id):
                        new_ready = True

                results.append(
                    {
                        "task_id": task_id,
                        "depends_on": depends_on,
                        "graph_node": entry.get("graph_node_name"),
                    }
                )

            if new_ready:
                self._cv.notify_all()

        return results

    # ------------------------------------------------------------------ #
    # Ready queue helpers
    # ------------------------------------------------------------------ #

    def _enqueue_ready_locked(self, task_id: str, *, front: bool = False) -> bool:
        """Add a task to the ready queue if it is pending and not already queued."""
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return False
        if task_id in self._ready_index:
            return False
        if front:
            self._ready_queue.appendleft(task_id)
        else:
            self._ready_queue.append(task_id)
        self._ready_index.add(task_id)
        record.last_queue_ts = time.time()
        self._merge_bucket_add(task_id)
        return True

    def _pop_ready_locked(self) -> Optional[str]:
        while self._ready_queue:
            task_id = self._ready_queue.popleft()
            self._ready_index.discard(task_id)
            record = self._tasks.get(task_id)
            if not record or record.status != TaskStatus.PENDING:
                continue
            return task_id
        return None

    def _remove_from_ready_locked(self, task_id: str) -> None:
        if task_id not in self._ready_index:
            return
        self._ready_index.discard(task_id)
        self._ready_queue = deque(t for t in self._ready_queue if t != task_id)

    def _merge_bucket_add(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        bucket = self._merge_buckets.setdefault(key, [])
        if task_id not in bucket:
            bucket.append(task_id)

    def _merge_bucket_remove(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        bucket = self._merge_buckets.get(key)
        if not bucket:
            return
        try:
            bucket.remove(task_id)
        except ValueError:
            pass
        if not bucket:
            self._merge_buckets.pop(key, None)

    def next_ready(self, stop_event: threading.Event, timeout: float = 1.0) -> Optional[str]:
        """
        Block until a task is ready or stop_event is set.
        Returns a task_id or None when stopping.
        """
        with self._cv:
            while not stop_event.is_set():
                task_id = self._pop_ready_locked()
                if task_id:
                    return task_id
                self._cv.wait(timeout)
            return None

    def mark_pending(self, task_id: str) -> None:
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            record.status = TaskStatus.PENDING
            record.assigned_worker = None
            record.topic = None
            record.dispatched_ts = None
            record.started_ts = None
            record.finished_ts = None
            record.error = None

    def requeue(self, task_id: str, *, front: bool = False) -> bool:
        """Reinsert a task into the ready queue."""
        with self._cv:
            added = self._enqueue_ready_locked(task_id, front=front)
            if added:
                self._cv.notify_all()
            return added

    def plan_merge(self, task_id: str, max_batch_size: int) -> List[str]:
        if max_batch_size <= 1:
            return []
        with self._cv:
            return self._plan_merge_locked(task_id, max_batch_size)

    def _plan_merge_locked(self, task_id: str, max_batch_size: int) -> List[str]:
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return []
        if record.merge_key is None:
            return []
        if self._merge_children_map.get(task_id):
            return []
        bucket = list(self._merge_buckets.get(record.merge_key, []))
        if not bucket or len(bucket) <= 1:
            return []
        siblings: List[str] = []
        for candidate in bucket:
            if candidate == task_id:
                continue
            if len(siblings) >= max_batch_size - 1:
                break
            candidate_record = self._tasks.get(candidate)
            if not candidate_record or candidate_record.status != TaskStatus.PENDING:
                continue
            if candidate not in self._ready_index:
                continue
            siblings.append(candidate)
        if not siblings:
            return []

        merged_payload: List[Dict[str, Any]] = []
        for sibling in siblings:
            payload: Dict[str, Any] = {"task_id": sibling}
            child_record = self._tasks.get(sibling)
            if child_record:
                child_parsed = child_record.parsed or {}
                spec_copy = copy.deepcopy(child_parsed.get("spec") or {})
                if spec_copy:
                    payload["spec"] = spec_copy
                metadata_copy = copy.deepcopy(child_parsed.get("metadata")) if child_parsed.get("metadata") else None
                if metadata_copy:
                    payload["metadata"] = metadata_copy
            merged_payload.append(payload)

        record.merged_children = merged_payload
        self._merge_children_map[task_id] = list(siblings)
        for sibling in siblings:
            self._merge_parent_map[sibling] = task_id
            self._remove_from_ready_locked(sibling)
            self._merge_bucket_remove(sibling)
            sibling_record = self._tasks.get(sibling)
            if sibling_record:
                sibling_record.status = TaskStatus.DISPATCHED
                sibling_record.merged_parent_id = task_id
                sibling_record.assigned_worker = None
                sibling_record.merge_slice = None
        return siblings

    def release_merge(self, task_id: str) -> None:
        with self._cv:
            self._release_merge_locked(task_id)

    def _release_merge_locked(self, task_id: str) -> None:
        children = self._merge_children_map.pop(task_id, [])
        if not children:
            parent = self._tasks.get(task_id)
            if parent:
                parent.merged_children = None
            return
        parent = self._tasks.get(task_id)
        if parent:
            parent.merged_children = None
        for child_id in children:
            self._merge_parent_map.pop(child_id, None)
            child_record = self._tasks.get(child_id)
            if not child_record:
                continue
            if child_record.status == TaskStatus.DONE:
                continue
            child_record.status = TaskStatus.PENDING
            child_record.merged_parent_id = None
            child_record.merge_slice = None
            if child_id not in self._ready_index:
                self._enqueue_ready_locked(child_id, front=True)
            else:
                self._remove_from_ready_locked(child_id)
                self._enqueue_ready_locked(child_id, front=True)
        self._cv.notify_all()

    def _finalize_merged_child_success(
        self,
        child_id: str,
        worker_id: Optional[str],
        finished_ts: float,
        started_ts: Optional[float],
    ) -> List[str]:
        ready_children: List[str] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return ready_children
        child_record.status = TaskStatus.DONE
        child_record.error = None
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        if worker_id:
            child_record.assigned_worker = worker_id
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        self._completed.add(child_id)
        self._failed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)
        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is None:
                continue
            pending.discard(child_id)
            if not pending:
                dep_record = self._tasks.get(dep_id)
                if dep_record and dep_record.status == TaskStatus.PENDING:
                    if self._enqueue_ready_locked(dep_id):
                        ready_children.append(dep_id)
        return ready_children

    def _finalize_merged_child_failure(
        self,
        child_id: str,
        reason: str,
        finished_ts: float,
        started_ts: Optional[float],
    ) -> List[Tuple[str, str]]:
        impacted: List[Tuple[str, str]] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return impacted
        child_record.status = TaskStatus.FAILED
        child_record.error = reason
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        child_record.assigned_worker = None
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        self._failed.add(child_id)
        self._completed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)

        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is not None:
                pending.discard(child_id)
            dep_record = self._tasks.get(dep_id)
            if not dep_record or dep_record.status != TaskStatus.PENDING:
                continue
            fail_reason = f"Dependency {child_id} failed"
            dep_record.status = TaskStatus.FAILED
            dep_record.error = fail_reason
            dep_record.assigned_worker = None
            dep_record.finished_ts = time.time()
            self._pending_deps.pop(dep_id, None)
            self._remove_from_ready_locked(dep_id)
            impacted.append((dep_id, fail_reason))
        return impacted

    # ------------------------------------------------------------------ #
    # State updates (dispatch & events)
    # ------------------------------------------------------------------ #

    def mark_dispatched(self, task_id: str, worker_id: str) -> None:
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            record.status = TaskStatus.DISPATCHED
            record.assigned_worker = worker_id
            record.topic = "tasks"
            record.dispatched_ts = time.time()
            record.next_retry_at = None
            self._remove_from_ready_locked(task_id)
            self._merge_bucket_remove(task_id)

    def mark_started(self, task_id: str, worker_id: Optional[str], payload: Dict[str, object], ts: str) -> None:
        started_ts = parse_iso_ts(str(payload.get("started_at") or ts))
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            record.status = TaskStatus.DISPATCHED
            record.started_ts = started_ts
            if worker_id:
                record.assigned_worker = worker_id

    def mark_succeeded(
        self,
        task_id: str,
        worker_id: Optional[str],
        payload: Dict[str, object],
        ts: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Mark a task as completed and enqueue any dependents that have become ready.
        Returns (ready_children, merged_children_ids).
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None

        with self._cv:
            record = self._tasks.get(task_id)
            if record:
                record.status = TaskStatus.DONE
                record.error = None
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None

            self._completed.add(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            ready_children: List[str] = []
            merged_children_ids: List[str] = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is None:
                    continue
                pending.discard(task_id)
                if not pending:
                    child_record = self._tasks.get(child)
                    if child_record and child_record.status == TaskStatus.PENDING:
                        if self._enqueue_ready_locked(child):
                            ready_children.append(child)

            for merged_child in merged_children_ids:
                ready_children.extend(
                    self._finalize_merged_child_success(
                        merged_child,
                        worker_id,
                        finished_ts,
                        started_ts,
                    )
                )

            if ready_children:
                self._cv.notify_all()

            return ready_children, merged_children_ids

    def mark_failed(
        self,
        task_id: str,
        worker_id: Optional[str],
        payload: Dict[str, object],
        ts: str,
        *,
        error: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Mark a task as failed. Dependent tasks still waiting on this task are
        automatically failed to avoid running without prerequisites.

        Returns (impacted_dependents, merged_children_ids).
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        message = error or str(payload.get("error") or "task failed")

        with self._cv:
            record = self._tasks.get(task_id)
            if record:
                record.status = TaskStatus.FAILED
                record.error = message
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None

            self._failed.add(task_id)
            self._completed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            merged_children_ids = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            impacted: List[Tuple[str, str]] = []
            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is not None:
                    pending.discard(task_id)
                child_record = self._tasks.get(child)
                if not child_record or child_record.status != TaskStatus.PENDING:
                    continue
                reason = f"Dependency {task_id} failed"
                child_record.status = TaskStatus.FAILED
                child_record.error = reason
                child_record.assigned_worker = None
                child_record.finished_ts = time.time()
                self._pending_deps.pop(child, None)
                self._remove_from_ready_locked(child)
                impacted.append((child, reason))

            for merged_child in merged_children_ids:
                impacted.extend(
                    self._finalize_merged_child_failure(
                        merged_child,
                        f"Parent {task_id} failed",
                        finished_ts,
                        started_ts,
                    )
                )

            return impacted, merged_children_ids

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_record(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def describe_task(self, task_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            data = record.model_dump()
            data["depends_on"] = sorted(self._original_deps.get(task_id, set()))
            data["pending_dependencies"] = sorted(self._pending_deps.get(task_id, set()))
            data["dependents"] = sorted(self._dependents.get(task_id, set()))
            data["completed"] = task_id in self._completed
            data["failed"] = task_id in self._failed
            return data

    def list_tasks(self) -> List[Dict[str, object]]:
        with self._lock:
            return [
                {
                    **record.model_dump(),
                    "depends_on": sorted(self._original_deps.get(task_id, set())),
                    "pending_dependencies": sorted(self._pending_deps.get(task_id, set())),
                    "dependents": sorted(self._dependents.get(task_id, set())),
                    "completed": task_id in self._completed,
                    "failed": task_id in self._failed,
                }
                for task_id, record in self._tasks.items()
            ]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @property
    def tasks(self) -> Dict[str, TaskRecord]:
        return self._tasks

    def recover_tasks_for_worker(self, worker_id: str) -> List[str]:
        """
        Move DISPATCHED tasks assigned to a departed worker back to the ready queue.
        Returns affected task_ids.
        """
        recovered: List[str] = []
        with self._cv:
            for task_id, record in self._tasks.items():
                if record.assigned_worker != worker_id:
                    continue
                if record.status != TaskStatus.DISPATCHED:
                    continue
                self._release_merge_locked(task_id)
                record.status = TaskStatus.PENDING
                record.assigned_worker = None
                record.dispatched_ts = None
                record.started_ts = None
                record.finished_ts = None
                record.error = record.error or "Requeued after worker departure"
                if self._enqueue_ready_locked(task_id, front=True):
                    recovered.append(task_id)
            if recovered:
                self._cv.notify_all()
        return recovered

    def shutdown(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def ready_queue_length(self) -> int:
        with self._cv:
            return len(self._ready_queue)
