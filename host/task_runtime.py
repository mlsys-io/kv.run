from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .task_models import TaskRecord, TaskStatus, categorize_task_type
from .task_parser import parse_task_yaml
from .utils import parse_iso_ts


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
    ) -> List[str]:
        """
        Mark a task as completed and enqueue any dependents that have become ready.
        Returns a list of task_ids that were newly enqueued.
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

            self._completed.add(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            ready_children: List[str] = []

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

            if ready_children:
                self._cv.notify_all()

            return ready_children

    def mark_failed(
        self,
        task_id: str,
        worker_id: Optional[str],
        payload: Dict[str, object],
        ts: str,
        *,
        error: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Mark a task as failed. Dependent tasks still waiting on this task are
        automatically failed to avoid running without prerequisites.

        Returns a list of (task_id, reason) for dependents that were auto-failed.
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

            self._failed.add(task_id)
            self._completed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)

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

            return impacted

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
