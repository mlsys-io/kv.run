"""Retry queue management for orchestrator tasks."""

from __future__ import annotations

import heapq
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class RetryQueueManager:
    """Manage delayed retries for orchestrator tasks."""

    def __init__(
        self,
        tasks: Dict[str, Any],
        tasks_lock: threading.RLock,
        task_store,
        logger,
        task_status,
        base_delay_sec: float,
        max_delay_sec: float,
    ) -> None:
        self._tasks = tasks
        self._tasks_lock = tasks_lock
        self._task_store = task_store
        self._logger = logger
        self._task_status = task_status
        self._base_delay_sec = float(base_delay_sec)
        self._max_delay_sec = float(max_delay_sec)

        self._queue: List[tuple[float, str]] = []
        self._index: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._event = threading.Event()

    # ---------- Internal helpers ----------
    @staticmethod
    def _iso_from_ts(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _calc_retry_delay(self, rec) -> float:
        retries = max(getattr(rec, "retries", 0), 0)
        delay = self._base_delay_sec * (2 ** retries)
        return float(min(delay, self._max_delay_sec))

    # ---------- Public API ----------
    @property
    def event(self) -> threading.Event:
        return self._event

    def schedule(
        self,
        task_id: str,
        rec,
        reason: str,
        delay_sec: Optional[float] = None,
        exclude_worker_id: Optional[str] = None,
    ) -> None:
        delay = self._calc_retry_delay(rec) if delay_sec is None else max(float(delay_sec), 0.0)
        eta = time.time() + delay
        with self._tasks_lock:
            rec.status = self._task_status.WAITING
            rec.error = reason
            rec.next_retry_at = self._iso_from_ts(eta)
            if exclude_worker_id is not None:
                rec.last_failed_worker = exclude_worker_id
            rec.assigned_worker = None
            rec.topic = None
        with self._lock:
            self._index[task_id] = eta
            heapq.heappush(self._queue, (eta, task_id))
            self._event.set()

    def log_snapshot(self) -> None:
        try:
            with self._tasks_lock:
                total = len(self._tasks)
                status_counts = Counter(rec.status for rec in self._tasks.values())
                waiting_ts = [
                    rec.next_retry_at
                    for rec in self._tasks.values()
                    if rec.status == self._task_status.WAITING and rec.next_retry_at
                ]
            with self._lock:
                waiting_queue_len = len(self._queue)
                next_wait_eta = self._queue[0][0] if self._queue else None
            pending_tasks = len(self._task_store.list_waiting_tasks())

            next_retry_iso = min(waiting_ts) if waiting_ts else None
            next_queue_iso = self._iso_from_ts(next_wait_eta) if next_wait_eta else None

            self._logger.info(
                "Queue snapshot: total=%d statuses=%s pending_store=%d waiting_queue=%d next_retry=%s next_queue_eta=%s",
                total,
                dict(status_counts),
                pending_tasks,
                waiting_queue_len,
                next_retry_iso,
                next_queue_iso,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            self._logger.debug("Queue snapshot logging failed: %s", exc)

    def run_loop(self, dispatch_fn: Callable[[str, dict, Any, Optional[str]], None], default_wait: float = 5.0) -> None:
        while True:
            now = time.time()
            to_dispatch: Optional[str] = None
            eta = None
            with self._lock:
                if self._queue:
                    eta, candidate = self._queue[0]
                    if eta <= now:
                        heapq.heappop(self._queue)
                        current = self._index.get(candidate)
                        if current is None or abs(current - eta) > 1e-6:
                            continue
                        del self._index[candidate]
                        to_dispatch = candidate
                else:
                    eta = None

            if to_dispatch:
                parsed = self._task_store.get_parsed(to_dispatch)
                with self._tasks_lock:
                    rec = self._tasks.get(to_dispatch)
                    if not rec or not parsed:
                        continue
                    rec.status = self._task_status.PENDING
                    rec.next_retry_at = None
                    exclude_id = rec.last_failed_worker
                    rec.assigned_worker = None
                    rec.topic = None
                try:
                    dispatch_fn(to_dispatch, parsed, rec, exclude_id)
                    if rec.status == self._task_status.DISPATCHED:
                        with self._tasks_lock:
                            rec.last_failed_worker = None
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.warning("Retry dispatch crashed for %s: %s", to_dispatch, exc)
                continue

            if eta is None:
                self._event.wait(timeout=default_wait)
                self._event.clear()
            else:
                timeout = min(max(eta - time.time(), 0.1), default_wait)
                triggered = self._event.wait(timeout=timeout)
                if triggered:
                    self._event.clear()
