from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Iterable, Optional, Set, List

from .utils import parse_iso_ts
from .worker_registry import (
    ELASTIC_DISABLED_SET,
    get_elastic_disabled_workers,
    set_worker_elastic_state,
    get_worker_from_redis,
    sort_workers,
    Worker,
)


class ElasticCoordinator:
    """Tracks host-side elastic enable/disable flags for workers."""

    def __init__(
        self,
        rds,
        metrics_recorder,
        runtime,
        *,
        enabled: bool,
        auto_disable_idle_secs: int = 0,
        auto_enable_queue_threshold: int = 0,
        auto_disable_queue_max: int = 0,
        auto_poll_interval: int = 30,
        auto_toggle_cooldown: int = 120,
        min_active_workers: int = 1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._rds = rds
        self._metrics = metrics_recorder
        self._runtime = runtime
        self._enabled = enabled
        self._logger = logger or logging.getLogger("elastic")
        self._lock = threading.RLock()
        self._disabled: Set[str] = set()
        self._current_status: Dict[str, str] = {}
        self._last_running_ts: Dict[str, float] = {}
        self._last_toggle_ts: Dict[str, float] = {}
        self._auto_disable_idle_secs = max(0, int(auto_disable_idle_secs))
        self._auto_enable_queue_threshold = max(0, int(auto_enable_queue_threshold))
        self._auto_disable_queue_max = max(0, int(auto_disable_queue_max))
        self._auto_poll_interval = max(5, int(auto_poll_interval))
        self._auto_toggle_cooldown = max(30, int(auto_toggle_cooldown))
        self._min_active_workers = max(0, int(min_active_workers))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if enabled:
            self._bootstrap()
            self._maybe_start_manager()

    def _bootstrap(self) -> None:
        disabled = get_elastic_disabled_workers(self._rds)
        with self._lock:
            self._disabled = set(disabled)
        for worker_id in disabled:
            self._metrics.set_worker_elastic_disabled(worker_id, True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disabled_workers(self) -> Set[str]:
        with self._lock:
            return set(self._disabled)

    def is_disabled(self, worker_id: Optional[str]) -> bool:
        if not worker_id:
            return False
        with self._lock:
            return worker_id in self._disabled

    def set_worker_state(self, worker_id: str, *, enabled: bool, source: str = "manual") -> None:
        if not self._enabled:
            raise RuntimeError("Elastic scaling control is disabled")
        normalized = worker_id.strip()
        if not normalized:
            raise ValueError("worker_id must be non-empty")
        with self._lock:
            currently_disabled = normalized in self._disabled
        if enabled and not currently_disabled:
            return
        if not enabled and currently_disabled:
            return
        set_worker_elastic_state(self._rds, normalized, enabled=enabled)
        with self._lock:
            if enabled:
                self._disabled.discard(normalized)
            else:
                self._disabled.add(normalized)
            self._last_toggle_ts[normalized] = time.time()
        self._metrics.set_worker_elastic_disabled(normalized, not enabled)
        self._logger.info(
            "Elastic coordinator %s worker %s via %s",
            "enabled" if enabled else "disabled",
            normalized,
            source,
        )

    def refresh(self) -> None:
        """Reload the disabled worker list from Redis."""
        if not self._enabled:
            return
        disabled = get_elastic_disabled_workers(self._rds)
        with self._lock:
            self._disabled = set(disabled)
        for worker_id in disabled:
            self._metrics.set_worker_elastic_disabled(worker_id, True)

    def include_disabled(self, collection: Iterable[str]) -> Set[str]:
        base = self.disabled_workers()
        for value in collection:
            if value:
                base.add(value)
        return base

    def start(self) -> None:
        if not self._enabled:
            return
        self._maybe_start_manager()

    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout=2.0)

    def record_worker_event(self, event) -> None:
        if not self._enabled:
            return
        worker_id = (event.worker_id or "").strip()
        if not worker_id:
            return
        ts = self._parse_ts(event.ts)
        with self._lock:
            if event.type == "REGISTER":
                self._current_status[worker_id] = event.status or "IDLE"
                self._last_running_ts.setdefault(worker_id, ts)
            elif event.type == "UNREGISTER":
                self._current_status.pop(worker_id, None)
                self._last_running_ts.pop(worker_id, None)
                self._last_toggle_ts.pop(worker_id, None)
                self._disabled.discard(worker_id)
                self._metrics.set_worker_elastic_disabled(worker_id, False)
            elif event.type == "STATUS":
                status = (event.status or "").upper()
                self._current_status[worker_id] = status
                if status == "RUNNING":
                    self._last_running_ts[worker_id] = ts
            elif event.type == "HEARTBEAT":
                pass

    def _maybe_start_manager(self) -> None:
        if (
            self._thread is not None
            or (self._auto_disable_idle_secs <= 0 and self._auto_enable_queue_threshold <= 0)
        ):
            return
        self._thread = threading.Thread(target=self._manager_loop, name="elastic-manager", daemon=True)
        self._thread.start()

    def _manager_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._auto_manage()
            except Exception as exc:  # pragma: no cover
                self._logger.debug("Elastic manager loop error: %s", exc)
            self._stop_event.wait(self._auto_poll_interval)

    def _auto_manage(self) -> None:
        if self._auto_disable_idle_secs > 0:
            self._auto_disable_idle_workers()
        if self._auto_enable_queue_threshold > 0:
            self._auto_enable_disabled_workers()

    def _auto_disable_idle_workers(self) -> None:
        queue_len = self._runtime.ready_queue_length()
        if queue_len > self._auto_disable_queue_max:
            return
        now = time.time()
        candidates: List[str] = []
        with self._lock:
            active_workers = [w for w in self._current_status if w not in self._disabled]
            for worker_id in active_workers:
                status = self._current_status.get(worker_id)
                if status != "IDLE":
                    continue
                last_run = self._last_running_ts.get(worker_id)
                if last_run is None:
                    continue
                if now - last_run < self._auto_disable_idle_secs:
                    continue
                last_toggle = self._last_toggle_ts.get(worker_id, 0.0)
                if now - last_toggle < self._auto_toggle_cooldown:
                    continue
                candidates.append(worker_id)

        if not candidates:
            return

        for worker_id in candidates:
            with self._lock:
                active_workers = [w for w in self._current_status if w not in self._disabled]
            if len(active_workers) <= max(1, self._min_active_workers):
                break
            try:
                self.set_worker_state(worker_id, enabled=False, source="auto-idle")
            except Exception as exc:  # pragma: no cover
                self._logger.debug("Failed to auto-disable worker %s: %s", worker_id, exc)

    def _auto_enable_disabled_workers(self) -> None:
        queue_len = self._runtime.ready_queue_length()
        if queue_len < self._auto_enable_queue_threshold:
            return
        now = time.time()
        with self._lock:
            disabled_ids = [w for w in self._disabled]

        candidates: List[Worker] = []
        for worker_id in disabled_ids:
            last_toggle = self._last_toggle_ts.get(worker_id, 0.0)
            if now - last_toggle < self._auto_toggle_cooldown:
                continue
            worker = get_worker_from_redis(self._rds, worker_id)
            if not worker:
                continue
            if worker.status and worker.status.upper() not in {"IDLE", "UNKNOWN"}:
                continue
            candidates.append(worker)

        if not candidates:
            return

        ordered = sort_workers(candidates)
        for worker in ordered:
            worker_id = worker.worker_id
            last_toggle = self._last_toggle_ts.get(worker_id, 0.0)
            if now - last_toggle < self._auto_toggle_cooldown:
                continue
            try:
                self.set_worker_state(worker_id, enabled=True, source="auto-queue")
            except Exception as exc:  # pragma: no cover
                self._logger.debug("Failed to auto-enable worker %s: %s", worker_id, exc)
                continue
            break

    def _parse_ts(self, value: Optional[str]) -> float:
        if not value:
            return time.time()
        try:
            return parse_iso_ts(value)
        except Exception:
            return time.time()


def is_elastic_enabled_env(value: bool) -> bool:
    """Compatibility helper for unit tests to inspect coordinator state."""
    return value
