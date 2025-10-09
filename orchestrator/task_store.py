from __future__ import annotations

import os
import copy
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from worker_cls import Worker, get_worker_from_redis, is_stale_by_redis
from scheduler import idle_satisfying_pool, select_workers_for_task
from utils import safe_get

from load_func import DEFAULT_TASK_LOAD
from parser import ParsedTaskSpec, parse_task_specs


# -----------------------------
# Pool entries and dispatch plan
# -----------------------------

@dataclass
class PoolEntry:
    task_id: str
    task: Dict[str, Any]
    record: Any
    exclude_worker_id: Optional[str] = None
    enqueued_at: float = field(default_factory=time.time)

    def slo_progress(self, now: Optional[float] = None) -> float:
        """Return fraction of SLO window already consumed (0.0 if no/invalid SLO)."""
        slo = getattr(self.record, "slo_seconds", None)
        if not slo or slo <= 0:
            return 0.0
        submitted_ts = getattr(self.record, "submitted_ts", None) or self.enqueued_at
        elapsed = max(0.0, (now or time.time()) - submitted_ts)
        return 0.0 if slo <= 0 else elapsed / slo


@dataclass
class DispatchPlan:
    task_id: str
    parsed: Dict[str, Any]
    record: Any
    exclude_worker_id: Optional[str] = None
    preferred_worker_id: Optional[str] = None


# -----------------------------
# In-memory pool with SLO gating
# -----------------------------

class TaskPool:
    """
    A small in-memory pool. Entries are drained when either:
      - the pool size reaches batch_size, or
      - any entry's SLO progress crosses the configured threshold.
    """

    def __init__(self, batch_size: int, slo_fraction: float) -> None:
        self._batch_size = max(1, int(batch_size))
        self._slo_fraction = max(0.0, float(slo_fraction))
        self._entries: Dict[str, PoolEntry] = {}
        self._lock: Optional[threading.RLock] = None

    @property
    def _thread_lock(self) -> threading.RLock:
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock

    def add(self, entry: PoolEntry) -> List[PoolEntry]:
        """Add an entry and return a batch to dispatch if threshold met."""
        with self._thread_lock:
            self._entries[entry.task_id] = entry
            return self._flush_if_needed_locked()

    def requeue(self, entries: List[PoolEntry]) -> None:
        """Put back deferred entries (e.g., deps not ready yet)."""
        if not entries:
            return
        with self._thread_lock:
            for entry in entries:
                self._entries[entry.task_id] = entry

    def pop_due(self) -> List[PoolEntry]:
        """
        Pop a due batch based on SLO threshold.
        Returns at most batch_size entries to avoid burst dispatch.
        """
        with self._thread_lock:
            if not self._entries:
                return []
            now = time.time()
            if any(e.slo_progress(now) >= self._slo_fraction for e in self._entries.values()):
                return self._take_n_locked(self._batch_size)
            return []

    def clear_task(self, task_id: str) -> None:
        """Remove a single task from the pool (e.g., after dispatch or cancel)."""
        with self._thread_lock:
            self._entries.pop(task_id, None)

    def has_entries(self) -> bool:
        with self._thread_lock:
            return bool(self._entries)

    # -------- internals --------

    def _flush_if_needed_locked(self) -> List[PoolEntry]:
        """Return a batch if pool-size or SLO threshold is met."""
        if len(self._entries) >= self._batch_size:
            return self._take_n_locked(self._batch_size)

        now = time.time()
        if any(e.slo_progress(now) >= self._slo_fraction for e in self._entries.values()):
            return self._take_n_locked(self._batch_size)

        return []

    def _take_n_locked(self, n: int) -> List[PoolEntry]:
        """
        Take at most n entries out of the pool, ordering by:
          1) higher SLO progress first,
          2) earlier submitted_ts next (FIFO within same progress).
        """
        items = list(self._entries.values())
        # Rank by urgency then FIFO; this improves fairness under pressure.
        now = time.time()
        items.sort(
            key=lambda e: (e.slo_progress(now), -(getattr(e.record, "submitted_ts", e.enqueued_at))),
            reverse=True,
        )
        batch = items[:n]
        for b in batch:
            self._entries.pop(b.task_id, None)
        return batch

    def time_until_threshold(self, slo_fraction: float) -> Optional[float]:
        """
        Return the minimum time (seconds) until any entry reaches the given SLO fraction.
        0.0 is returned if at least one entry already meets/exceeds the fraction.
        """
        with self._thread_lock:
            if not self._entries:
                return None
            now = time.time()
            min_delay: Optional[float] = None
            for entry in self._entries.values():
                slo = getattr(entry.record, "slo_seconds", None)
                if not slo or slo <= 0:
                    continue
                target_ts = (getattr(entry.record, "submitted_ts", None) or entry.enqueued_at) + (
                    float(slo) * float(slo_fraction)
                )
                delay = target_ts - now
                if delay <= 0:
                    return 0.0
                if min_delay is None or delay < min_delay:
                    min_delay = delay
            return min_delay


# -----------------------------
# Pool manager: timers, deps, co-location
# -----------------------------

class TaskPoolManager:
    """
    Bridges the TaskPool with actual dispatch:
      - Enqueue entries and schedule flushes.
      - On flush, filter for dependency and retry readiness.
      - Optimize the batch (co-location, model stickiness), then call dispatch_fn.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        slo_fraction: float,
        redis_client,
        dispatch_fn: Callable[[str, Dict[str, Any], Any, Optional[str], Optional[str]], None],
        logger,
        task_lookup: Callable[[str], Any],
        tasks_lock,
        dependency_resolver: Callable[[str], List[str]],
        pending_status: str,
        done_status: str,
    ) -> None:
        self._pool = TaskPool(batch_size, slo_fraction)
        self._dispatch_fn = dispatch_fn
        self._logger = logger
        self._rds = redis_client
        self._task_lookup = task_lookup
        self._tasks_lock = tasks_lock
        self._dependency_resolver = dependency_resolver
        self._pending_status = pending_status
        self._done_status = done_status

        self._model_worker_cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

        self._flush_lock = threading.RLock()
        self._flush_timer: Optional[threading.Timer] = None
        self._slo_fraction = float(slo_fraction)
        self._stickiness_ttl = max(0.0, float(os.getenv("MODEL_STICKINESS_TTL_SEC", "180")))
        self._stickiness_capacity = max(1, int(os.getenv("MODEL_STICKINESS_MAX_ENTRIES", "256")))

    # ---- public API ----

    def enqueue(
        self,
        task_id: str,
        task: Dict[str, Any],
        record: Any,
        *,
        exclude_worker_id: Optional[str] = None,
    ) -> None:
        batch = self._pool.add(
            PoolEntry(task_id=task_id, task=task, record=record, exclude_worker_id=exclude_worker_id)
        )
        if batch:
            self._dispatch_batch(batch)
        self._ensure_flush_timer()

    def flush_due(self) -> None:
        # Clear timer ref first so a new one can be armed if needed.
        with self._flush_lock:
            self._flush_timer = None

        batch = self._pool.pop_due()
        if batch:
            self._dispatch_batch(batch)
        else:
            self._ensure_flush_timer()

    def clear_task(self, task_id: str) -> None:
        self._pool.clear_task(task_id)
        if not self._pool.has_entries():
            self._cancel_flush_timer()

    def forget_worker(self, worker_id: Optional[str]) -> None:
        if not worker_id:
            return
        stale = [
            model
            for model, (wid, _) in list(self._model_worker_cache.items())
            if wid == worker_id
        ]
        for model in stale:
            self._model_worker_cache.pop(model, None)

    def record_assignment(self, task: Dict[str, Any], worker_id: str) -> None:
        """Record a model→worker mapping for inference tasks (stickiness)."""
        if not worker_id or not self._is_inference_task(task):
            return
        model = self._extract_model_name(task)
        if not model:
            return
        if self._stickiness_ttl <= 0:
            return
        now = time.time()
        # Remove stale entries before inserting the new one.
        self._prune_model_cache(now)
        self._model_worker_cache[model] = (worker_id, now)
        self._model_worker_cache.move_to_end(model)
        while len(self._model_worker_cache) > self._stickiness_capacity:
            self._model_worker_cache.popitem(last=False)

    # ---- internal flow ----

    def _dispatch_batch(self, entries: List[PoolEntry]) -> None:
        """Filter, optimize and dispatch the given batch; requeue deferred entries."""
        ready_entries: List[PoolEntry] = []
        deferred: List[PoolEntry] = []

        for entry in entries:
            if not self._is_dependency_satisfied(entry.task_id):
                deferred.append(entry)
                continue
            if not self._is_retry_due(entry.record):
                deferred.append(entry)
                continue
            if getattr(entry.record, "status", None) != self._pending_status:
                deferred.append(entry)
                continue
            ready_entries.append(entry)

        if deferred:
            self._pool.requeue(deferred)

        if not ready_entries:
            self._ensure_flush_timer()
            return

        plans = self._optimize_batch(ready_entries)

        for plan in plans:
            try:
                self._dispatch_fn(
                    plan.task_id,
                    plan.parsed,
                    plan.record,
                    plan.exclude_worker_id,
                    plan.preferred_worker_id,
                )
            except Exception as exc:  # pragma: no cover
                self._logger.warning("Dispatch planning failed for %s: %s", plan.task_id, exc)

        self._ensure_flush_timer()

    def _optimize_batch(self, entries: List[PoolEntry]) -> List[DispatchPlan]:
        """
        Build dispatch plans and try to:
          - precompute eligible pools (with exclude ids),
          - co-locate when workers are tight (pair tasks that share workers),
          - apply model stickiness (reuse a cached IDLE worker for the same model).
        """
        if not entries:
            return []

        plans: List[DispatchPlan] = []
        pool_map: Dict[str, List[Worker]] = {}
        model_counts: Counter[str] = Counter()
        inference_entries: List[PoolEntry] = []

        # 1) Basic plan assembly + per-task eligible pools
        for entry in entries:
            exclude_ids: Set[str] = set()
            if entry.exclude_worker_id:
                exclude_ids.add(entry.exclude_worker_id)

            pool = idle_satisfying_pool(self._rds, entry.task, exclude_ids=exclude_ids)
            pool_map[entry.task_id] = pool

            plans.append(
                DispatchPlan(
                    task_id=entry.task_id,
                    parsed=entry.task,
                    record=entry.record,
                    exclude_worker_id=entry.exclude_worker_id,
                )
            )

            if self._is_inference_task(entry.task):
                inference_entries.append(entry)
                model_name = self._extract_model_name(entry.task)
                if model_name:
                    model_counts[model_name] += 1

        # 2) Co-location when workers are tight: pair tasks sharing common workers
        available_workers = {w.worker_id for workers in pool_map.values() for w in workers}
        worker_shortage = len(available_workers) < len(entries)

        if worker_shortage and len(inference_entries) >= 2:
            ordered = sorted(inference_entries, key=lambda e: getattr(e.record, "submitted_ts", 0.0))
            paired: Set[str] = set()
            for idx, entry in enumerate(ordered):
                if entry.task_id in paired:
                    continue
                pool_a = pool_map.get(entry.task_id, [])
                if not pool_a:
                    continue
                cand_a = {w.worker_id: w for w in pool_a}
                for other in ordered[idx + 1 :]:
                    if other.task_id in paired:
                        continue
                    pool_b = pool_map.get(other.task_id, [])
                    if not pool_b:
                        continue
                    cand_b = {w.worker_id: w for w in pool_b}
                    common_ids = list(set(cand_a.keys()) & set(cand_b.keys()))
                    if not common_ids:
                        continue
                    combined: Dict[str, Worker] = dict(cand_a)
                    combined.update(cand_b)
                    candidates = [combined[cid] for cid in common_ids]

                    chosen = select_workers_for_task(
                        candidates,
                        shard_count=1,
                        prefer_best=True,
                        task_load=max(getattr(entry.record, "load", 0), getattr(other.record, "load", 0)),
                    )
                    if not chosen:
                        continue

                    target_worker = chosen[0].worker_id
                    self._set_plan_preference(plans, entry.task_id, target_worker)
                    self._set_plan_preference(plans, other.task_id, target_worker)
                    paired.add(entry.task_id)
                    paired.add(other.task_id)
                    break

        # 3) Model stickiness: reuse previous IDLE worker for popular models
        for plan in plans:
            if not self._is_inference_task(plan.parsed):
                continue
            if plan.preferred_worker_id:
                continue
            model_name = self._extract_model_name(plan.parsed)
            if not model_name:
                continue
            if model_counts.get(model_name, 0) < 2:
                continue
            cached_worker = self._pop_cached_worker(model_name)
            if not cached_worker or cached_worker == plan.exclude_worker_id:
                continue
            pool = pool_map.get(plan.task_id, [])
            if any(w.worker_id == cached_worker for w in pool):
                plan.preferred_worker_id = cached_worker

        return plans

    # ---- dependency / retry / timers ----

    def _is_dependency_satisfied(self, task_id: str) -> bool:
        deps = self._dependency_resolver(task_id) or []
        if not deps:
            return True
        with self._tasks_lock:
            for dep_id in deps:
                dep = self._task_lookup(dep_id)
                if not dep or getattr(dep, "status", None) != self._done_status:
                    return False
        return True

    @staticmethod
    def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            cleaned = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _is_retry_due(self, record: Any) -> bool:
        eta = self._parse_iso_ts(getattr(record, "next_retry_at", None))
        if not eta:
            return True
        return eta <= datetime.now(timezone.utc)

    def _next_timer_delay(self) -> float:
        """
        Compute a near-optimal wake-up delay:
          - If any entry already meets SLO fraction, return a short delay (0).
          - Else compute the minimum time required for any entry to reach the threshold.
          - Fallback to 1.0s if nothing is actionable.
        """
        # Read entries under pool lock
        if not self._pool.has_entries():
            return 0.0

        delay = self._pool.time_until_threshold(self._slo_fraction)
        if delay is None:
            # No SLO-bound entries; probe at a low frequency.
            return 1.0
        # Clamp to sane bounds to avoid super short busy loops or excessively long sleeps.
        return max(0.0, min(delay, 30.0))

    def _ensure_flush_timer(self, delay: float = None) -> None:
        """Ensure a single timer is armed to wake up the pool scanner."""
        if not self._pool.has_entries():
            self._cancel_flush_timer()
            return
        if delay is None:
            delay = self._next_timer_delay()

        with self._flush_lock:
            if self._flush_timer and self._flush_timer.is_alive():
                return
            timer = threading.Timer(delay, self.flush_due)
            timer.daemon = True
            self._flush_timer = timer
            timer.start()

    def _cancel_flush_timer(self) -> None:
        with self._flush_lock:
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None

    # ---- helpers ----

    def _set_plan_preference(self, plans: List[DispatchPlan], task_id: str, worker_id: str) -> None:
        for plan in plans:
            if plan.task_id == task_id:
                plan.preferred_worker_id = worker_id
                return

    @staticmethod
    def _is_inference_task(task: Dict[str, Any]) -> bool:
        return str(safe_get(task, "spec.taskType") or "").lower() == "inference"

    @staticmethod
    def _extract_model_name(task: Dict[str, Any]) -> Optional[str]:
        candidates = (
            safe_get(task, "spec.model.source.identifier"),
            safe_get(task, "spec.model.name"),
            safe_get(task, "metadata.annotations.model"),
        )
        for cand in candidates:
            if cand:
                return str(cand)
        return None

    def _pop_cached_worker(self, model: Optional[str]) -> Optional[str]:
        if not model:
            return None
        now = time.time()
        self._prune_model_cache(now)
        cached = self._model_worker_cache.get(model)
        if not cached:
            return None
        wid, _ = cached
        worker = get_worker_from_redis(self._rds, wid)
        if not worker or is_stale_by_redis(self._rds, wid):
            self._model_worker_cache.pop(model, None)
            return None
        if worker.status != "IDLE":
            return None
        return wid

    def _prune_model_cache(self, now: Optional[float] = None) -> None:
        """Drop expired stickiness entries to avoid stale affinity."""
        if self._stickiness_ttl <= 0 or not self._model_worker_cache:
            return
        now = now or time.time()
        expired = [
            key for key, (_, ts) in self._model_worker_cache.items() if (now - ts) > self._stickiness_ttl
        ]
        for key in expired:
            self._model_worker_cache.pop(key, None)


# -----------------------------
# TaskStore: registration & queries
# -----------------------------

class TaskStore:
    """In-memory registry for parsed tasks and dependency relations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._parsed: Dict[str, Dict[str, Any]] = {}
        self._depends: Dict[str, Set[str]] = {}
        self._released: Set[str] = set()
        self._load: Dict[str, int] = {}
        self._slo: Dict[str, Optional[float]] = {}
        self._pool_manager: Optional[TaskPoolManager] = None
        self._dead_letters: List[Dict[str, Any]] = []
        self._dlq_limit = max(1, int(os.getenv("TASK_DLQ_LIMIT", "500")))

    # -------------------------
    # Pool management
    # -------------------------
    def configure_pool(
        self,
        *,
        batch_size: int,
        slo_fraction: float,
        redis_client,
        dispatch_fn: Callable[[str, Dict[str, Any], Any, Optional[str], Optional[str]], None],
        logger,
        task_lookup: Callable[[str], Any],
        tasks_lock,
        pending_status: str,
        done_status: str,
    ) -> None:
        self._pool_manager = TaskPoolManager(
            batch_size=batch_size,
            slo_fraction=slo_fraction,
            redis_client=redis_client,
            dispatch_fn=dispatch_fn,
            logger=logger,
            task_lookup=task_lookup,
            tasks_lock=tasks_lock,
            dependency_resolver=self.get_dependencies,
            pending_status=pending_status,
            done_status=done_status,
        )

    def enqueue_for_dispatch(
        self,
        task_id: str,
        task: Dict[str, Any],
        record: Any,
        exclude_worker_id: Optional[str] = None,
    ) -> None:
        if not self._pool_manager:
            raise RuntimeError("Task pool manager is not configured")
        self._pool_manager.enqueue(task_id, task, record, exclude_worker_id=exclude_worker_id)

    def clear_from_pool(self, task_id: str) -> None:
        if self._pool_manager:
            self._pool_manager.clear_task(task_id)

    def forget_worker(self, worker_id: Optional[str]) -> None:
        if self._pool_manager:
            self._pool_manager.forget_worker(worker_id)

    def record_assignment(self, task: Dict[str, Any], worker_id: str) -> None:
        if self._pool_manager:
            self._pool_manager.record_assignment(task, worker_id)

    def flush_pool(self) -> None:
        if self._pool_manager:
            self._pool_manager.flush_due()

    # -------------------------
    # Registration / Parsing
    # -------------------------
    def parse_and_register(self, yaml_text: str) -> List[Dict[str, Any]]:
        specs = parse_task_specs(yaml_text)
        results: List[Dict[str, Any]] = []
        local_ids: Dict[str, str] = {}

        for spec in specs:
            task_id = str(uuid.uuid4())
            depends_on = [
                local_ids.get(dep.strip(), dep.strip())
                for dep in spec.depends_on
                if dep and dep.strip()
            ]

            parsed_spec = copy.deepcopy(spec.spec)
            with self._lock:
                self._parsed[task_id] = parsed_spec
                self._depends[task_id] = set(depends_on)
                self._load[task_id] = int(spec.load)
                self._slo[task_id] = spec.slo_seconds

            results.append(
                {
                    "task_id": task_id,
                    "parsed": copy.deepcopy(parsed_spec),
                    "depends_on": depends_on,
                    "graph_node_name": spec.graph_node_name,
                    "load": spec.load,
                    "slo_seconds": spec.slo_seconds,
                }
            )

            if spec.local_name:
                local_ids[spec.local_name] = task_id

        return results

    def mark_released(self, task_id: str) -> None:
        with self._lock:
            self._released.add(task_id)

    # -------------------------
    # Queries
    # -------------------------
    def get_parsed(self, task_id: str) -> Dict[str, Any] | None:
        with self._lock:
            return self._parsed.get(task_id)

    def get_dependencies(self, task_id: str) -> List[str]:
        with self._lock:
            return list(self._depends.get(task_id, set()))

    def get_load(self, task_id: str) -> int:
        with self._lock:
            return int(self._load.get(task_id, DEFAULT_TASK_LOAD))

    def get_slo_seconds(self, task_id: str) -> Optional[float]:
        with self._lock:
            return self._slo.get(task_id)

    def list_waiting_tasks(self) -> List[str]:
        with self._lock:
            return [tid for tid in self._parsed.keys() if tid not in self._released]

    def ready_to_dispatch(self, is_dep_satisfied: Callable[[str], bool]) -> List[str]:
        ready: List[str] = []
        with self._lock:
            for tid, deps in self._depends.items():
                if tid in self._released:
                    continue
                if all(is_dep_satisfied(did) for did in deps):
                    ready.append(tid)
        return ready

    def record_dead_letter(self, entry: Dict[str, Any]) -> None:
        payload = dict(entry)
        payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._dead_letters.append(payload)
            # Keep the list bounded to avoid unbounded memory growth.
            if len(self._dead_letters) > self._dlq_limit:
                excess = len(self._dead_letters) - self._dlq_limit
                if excess > 0:
                    del self._dead_letters[:excess]

    def list_dead_letters(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(item) for item in self._dead_letters]
