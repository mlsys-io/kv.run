from __future__ import annotations

import os
import copy
import json
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from worker_cls import Worker, get_worker_from_redis, is_stale_by_redis
from scheduler import idle_satisfying_pool, select_workers_for_task
from utils import safe_get

from load_func import DEFAULT_TASK_LOAD
from parser import ParsedTaskSpec, parse_task_specs
from task import TaskStatus


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
        task_store,
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
        self._task_store = task_store
        self._tasks_lock = tasks_lock
        self._dependency_resolver = dependency_resolver
        self._pending_status = pending_status
        self._done_status = done_status

        self._model_worker_cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self._model_queue_counts: Counter[str] = Counter()
        self._model_queue_lock = threading.RLock()

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
        entry = PoolEntry(task_id=task_id, task=task, record=record, exclude_worker_id=exclude_worker_id)
        self._update_model_queue_counts([entry], delta=1)
        batch = self._pool.add(entry)
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

        self._update_model_queue_counts(ready_entries, delta=-1)
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

    def _update_model_queue_counts(self, entries: Iterable[PoolEntry], delta: int) -> None:
        if not entries or delta == 0:
            return
        with self._model_queue_lock:
            for entry in entries:
                if not self._is_inference_task(entry.task):
                    continue
                model = self._extract_model_name(entry.task)
                if not model:
                    continue
                new_value = self._model_queue_counts.get(model, 0) + delta
                if new_value <= 0:
                    self._model_queue_counts.pop(model, None)
                else:
                    self._model_queue_counts[model] = new_value

    def _snapshot_model_queue_counts(self) -> Counter[str]:
        with self._model_queue_lock:
            return Counter(self._model_queue_counts)

    def _apply_inference_merges(
        self,
        plans: List[DispatchPlan],
        pool_map: Dict[str, List[Worker]],
        inference_entries: List[PoolEntry],
    ) -> Tuple[List[DispatchPlan], List[PoolEntry]]:
        if not inference_entries:
            return plans, inference_entries

        plan_map: Dict[str, DispatchPlan] = {plan.task_id: plan for plan in plans}
        skip_ids: Set[str] = set()
        pending: Dict[str, PoolEntry] = {}

        for entry in inference_entries:
            if entry.task_id in skip_ids:
                continue
            plan = plan_map.get(entry.task_id)
            signature = self._merge_signature(plan.parsed if plan else entry.task)
            if not signature:
                continue
            partner = pending.get(signature)
            if partner and partner.task_id not in skip_ids:
                if self._merge_pair(partner, entry, plan_map, pool_map):
                    skip_ids.add(entry.task_id)
                    pending.pop(signature, None)
                    continue
            pending[signature] = entry

        if not skip_ids:
            return plans, inference_entries

        for sid in skip_ids:
            plan_map.pop(sid, None)
            pool_map.pop(sid, None)

        filtered_plans = [plan for plan in plans if plan.task_id not in skip_ids]
        remaining_entries = [entry for entry in inference_entries if entry.task_id not in skip_ids]
        return filtered_plans, remaining_entries

    def _merge_signature(self, task: Dict[str, Any]) -> Optional[str]:
        spec = copy.deepcopy(task.get("spec") or {})
        if not spec:
            return None
        spec.pop("data", None)
        try:
            return json.dumps(spec, sort_keys=True, default=str)
        except Exception:
            return None

    def _merge_pair(
        self,
        primary_entry: PoolEntry,
        secondary_entry: PoolEntry,
        plan_map: Dict[str, DispatchPlan],
        pool_map: Dict[str, List[Worker]],
    ) -> bool:
        primary_plan = plan_map.get(primary_entry.task_id)
        secondary_plan = plan_map.get(secondary_entry.task_id)
        if not primary_plan or not secondary_plan:
            return False

        primary_items = self._expand_data_items(primary_plan.parsed)
        secondary_items = self._expand_data_items(secondary_plan.parsed)
        if primary_items is None or secondary_items is None:
            return False
        if not primary_items or not secondary_items:
            return False

        combined_items = list(primary_items) + list(secondary_items)
        combined_spec = copy.deepcopy(primary_plan.parsed)
        combined_spec.setdefault("spec", {})["data"] = {"type": "list", "items": combined_items}

        slices: Dict[str, Tuple[int, int]] = {
            primary_entry.task_id: (0, len(primary_items)),
            secondary_entry.task_id: (len(primary_items), len(combined_items)),
        }

        if not self._update_merge_state(primary_entry, secondary_entry, combined_spec, slices):
            return False

        primary_plan.parsed = combined_spec
        primary_entry.task = combined_spec
        plan_map.pop(secondary_entry.task_id, None)
        pool_map.pop(secondary_entry.task_id, None)
        self._logger.info(
            "Merged inference tasks %s + %s into combined batch with %d inputs",
            primary_entry.task_id,
            secondary_entry.task_id,
            len(combined_items),
        )
        return True

    def _expand_data_items(self, task: Dict[str, Any]) -> Optional[List[Any]]:
        spec = (task or {}).get("spec") or {}
        data = spec.get("data") or {}
        dtype = str(data.get("type") or "list").lower()
        if dtype in {"", "list"}:
            items = data.get("items") or []
            if not isinstance(items, list):
                return None
            return list(items)
        if dtype == "dataset":
            return self._dataset_to_items(spec)
        return None

    def _dataset_to_items(self, spec: Dict[str, Any]) -> Optional[List[str]]:
        data = spec.get("data") or {}
        source = data.get("url")
        if not source:
            return None
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            self._logger.warning("Cannot merge dataset-based inference tasks: %s", exc)
            return None

        name = data.get("name")
        split = data.get("split", "train")
        try:
            dataset = load_dataset(source, name=name, split=split)
        except Exception as exc:
            self._logger.warning("Failed to load dataset %s for merge: %s", source, exc)
            return None

        if data.get("shuffle"):
            seed = int(data.get("seed", 42))
            buffer_size = data.get("buffer_size")
            try:
                if buffer_size is None:
                    dataset = dataset.shuffle(seed=seed)
                else:
                    dataset = dataset.shuffle(seed=seed, buffer_size=int(buffer_size))
            except Exception as exc:
                self._logger.warning("Dataset shuffle failed during merge: %s", exc)

        column = data.get("column", "text")
        if column not in dataset.column_names:
            self._logger.warning("Column %s not found when merging dataset inference tasks", column)
            return None

        try:
            return [str(value) for value in dataset[column]]
        except Exception as exc:
            self._logger.warning("Failed to extract dataset column %s for merge: %s", column, exc)
            return None

    def _update_merge_state(
        self,
        primary_entry: PoolEntry,
        secondary_entry: PoolEntry,
        combined_spec: Dict[str, Any],
        slices: Dict[str, Tuple[int, int]],
    ) -> bool:
        parent_id = primary_entry.task_id
        child_id = secondary_entry.task_id
        with self._tasks_lock:
            parent_rec = self._task_lookup(parent_id)
            child_rec = self._task_lookup(child_id)
            if not parent_rec or not child_rec:
                return False

            parent_children = list(parent_rec.merged_children or [])
            parent_children = [c for c in parent_children if c.get("task_id") != child_id]
            parent_children.append(
                {
                    "task_id": child_id,
                    "start": slices[child_id][0],
                    "end": slices[child_id][1],
                }
            )
            parent_rec.merged_children = parent_children
            parent_rec.merge_slice = {
                "start": slices[parent_id][0],
                "end": slices[parent_id][1],
            }
            parent_rec.parsed = copy.deepcopy(combined_spec)

            child_rec.status = TaskStatus.WAITING
            child_rec.error = f"Waiting on merged inference parent {parent_id}"
            child_rec.assigned_worker = None
            child_rec.next_retry_at = None
            child_rec.merged_parent_id = parent_id
            child_rec.merge_slice = {
                "start": slices[child_id][0],
                "end": slices[child_id][1],
            }
            child_rec.merged_children = None

        self._task_store.update_parsed(parent_id, combined_spec)
        self._task_store.register_merge(parent_id, slices)
        return True

    def _optimize_batch(self, entries: List[PoolEntry]) -> List[DispatchPlan]:
        """
        Build dispatch plans and try to:
          - precompute eligible pools (with exclude ids),
          - co-locate when workers are tight (pair tasks that share workers),
          - apply model stickiness (reuse a cached IDLE worker for the same model),
          - convey engine-retention hints when more work is queued for a model.
        """
        if not entries:
            return []

        plans: List[DispatchPlan] = []
        pool_map: Dict[str, List[Worker]] = {}
        inference_entries: List[PoolEntry] = []
        pending_counts = self._snapshot_model_queue_counts()

        # 1) Build dispatch plans and cache the eligible worker pools.
        for entry in entries:
            exclude_ids: Set[str] = {entry.exclude_worker_id} if entry.exclude_worker_id else set()
            pool = idle_satisfying_pool(self._rds, entry.task, exclude_ids=exclude_ids) or []
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

        plans, inference_entries = self._apply_inference_merges(plans, pool_map, inference_entries)

        # Keep worker pools that correspond to surviving plans only.
        if plans:
            active_ids = {plan.task_id for plan in plans}
            pool_map = {task_id: pool_map.get(task_id, []) for task_id in active_ids}

        model_counts: Counter[str] = Counter()
        for entry in inference_entries:
            model_name = self._extract_model_name(entry.task)
            if model_name:
                model_counts[model_name] += 1
        remaining_by_model = Counter(model_counts)

        # 2) Co-location when workers are tight: pair tasks sharing common workers
        available_workers = {w.worker_id for workers in pool_map.values() for w in workers}
        worker_shortage = len(available_workers) < len(pool_map)

        if worker_shortage and len(inference_entries) >= 2:
            ordered = sorted(inference_entries, key=lambda e: getattr(e.record, "submitted_ts", 0.0))
            paired: Set[str] = set()
            candidate_cache: Dict[str, Tuple[List[Worker], Dict[str, Worker]]] = {}
            for entry in inference_entries:
                task_pool = pool_map.get(entry.task_id, [])
                if task_pool:
                    candidate_cache[entry.task_id] = (task_pool, {w.worker_id: w for w in task_pool})

            for idx, entry in enumerate(ordered):
                task_id = entry.task_id
                if task_id in paired:
                    continue
                pool_a = candidate_cache.get(task_id)
                if not pool_a:
                    continue
                workers_a, cand_a = pool_a
                load_a = getattr(entry.record, "load", 0)

                for other in ordered[idx + 1 :]:
                    other_id = other.task_id
                    if other_id in paired:
                        continue
                    pool_b = candidate_cache.get(other_id)
                    if not pool_b:
                        continue
                    workers_b, cand_b = pool_b
                    common_ids = set(cand_a.keys()) & set(cand_b.keys())
                    if not common_ids:
                        continue

                    candidates: List[Worker] = []
                    seen: Set[str] = set()
                    for worker in workers_a:
                        worker_id = worker.worker_id
                        if worker_id in common_ids and worker_id not in seen:
                            candidates.append(worker)
                            seen.add(worker_id)
                    for worker in workers_b:
                        worker_id = worker.worker_id
                        if worker_id in common_ids and worker_id not in seen:
                            candidates.append(worker)
                            seen.add(worker_id)

                    if not candidates:
                        continue

                    task_load = max(load_a, getattr(other.record, "load", 0))
                    chosen = select_workers_for_task(
                        candidates,
                        shard_count=1,
                        prefer_best=True,
                        task_load=task_load,
                    )
                    if not chosen:
                        continue

                    target_worker = chosen[0].worker_id
                    self._set_plan_preference(plans, task_id, target_worker)
                    self._set_plan_preference(plans, other_id, target_worker)
                    paired.add(task_id)
                    paired.add(other_id)
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

        # 4) Host-driven engine retention: decide whether workers should keep the inference engine warm.
        for plan in plans:
            if not self._is_inference_task(plan.parsed):
                continue
            model_name = self._extract_model_name(plan.parsed)
            if not model_name:
                continue
            remaining = remaining_by_model.get(model_name, 0)
            future_in_batch = remaining - 1 if remaining > 0 else 0
            queued_elsewhere = pending_counts.get(model_name, 0)
            retain_engine = (future_in_batch + queued_elsewhere) > 0
            self._set_release_after_run_flag(plan, retain_engine)
            if remaining > 0:
                remaining_by_model[model_name] = future_in_batch

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

    def _set_release_after_run_flag(self, plan: DispatchPlan, retain_engine: bool) -> None:
        spec = plan.parsed.setdefault("spec", {})
        model_cfg = spec.get("model")
        if isinstance(model_cfg, dict):
            vllm_cfg = model_cfg.get("vllm")
            if vllm_cfg is None:
                vllm_cfg = {}
                model_cfg["vllm"] = vllm_cfg
            if isinstance(vllm_cfg, dict):
                vllm_cfg["release_after_run"] = not retain_engine

        metadata = plan.parsed.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations["retain_inference_engine"] = bool(retain_engine)

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
        self._merge_slices: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self._merge_parent: Dict[str, str] = {}

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
            task_store=self,
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

    def update_parsed(self, task_id: str, parsed: Dict[str, Any]) -> None:
        with self._lock:
            if task_id in self._parsed:
                self._parsed[task_id] = copy.deepcopy(parsed)

    def register_merge(self, parent_id: str, slices: Dict[str, Tuple[int, int]]) -> None:
        with self._lock:
            existing = self._merge_slices.get(parent_id, {})
            existing.update(slices)
            self._merge_slices[parent_id] = existing
            for task_id in slices.keys():
                if task_id != parent_id:
                    self._merge_parent[task_id] = parent_id

    def clear_merge(self, parent_id: str) -> None:
        with self._lock:
            slices = self._merge_slices.pop(parent_id, {})
            for task_id in list(slices.keys()):
                if task_id != parent_id:
                    self._merge_parent.pop(task_id, None)

    def get_merge_slices(self, parent_id: str) -> Dict[str, Tuple[int, int]]:
        with self._lock:
            return copy.deepcopy(self._merge_slices.get(parent_id, {}))

    def get_merge_parent(self, task_id: str) -> Optional[str]:
        with self._lock:
            return self._merge_parent.get(task_id)

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
            return [
                tid
                for tid in self._parsed.keys()
                if tid not in self._released and tid not in self._merge_parent
            ]

    def ready_to_dispatch(self, is_dep_satisfied: Callable[[str], bool]) -> List[str]:
        ready: List[str] = []
        with self._lock:
            for tid, deps in self._depends.items():
                if tid in self._released:
                    continue
                if tid in self._merge_parent:
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

    # -------------------------
    # Persistence helpers
    # -------------------------

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the in-memory state."""
        with self._lock:
            merge_slices: Dict[str, Dict[str, List[int]]] = {}
            for parent_id, slices in self._merge_slices.items():
                merge_slices[parent_id] = {
                    child_id: [int(slice_vals[0]), int(slice_vals[1])]
                    for child_id, slice_vals in slices.items()
                }
            return {
                "parsed": copy.deepcopy(self._parsed),
                "depends": {k: list(v) for k, v in self._depends.items()},
                "released": list(self._released),
                "load": copy.deepcopy(self._load),
                "slo": copy.deepcopy(self._slo),
                "dead_letters": copy.deepcopy(self._dead_letters),
                "merge_slices": merge_slices,
                "merge_parent": copy.deepcopy(self._merge_parent),
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore TaskStore internals from a snapshot."""
        if not state:
            return
        with self._lock:
            self._parsed = copy.deepcopy(state.get("parsed", {}))
            depends = state.get("depends", {})
            self._depends = {task_id: set(items or []) for task_id, items in depends.items()}
            self._released = set(state.get("released", []))
            self._load = {tid: int(val) for tid, val in (state.get("load") or {}).items()}
            self._slo = copy.deepcopy(state.get("slo", {}))
            self._dead_letters = copy.deepcopy(state.get("dead_letters", []))

            merge_state: Dict[str, Dict[str, Tuple[int, int]]] = {}
            for parent_id, slices in (state.get("merge_slices") or {}).items():
                nested: Dict[str, Tuple[int, int]] = {}
                if isinstance(slices, dict):
                    for child_id, values in slices.items():
                        if isinstance(values, (list, tuple)) and len(values) == 2:
                            nested[child_id] = (int(values[0]), int(values[1]))
                merge_state[parent_id] = nested
            self._merge_slices = merge_state
            self._merge_parent = {k: v for k, v in (state.get("merge_parent") or {}).items()}

    def reset_release(self, task_id: str) -> None:
        """Allow a task to be redispatched by clearing its released flag."""
        with self._lock:
            self._released.discard(task_id)
