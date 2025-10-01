# scheduler.py
"""Capacity-aware scheduling & sharding.

Data parallel only activates when:
  - spec.taskType == "inference"
  - spec.parallel.enabled == true

When enabled and capacity is high, we:
  - fan out shards across workers,
  - override spec.data.split into per-shard sub-ranges,
  - stamp spec.shard = {index,total,parent_task_id}.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import safe_get, parse_int_env, WORKERS_SET
from assigner import Worker, is_stale_by_redis, get_worker_from_redis, _hw_satisfies
from task_load import HEAVY_LOAD_THRESHOLD, MEDIUM_LOAD_THRESHOLD

# Tunables (env)
CPU_ONLY_WEIGHT = float(os.getenv("CPU_ONLY_WEIGHT", "0.6"))
GPU_WEIGHT = float(os.getenv("GPU_WEIGHT", "1.0"))
HIGH_AVAIL_RATIO = float(os.getenv("HIGH_AVAIL_RATIO", "2.0"))
LOW_AVAIL_RATIO = float(os.getenv("LOW_AVAIL_RATIO", "0.6"))
MAX_AUTO_SHARDS = parse_int_env("MAX_AUTO_SHARDS", 8)

# ---------------------------------------------------------------------------
# Feature gates
# ---------------------------------------------------------------------------

def is_data_parallel_enabled(task: Dict[str, Any]) -> bool:
    """Data-parallel is allowed only when taskType=='inference' AND parallel.enabled==true."""
    task_type = str(safe_get(task, "spec.taskType", "")).lower()
    para_enabled = bool(safe_get(task, "spec.parallel.enabled", False))
    return task_type == "inference" and para_enabled

def max_auto_shards_for(task: Dict[str, Any]) -> int:
    """Max shard fan-out for a given task (spec.parallel.max_shards overrides env)."""
    override = safe_get(task, "spec.parallel.max_shards", None)
    if override is None:
        return MAX_AUTO_SHARDS
    try:
        v = int(override)
        return max(1, v)
    except Exception:
        return MAX_AUTO_SHARDS

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

def _has_gpus(worker: Worker) -> bool:
    gpus = safe_get(worker.hardware, "gpu.gpus", []) or []
    return len(gpus) > 0

def _weight_for(worker: Worker) -> float:
    return GPU_WEIGHT if _has_gpus(worker) else CPU_ONLY_WEIGHT

def idle_satisfying_pool(rds, task: Dict[str, Any], exclude_ids: Optional[Set[str]] = None) -> List[Worker]:
    """Return IDLE, non-stale workers that satisfy task requirements."""
    exclude_ids = exclude_ids or set()
    out: List[Worker] = []
    for wid in rds.smembers(WORKERS_SET):
        if wid in exclude_ids:
            continue
        w = get_worker_from_redis(rds, wid)
        if not w or w.status != "IDLE":
            continue
        if is_stale_by_redis(rds, w.worker_id):
            continue
        if _hw_satisfies(w, task):
            out.append(w)
    return out

def weighted_available(workers: List[Worker]) -> float:
    return sum(_weight_for(w) for w in workers)

from utils import safe_get, parse_mem_to_bytes  # ensure these are imported

def sort_workers(workers: List[Worker]) -> List[Worker]:
    """Prefer GPU workers; then by total GPU VRAM (desc), system RAM (desc), CPU cores (desc)."""

    def _gpu_count_and_mem_bytes(w: Worker) -> tuple[int, int]:
        """Return (#GPUs, total GPU memory in bytes) with tolerant field parsing."""
        gpus = safe_get(w.hardware, "gpu.gpus", []) or []
        count = len(gpus)
        total_bytes = 0
        for g in gpus:
            # Try common fields first; fall back to nested structures/strings with units.
            cand = (
                safe_get(g, "memory.total_bytes")
                or safe_get(g, "memory_bytes")
                or safe_get(g, "vram_bytes")
                or safe_get(g, "memory.total")
                or safe_get(g, "vram")
                or safe_get(safe_get(g, "memory", {}) or {}, "total_bytes")
                or safe_get(safe_get(g, "memory", {}) or {}, "bytes")
                or safe_get(safe_get(g, "memory", {}) or {}, "total")
            )
            try:
                bytes_val = parse_mem_to_bytes(cand) if isinstance(cand, str) else int(cand or 0)
            except Exception:
                bytes_val = 0
            total_bytes += int(bytes_val)
        return count, total_bytes

    return sorted(
        workers,
        key=lambda w: (
            1 if _gpu_count_and_mem_bytes(w)[0] > 0 else 0,                         # GPU present first
            _gpu_count_and_mem_bytes(w)[1],                                         # total GPU VRAM
            int(safe_get(w.hardware, "memory.total_bytes", 0) or 0),                # system RAM
            int(safe_get(w.hardware, "cpu.logical_cores", 0) or 0),                 # CPU cores
        ),
        reverse=True,
    )


def estimate_queue_length(task_store) -> int:
    """Approximate queued logical tasks (not yet released)."""
    try:
        return len(task_store.list_waiting_tasks())
    except Exception:
        return 1

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    def __init__(self, mode: str, shard_count: int = 1, prefer_best: bool = True) -> None:
        self.mode = mode  # "first_fit" | "best_fit" | "data_parallel"
        self.shard_count = shard_count
        self.prefer_best = prefer_best

    def __repr__(self) -> str:
        return f"Strategy(mode={self.mode}, shard_count={self.shard_count}, prefer_best={self.prefer_best})"

def choose_strategy(task: Dict[str, Any], idle_pool: List[Worker], queued: int) -> Strategy:
    """Pick first/best/data-parallel based on availability:demand ratio."""
    demand = max(1, queued)
    avail_weight = weighted_available(idle_pool)
    ratio = avail_weight / demand

    base_replicas = int(safe_get(task, "spec.resources.replicas", 1) or 1)
    max_shards = max(1, min(len(idle_pool), max_auto_shards_for(task)))

    # Data parallel only if enabled by task
    if is_data_parallel_enabled(task) and ratio >= HIGH_AVAIL_RATIO and len(idle_pool) > 1:
        target = max(base_replicas, max_shards)
        return Strategy(mode="data_parallel", shard_count=target, prefer_best=True)

    if ratio <= LOW_AVAIL_RATIO:
        return Strategy(mode="first_fit", shard_count=1, prefer_best=False)

    return Strategy(mode="best_fit", shard_count=1, prefer_best=True)

# ---------------------------------------------------------------------------
# Selection & sharding
# ---------------------------------------------------------------------------

def select_workers_for_task(
    pool: List[Worker],
    shard_count: int,
    prefer_best: bool,
    task_load: int = 0,
) -> List[Worker]:
    if not pool:
        return []
    if prefer_best or task_load >= HEAVY_LOAD_THRESHOLD:
        pool = sort_workers(pool)
    elif task_load >= MEDIUM_LOAD_THRESHOLD:
        pool = sort_workers(pool)
    return pool[:shard_count]

_SPLIT_RE = re.compile(r"^(?P<name>[^\[\]]+)(?:\[(?P<rng>[^\]]*)\])?$")

def _format_pct(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s

def _parse_range_to_pct(rng: Optional[str]) -> tuple[float, float]:
    """Parse 'a%:b%' / ':b%' / 'a%:' / '' into (start_pct, end_pct)."""
    if not rng or rng.strip() == "":
        return 0.0, 100.0
    parts = rng.split(":")
    if len(parts) == 1:
        a = parts[0].strip()
        if a.endswith("%"):
            return 0.0, float(a[:-1])
        return 0.0, 100.0
    a, b = parts[0].strip(), parts[1].strip()
    start = float(a[:-1]) if a.endswith("%") and a[:-1] != "" else 0.0
    end   = float(b[:-1]) if b.endswith("%") and b[:-1] != "" else 100.0
    start = max(0.0, min(100.0, start))
    end   = max(0.0, min(100.0, end))
    if end < start:
        start, end = end, start
    return start, end

def compute_shard_split(base_split: str, index: int, total: int) -> str:
    """
    Given a HF 'split' string like 'train[:10%]' or 'train[5%:15%]',
    divide its covered percentage evenly into `total` parts and return
    the sub-range for `index`. If parsing fails, return base_split.
    """
    m = _SPLIT_RE.match(base_split.strip())
    if not m:
        return base_split
    name = m.group("name").strip()
    rng  = m.group("rng")
    start, end = _parse_range_to_pct(rng)
    span = max(0.0, end - start)
    if total <= 0:
        total = 1
    step = span / total
    s_i = start + step * index
    e_i = start + step * (index + 1)
    return f"{name}[{_format_pct(s_i)}%:{_format_pct(e_i)}%]"

def make_shard_messages(
    parent_task_id: str,
    base_task: Dict[str, Any],
    shard_workers: List[Worker],
) -> List[Tuple[str, Dict[str, Any], Worker]]:
    """
    Create child messages with:
      - spec.shard = {index,total,parent_task_id}
      - data.split overridden per shard when possible
    """
    total = len(shard_workers)
    out: List[Tuple[str, Dict[str, Any], Worker]] = []
    base_data = dict(safe_get(base_task, "spec.data", {}) or {})
    base_split = str(base_data.get("split", "")).strip()

    for i, w in enumerate(shard_workers):
        child_id = str(uuid.uuid4())
        task_copy = dict(base_task)
        spec = dict(task_copy.get("spec", {}))

        # Stamp shard metadata
        spec["shard"] = {"index": i, "total": total, "parent_task_id": parent_task_id}

        # If data.split exists, try to carve it per-shard
        if base_split:
            try:
                shard_split = compute_shard_split(base_split, i, total)
                data = dict(spec.get("data", base_data))
                data["split"] = shard_split
                spec["data"] = data
            except Exception:
                # Fallback: keep original split and rely on spec.shard at executor
                pass

        task_copy["spec"] = spec
        out.append((child_id, task_copy, w))
    return out
