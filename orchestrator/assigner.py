# assigner.py
"""Worker data model, Redis helpers, and assignment logic.

This module contains:
- The Worker pydantic model.
- Utilities to read/update workers stored in Redis.
- Freshness checks via heartbeat TTL.
- Hardware capability checks.
- Idle worker discovery and selection for task assignment.
"""
from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from utils import (
    now_iso,
    safe_get,
    parse_mem_to_bytes,
    r_worker_key,
    r_hb_key,
    WORKERS_SET,
)

logger = logging.getLogger(__name__)


class Worker(BaseModel):
    worker_id: str
    status: str = Field(default="UNKNOWN")
    started_at: Optional[str] = None
    pid: Optional[int] = None
    env: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    last_seen: Optional[str] = None


# -------------------------
# Freshness / heartbeats
# -------------------------

def is_stale_by_redis(rds, worker_id: str) -> bool:
    """A worker is stale if its heartbeat key is missing or expired."""
    ttl = rds.ttl(r_hb_key(worker_id))
    return (ttl is None) or (ttl < 0)


# -------------------------
# Worker read/update helpers
# -------------------------

def get_worker_from_redis(rds, worker_id: str) -> Optional[Worker]:
    """Read a worker hash and deserialize fields stored as JSON strings."""
    h = rds.hgetall(r_worker_key(worker_id))
    if not h:
        return None
    try:
        env = json.loads(h.get("env_json", "{}") or "{}")
        hardware = json.loads(h.get("hardware_json", "{}") or "{}")
        tags = json.loads(h.get("tags_json", "[]") or "[]")
    except Exception:
        env, hardware, tags = {}, {}, []
    return Worker(
        worker_id=h.get("worker_id", worker_id),
        status=h.get("status", "UNKNOWN"),
        started_at=h.get("started_at") or None,
        pid=int(h.get("pid") or 0) or None,
        env=env,
        hardware=hardware,
        tags=tags,
        last_seen=h.get("last_seen") or None,
    )


def list_workers_from_redis(rds) -> List[Dict[str, Any]]:
    """Return a list of workers with a computed `stale` field."""
    ids = list(rds.smembers(WORKERS_SET))
    out: List[Dict[str, Any]] = []
    for wid in ids:
        w = get_worker_from_redis(rds, wid)
        if not w:
            continue
        stale = is_stale_by_redis(rds, wid)
        if not w.last_seen:
            hb_ts = rds.get(r_hb_key(wid))
            w.last_seen = hb_ts
        out.append({**w.model_dump(), "stale": stale})
    return out


def update_worker_status(rds, worker_id: str, status: str) -> None:
    """Update worker status and last_seen, and publish a STATUS event."""
    now = now_iso()
    with rds.pipeline() as p:
        p.hset(r_worker_key(worker_id), mapping={"status": status, "last_seen": now})
        p.publish("workers.events", json.dumps({
            "type": "STATUS", "worker_id": worker_id, "status": status, "ts": now
        }, ensure_ascii=False))
        p.execute()


def unregister_worker(rds, worker_id: str) -> None:
    """Remove worker from the set and delete its keys; publish an UNREGISTER event."""
    with rds.pipeline() as p:
        p.srem(WORKERS_SET, worker_id)
        p.delete(r_worker_key(worker_id))
        p.delete(r_hb_key(worker_id))
        p.publish("workers.events", json.dumps({
            "type": "UNREGISTER", "worker_id": worker_id, "ts": now_iso()
        }, ensure_ascii=False))
        p.execute()
    logger.info("Worker unregistered: %s", worker_id)


def cleanup_stale_workers(rds) -> int:
    """Unregister all workers that are stale by heartbeat TTL."""
    removed = 0
    for wid in list(rds.smembers(WORKERS_SET)):
        if is_stale_by_redis(rds, wid):
            unregister_worker(rds, wid)
            removed += 1
    return removed


# -------------------------
# Capability checks / assignment
# -------------------------

def _hw_satisfies(worker: Worker, task: Dict[str, Any]) -> bool:
    """Check if a worker meets the task's declared hardware requirements."""
    hw = worker.hardware or {}

    # CPU (task value may contain non-digits, keep digits only)
    cpu_req = int(re.sub(r"\D", "", str(safe_get(task, "spec.resources.hardware.cpu", "0"))) or "0")
    cpu_have = int(safe_get(hw, "cpu.logical_cores", 0) or 0)
    if cpu_have < cpu_req:
        return False

    # Memory
    mem_req = parse_mem_to_bytes(str(safe_get(task, "spec.resources.hardware.memory", "0"))) or 0
    mem_have = int(safe_get(hw, "memory.total_bytes", 0) or 0)
    if mem_have and mem_req and mem_have < mem_req:
        return False

    # GPU (optional)
    gpu_req = safe_get(task, "spec.resources.hardware.gpu", None)
    if gpu_req:
        want_count = int(safe_get(gpu_req, "count", 0) or 0)
        want_type = str(safe_get(gpu_req, "type", "any") or "any").lower()
        gpus = (safe_get(hw, "gpu.gpus", []) or [])
        if len(gpus) < want_count:
            return False
        if want_type != "any":
            names = [str(g.get("name", "")).lower() for g in gpus]
            if not any(want_type in n for n in names):
                return False

    return True


def _load_idle_workers_sorted(rds, exclude: Optional[Set[str]] = None) -> List[Worker]:
    """Return IDLE, non-stale workers sorted by cores and memory (desc).

    Args:
        rds: Redis client.
        exclude: optional set of worker_ids to skip (e.g., previously failed).
    """
    workers: List[Worker] = []
    exclude = exclude or set()
    for wid in rds.smembers(WORKERS_SET):
        if wid in exclude:
            continue
        w = get_worker_from_redis(rds, wid)
        if not w:
            continue
        if w.status != "IDLE":
            continue
        if is_stale_by_redis(rds, w.worker_id):
            continue
        workers.append(w)
    workers.sort(
        key=lambda w: (
            int(safe_get(w.hardware, "cpu.logical_cores", 0) or 0),
            int(safe_get(w.hardware, "memory.total_bytes", 0) or 0),
        ),
        reverse=True,
    )
    return workers


def pick_idle_worker(rds, task: Dict[str, Any]) -> Optional[Worker]:
    """Pick the first IDLE worker that satisfies task requirements."""
    for w in _load_idle_workers_sorted(rds):
        if _hw_satisfies(w, task):
            return w
    return None


def pick_idle_worker_excluding(
    rds,
    task: Dict[str, Any],
    exclude_worker_id: Optional[str] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> Optional[Worker]:
    """Pick an IDLE, non-stale worker that satisfies the task but excludes some ids.

    Args:
        exclude_worker_id: single worker id to avoid (e.g., the one that just failed).
        exclude_ids: a set of worker ids to avoid. If both provided, union is used.

    Returns:
        A Worker or None if no suitable candidate exists.
    """
    ex: Set[str] = set(exclude_ids or set())
    if exclude_worker_id:
        ex.add(exclude_worker_id)

    for w in _load_idle_workers_sorted(rds, exclude=ex if ex else None):
        if _hw_satisfies(w, task):
            return w
    return None
