from __future__ import annotations

"""
Worker data model, Redis helpers, and capability checks.

Design goals:
- Keep Redis I/O minimal and robust (graceful JSON decode).
- Make capability checks transparent: on rejection, log the exact reason.
- Be lenient with hardware schemas (per-GPU list OR aggregate gpu.count/type).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, MutableMapping, Tuple

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


# ------------------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------------------

class Worker(BaseModel):
    """Runtime snapshot of a worker process."""
    worker_id: str
    status: str = Field(default="UNKNOWN")  # e.g. IDLE/RUNNING/UNKNOWN
    started_at: Optional[str] = None        # ISO timestamp
    pid: Optional[int] = None
    env: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    last_seen: Optional[str] = None         # ISO timestamp of last status update


# ------------------------------------------------------------------------------
# Freshness / heartbeats
# ------------------------------------------------------------------------------

def is_stale_by_redis(rds, worker_id: str) -> bool:
    """
    A worker is stale if its heartbeat key has no TTL (missing or not expiring).
    Redis TTL semantics:
      - -2: key does not exist (some clients map to None)
      - -1: key exists but has no associated expire
    We treat None or <0 as stale.
    """
    ttl = rds.ttl(r_hb_key(worker_id))
    return (ttl is None) or (ttl < 0)


# ------------------------------------------------------------------------------
# Worker read/update helpers
# ------------------------------------------------------------------------------

def get_worker_from_redis(rds, worker_id: str) -> Optional[Worker]:
    """
    Load a worker record from Redis and decode JSON subfields gracefully.
    Returns None if the worker hash is missing.
    """
    h = rds.hgetall(r_worker_key(worker_id))
    if not h:
        return None

    def _loads(s: Optional[str], default: Any) -> Any:
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default

    env = _loads(h.get("env_json"), {})
    hardware = _loads(h.get("hardware_json"), {})
    tags = _loads(h.get("tags_json"), [])

    pid_val = h.get("pid")
    try:
        pid = int(pid_val) if pid_val is not None else None
    except (TypeError, ValueError):
        pid = None

    return Worker(
        worker_id=h.get("worker_id", worker_id),
        status=h.get("status", "UNKNOWN"),
        started_at=h.get("started_at") or None,
        pid=pid or None,
        env=env,
        hardware=hardware,
        tags=tags,
        last_seen=h.get("last_seen") or None,
    )


def list_workers_from_redis(rds) -> List[Dict[str, Any]]:
    """
    Return a list of worker dicts augmented with 'stale' boolean.
    Falls back to reading the heartbeat timestamp into last_seen if missing.
    """
    out: List[Dict[str, Any]] = []
    for wid in list(rds.smembers(WORKERS_SET)):
        w = get_worker_from_redis(rds, wid)
        if not w:
            continue
        stale = is_stale_by_redis(rds, wid)
        if not w.last_seen:
            hb_ts = rds.get(r_hb_key(wid))
            if hb_ts:
                w.last_seen = hb_ts
        out.append({**w.model_dump(), "stale": stale})
    return out


def update_worker_status(rds, worker_id: str, status: str) -> None:
    """
    Update worker status and publish a STATUS event.
    """
    now = now_iso()
    payload = {
        "type": "STATUS",
        "worker_id": worker_id,
        "status": status,
        "ts": now,
    }
    with rds.pipeline() as p:
        p.hset(r_worker_key(worker_id), mapping={"status": status, "last_seen": now})
        p.publish("workers.events", json.dumps(payload, ensure_ascii=False))
        p.execute()


def unregister_worker(rds, worker_id: str) -> None:
    """
    Remove worker from the set and delete its hash/heartbeat keys.
    Publish an UNREGISTER event.
    """
    payload = {"type": "UNREGISTER", "worker_id": worker_id, "ts": now_iso()}
    with rds.pipeline() as p:
        p.srem(WORKERS_SET, worker_id)
        p.delete(r_worker_key(worker_id))
        p.delete(r_hb_key(worker_id))
        p.publish("workers.events", json.dumps(payload, ensure_ascii=False))
        p.execute()
    logger.info("Worker unregistered: %s", worker_id)


def cleanup_stale_workers(rds) -> int:
    """
    Unregister all workers whose heartbeat TTL is stale.
    Returns the number of removed workers.
    """
    removed = 0
    for wid in list(rds.smembers(WORKERS_SET)):
        if is_stale_by_redis(rds, wid):
            unregister_worker(rds, wid)
            removed += 1
    return removed


# ------------------------------------------------------------------------------
# Capability checks
# ------------------------------------------------------------------------------

def _hw_satisfies(worker: Worker, task: Dict[str, Any]) -> bool:
    """
    Return True if the worker satisfies the task's hardware requirements.

    Supported requirements schema (all optional):
      spec.resources.hardware.cpu        -> int or string (e.g. "16")
      spec.resources.hardware.memory     -> string (e.g. "64GiB", "512MiB")
      spec.resources.hardware.gpu.count  -> int
      spec.resources.hardware.gpu.type   -> string (e.g. "a100", "l40s")

    Worker hardware schema (accepted variants):
      hardware.cpu.logical_cores         -> int
      hardware.memory.total_bytes        -> int
      # GPUs (preferred detailed form)
      hardware.gpu.gpus                  -> list[ {name: "...", memory_gb: ...}, ... ]
      # Fallback aggregate form:
      hardware.gpu.count / hardware.gpu.type
      # Legacy top-level fallbacks:
      gpu.count / gpu.type    (and hardware.gpus as list)

    Notes:
    - GPU type match is case-insensitive and allows substring containment
      (e.g., want_type "a100" will match "NVIDIA A100-PCIE-80GB").
    - When the detailed per-GPU list is missing, we fall back to aggregate fields.
    """
    hw = worker.hardware or {}

    # ---- CPU check ----
    cpu_req = _parse_int(safe_get(task, "spec.resources.hardware.cpu", 0))
    cpu_have = int(safe_get(hw, "cpu.logical_cores", 0) or 0)
    if cpu_have and cpu_req and cpu_have < cpu_req:
        logger.debug("Reject %s: cpu logical_cores have=%s < req=%s",
                     worker.worker_id, cpu_have, cpu_req)
        return False

    # ---- Memory check ----
    mem_req = parse_mem_to_bytes(str(safe_get(task, "spec.resources.hardware.memory", "0"))) or 0
    mem_have = int(safe_get(hw, "memory.total_bytes", 0) or 0)
    if mem_have and mem_req and mem_have < mem_req:
        logger.debug("Reject %s: memory total_bytes have=%s < req=%s",
                     worker.worker_id, mem_have, mem_req)
        return False

    # ---- GPU check (optional) ----
    gpu_req = safe_get(task, "spec.resources.hardware.gpu", None)
    if not gpu_req:
        return True

    want_count = _parse_int(safe_get(gpu_req, "count", 0))
    want_type = str(safe_get(gpu_req, "type", "any") or "any").strip().lower()

    # Preferred forms
    gpu_info = safe_get(hw, "gpu", {}) or {}
    gpus_detailed = safe_get(gpu_info, "gpus", []) or safe_get(hw, "gpus", []) or []

    have_count, have_types = _extract_gpu_inventory(gpu_info, gpus_detailed, hw)

    # Count check
    if want_count and have_count < want_count:
        logger.debug("Reject %s: gpu count have=%s < req=%s",
                     worker.worker_id, have_count, want_count)
        return False

    # Type check
    if want_type != "any":
        if not _gpu_type_matches(want_type, have_types):
            logger.debug("Reject %s: gpu type miss match have=%s, req~=%s",
                         worker.worker_id, have_types, want_type)
            return False

    return True


# ------------------------------------------------------------------------------
# Helpers (parsing/matching)
# ------------------------------------------------------------------------------

_INT_RE = re.compile(r"-?\d+")

def _parse_int(v: Any) -> int:
    """
    Robust int parsing: accepts ints/strings like "16" or "CPU: 16".
    Returns 0 on failure.
    """
    if isinstance(v, int):
        return v
    s = str(v or "").strip()
    if not s:
        return 0
    m = _INT_RE.search(s)
    return int(m.group()) if m else 0


def _extract_gpu_inventory(
    gpu_info: Dict[str, Any],
    gpus_detailed: List[Dict[str, Any]],
    hw_root: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """
    Returns (count, types) populated from detailed list if available,
    otherwise falling back to aggregate or legacy locations.
    - count: total number of GPUs
    - types: lowercased list of type strings (may be empty if unknown)
    """
    # Case 1: detailed list is available
    if gpus_detailed:
        names = []
        for g in gpus_detailed:
            name = str(g.get("name") or g.get("type") or "").strip().lower()
            if name:
                names.append(name)
        return len(gpus_detailed), names

    # Case 2: aggregate fields on hardware.gpu
    count = _parse_int(safe_get(gpu_info, "count", 0))
    gtype = str(safe_get(gpu_info, "type", "") or "").strip().lower()
    if count > 0 or gtype:
        types = [gtype] if gtype else []
        return count, types

    # Case 3: legacy top-level fallbacks
    count = _parse_int(safe_get(hw_root, "gpu.count", 0))
    gtype = str(safe_get(hw_root, "gpu.type", "") or "").strip().lower()
    types = [gtype] if gtype else []
    return count, types


def _gpu_type_matches(want_type: str, have_types: List[str]) -> bool:
    """
    Case-insensitive, fuzzy containment match for GPU types.
    Examples:
      want 'a100' matches 'nvidia a100-pcie-80gb'
      want 'l40s' matches 'nvidia l40s'
    If have_types is empty, we conservatively fail the match.
    """
    if not have_types:
        return False
    want = want_type.lower()
    for t in have_types:
        t = (t or "").lower()
        if want in t or t in want:  # allow both directions (A100 vs NVIDIA A100)
            return True
    return False
