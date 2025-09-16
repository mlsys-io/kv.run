# assigner.py
"""Worker data model, Redis helpers, capability checks."""

from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List, Optional

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
    ttl = rds.ttl(r_hb_key(worker_id))
    return (ttl is None) or (ttl < 0)

# -------------------------
# Worker read/update helpers
# -------------------------

def get_worker_from_redis(rds, worker_id: str) -> Optional[Worker]:
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
    now = now_iso()
    with rds.pipeline() as p:
        p.hset(r_worker_key(worker_id), mapping={"status": status, "last_seen": now})
        p.publish("workers.events", json.dumps({
            "type": "STATUS", "worker_id": worker_id, "status": status, "ts": now
        }, ensure_ascii=False))
        p.execute()

def unregister_worker(rds, worker_id: str) -> None:
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
    removed = 0
    for wid in list(rds.smembers(WORKERS_SET)):
        if is_stale_by_redis(rds, wid):
            unregister_worker(rds, wid)
            removed += 1
    return removed

# -------------------------
# Capability checks
# -------------------------

def _hw_satisfies(worker: Worker, task: Dict[str, Any]) -> bool:
    hw = worker.hardware or {}
    cpu_req = int(re.sub(r"\D", "", str(safe_get(task, "spec.resources.hardware.cpu", "0"))) or "0")
    cpu_have = int(safe_get(hw, "cpu.logical_cores", 0) or 0)
    if cpu_have < cpu_req:
        return False

    mem_req = parse_mem_to_bytes(str(safe_get(task, "spec.resources.hardware.memory", "0"))) or 0
    mem_have = int(safe_get(hw, "memory.total_bytes", 0) or 0)
    if mem_have and mem_req and mem_have < mem_req:
        return False

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
