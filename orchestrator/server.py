#!/usr/bin/env python3
"""
Orchestrator

Admin/ops (HTTP):
  * GET  /workers, GET /workers/{id}, GET /healthz
  * GET  /api/v1/tasks, GET /api/v1/tasks/{id}
  * POST /api/v1/tasks        -> submit YAML task (Content-Type: text/yaml)
  * POST /admin/cleanup       -> cleanup stale workers (new)

Workers (Redis-only protocol; HTTP optional/back-compat):
  * REGISTER:   HSET worker:{id} ...; SADD workers:ids {id}; PUBLISH workers.events {"type":"REGISTER",...}
  * HEARTBEAT:  SETEX worker:{id}:hb <HB_TTL_SEC> <iso_ts>
  * STATUS:     HSET worker:{id} status=<STATUS>; HSET worker:{id} last_seen=<iso_ts>

ENV:
  ORCHESTRATOR_TOKEN   optional bearer for simple auth (protects HTTP endpoints)
  HEARTBEAT_TTL_SEC    stale detection window (default 120)
  REDIS_URL            REQUIRED, e.g. redis://localhost:6379/0
  LOG_FILE             log path (default orchestrator.log)
  LOG_MAX_BYTES        rotating size (default 5_242_880 ~5MB)
  LOG_BACKUP_COUNT     rotating backups (default 5)
  LOG_LEVEL            INFO/DEBUG...
  PORT                 server port (default 8000)
"""
from __future__ import annotations
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path as FSPath
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union
import uuid

import yaml
import redis
from fastapi import FastAPI, HTTPException, Path as ApiPath, Depends, Request, Body
from pydantic import BaseModel, Field

# -------------------------
# Helpers
# -------------------------
def _parse_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val.replace("_", ""))
    except Exception:
        return default

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_mem_to_bytes(mem_str: str) -> Optional[int]:
    if not isinstance(mem_str, str):
        return None
    m = re.match(r"^\s*([0-9]+)\s*([KkMmGgTt][Ii]?[Bb]?)?\s*$", mem_str)
    if not m:
        return None
    qty = int(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("k","kb","kib"): return qty * 1024
    if unit in ("m","mb","mib"): return qty * 1024**2
    if unit in ("g","gb","gib"): return qty * 1024**3
    if unit in ("t","tb","tib"): return qty * 1024**4
    if unit == "": return qty
    return None

# -------------------------
# Logging
# -------------------------
LOG_FILE = os.getenv("LOG_FILE", "orchestrator.log")
LOG_MAX_BYTES = _parse_int_env("LOG_MAX_BYTES", 5_242_880)
LOG_BACKUP_COUNT = _parse_int_env("LOG_BACKUP_COUNT", 5)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

log_path = FSPath(LOG_FILE)
if log_path.parent and not log_path.parent.exists():
    log_path.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("orchestrator")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.addHandler(ch)

# -------------------------
# Redis
# -------------------------
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    logger.error("REDIS_URL is required for Redis-only task dispatch")
    raise SystemExit(1)

try:
    rds = redis.from_url(REDIS_URL, decode_responses=True)
    rds.ping()
    logger.info("Connected to Redis: %s", REDIS_URL)
except Exception as e:
    logger.exception("Failed to connect to Redis: %s", e)
    raise SystemExit(1)

# -------------------------
# Auth
# -------------------------
ORCH_BEARER = os.getenv("ORCHESTRATOR_TOKEN")

async def require_auth(request: Request):
    if not ORCH_BEARER:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != ORCH_BEARER:
        raise HTTPException(status_code=403, detail="Invalid token")

# -------------------------
# Models
# -------------------------
class Worker(BaseModel):
    worker_id: str
    status: str = Field(default="UNKNOWN")
    started_at: Optional[str] = None
    pid: Optional[int] = None
    env: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    last_seen: Optional[str] = None

class RegisterIn(BaseModel):
    worker_id: str
    status: str
    started_at: str
    pid: Optional[int] = None
    env: Dict[str, Any]
    hardware: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)

HB_TTL_SEC = _parse_int_env("HEARTBEAT_TTL_SEC", 120)

class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"

class TaskRecord(BaseModel):
    task_id: str
    raw_yaml: str
    parsed: Dict[str, Any]
    status: str = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    topic: Optional[str] = None
    submitted_at: str = Field(default_factory=_now_iso)
    error: Optional[str] = None

TASKS: Dict[str, TaskRecord] = {}

# -------------------------
# Redis keys
# -------------------------
WORKERS_SET = "workers:ids"
def _r_worker_key(worker_id: str) -> str: return f"worker:{worker_id}"
def _r_hb_key(worker_id: str) -> str: return f"worker:{worker_id}:hb"

def _is_stale_by_redis(worker_id: str) -> bool:
    ttl = rds.ttl(_r_hb_key(worker_id))
    return (ttl is None) or (ttl < 0)

def get_worker_from_redis(worker_id: str) -> Optional[Worker]:
    h = rds.hgetall(_r_worker_key(worker_id))
    if not h: return None
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

def list_workers_from_redis() -> List[Dict[str, Any]]:
    ids = list(rds.smembers(WORKERS_SET))
    out: List[Dict[str, Any]] = []
    for wid in ids:
        w = get_worker_from_redis(wid)
        if not w: continue
        stale = _is_stale_by_redis(wid)
        if not w.last_seen:
            hb_ts = rds.get(_r_hb_key(wid))
            w.last_seen = hb_ts
        out.append({**w.model_dump(), "stale": stale})
    return out

def update_worker_status(worker_id: str, status: str) -> None:
    now = _now_iso()
    with rds.pipeline() as p:
        p.hset(_r_worker_key(worker_id), mapping={"status": status, "last_seen": now})
        p.publish("workers.events", json.dumps({
            "type": "STATUS", "worker_id": worker_id, "status": status, "ts": now
        }, ensure_ascii=False))
        p.execute()

def record_heartbeat(worker_id: str, ts: Optional[str] = None) -> None:
    ts = ts or _now_iso()
    with rds.pipeline() as p:
        p.setex(_r_hb_key(worker_id), HB_TTL_SEC, ts)
        p.hset(_r_worker_key(worker_id), mapping={"last_seen": ts})
        p.publish("workers.events", json.dumps({
            "type": "HEARTBEAT", "worker_id": worker_id, "ts": ts
        }, ensure_ascii=False))
        p.execute()

# --- New: unregister & cleanup ---
def unregister_worker(worker_id: str) -> None:
    with rds.pipeline() as p:
        p.srem(WORKERS_SET, worker_id)
        p.delete(_r_worker_key(worker_id))
        p.delete(_r_hb_key(worker_id))
        p.publish("workers.events", json.dumps({
            "type": "UNREGISTER", "worker_id": worker_id, "ts": _now_iso()
        }, ensure_ascii=False))
        p.execute()
    logger.info("Worker unregistered: %s", worker_id)

def cleanup_stale_workers() -> int:
    removed = 0
    for wid in list(rds.smembers(WORKERS_SET)):
        if _is_stale_by_redis(wid):
            unregister_worker(wid)
            removed += 1
    return removed

# -------------------------
# App
# -------------------------
app = FastAPI(title="Orchestrator", version="1.0.0")

@app.get("/healthz")
async def healthz(): return {"ok": True}

@app.get("/workers")
async def list_workers(_: Any = Depends(require_auth)):
    return list_workers_from_redis()

@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str, _: Any = Depends(require_auth)):
    w = get_worker_from_redis(worker_id)
    if not w: raise HTTPException(status_code=404, detail="worker not found")
    stale = _is_stale_by_redis(worker_id)
    return {**w.model_dump(), "stale": stale}

@app.post("/admin/cleanup")
async def admin_cleanup(_: Any = Depends(require_auth)):
    removed = cleanup_stale_workers()
    return {"ok": True, "removed": removed}

# -------------------------
# Task Submission (Redis-only dispatch)
# -------------------------

def validate_and_parse_task_yaml(yaml_text: str) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    required = [
        "apiVersion", "kind", "metadata.name",
        "spec.taskType", "spec.resources.replicas",
        "spec.resources.hardware.cpu",
        "spec.resources.hardware.memory",
    ]
    for p in required:
        cur = data
        for k in p.split('.'):
            if not isinstance(cur, dict) or k not in cur:
                raise HTTPException(status_code=400, detail=f"Missing required field: {p}")
            cur = cur[k]
    # optional gpu
    gpu = _safe_get(data, "spec.resources.hardware.gpu", {})
    if gpu:
        if _safe_get(gpu, "count") is None:
            raise HTTPException(status_code=400, detail="Missing spec.resources.hardware.gpu.count")
        if _safe_get(gpu, "type") is None:
            raise HTTPException(status_code=400, detail="Missing spec.resources.hardware.gpu.type")
    return data


def _is_stale(worker: Worker) -> bool:
    # Redis is source of truth now: rely on hb key
    return _is_stale_by_redis(worker.worker_id)


def _hw_satisfies(worker: Worker, task: Dict[str, Any]) -> bool:
    hw = worker.hardware or {}
    # CPU
    cpu_req = int(re.sub(r"\D", "", str(_safe_get(task, "spec.resources.hardware.cpu", "0")) ) or "0")
    cpu_have = int(_safe_get(hw, "cpu.logical_cores", 0) or 0)
    if cpu_have < cpu_req:
        return False
    # Memory
    mem_req = parse_mem_to_bytes(str(_safe_get(task, "spec.resources.hardware.memory", "0"))) or 0
    mem_have = int(_safe_get(hw, "memory.total_bytes", 0) or 0)
    if mem_have and mem_req and mem_have < mem_req:
        return False
    # GPU
    gpu_req = _safe_get(task, "spec.resources.hardware.gpu", None)
    if gpu_req:
        want_count = int(_safe_get(gpu_req, "count", 0) or 0)
        want_type = str(_safe_get(gpu_req, "type", "any") or "any").lower()
        gpus = (_safe_get(hw, "gpu.gpus", []) or [])
        if len(gpus) < want_count:
            return False
        if want_type != "any":
            names = [str(g.get("name","")).lower() for g in gpus]
            if not any(want_type in n for n in names):
                return False
    return True


def _load_idle_workers_sorted() -> List[Worker]:
    # Load all workers from Redis and filter IDLE & non-stale
    workers: List[Worker] = []
    for wid in rds.smembers(WORKERS_SET):
        w = get_worker_from_redis(wid)
        if not w:
            continue
        if w.status != "IDLE":
            continue
        if _is_stale(w):
            continue
        workers.append(w)
    # prefer more cores & memory
    workers.sort(key=lambda w: (
        int(_safe_get(w.hardware, "cpu.logical_cores", 0) or 0),
        int(_safe_get(w.hardware, "memory.total_bytes", 0) or 0)
    ), reverse=True)
    return workers


def _pick_idle_worker(task: Dict[str, Any]) -> Optional[Worker]:
    for w in _load_idle_workers_sorted():
        if _hw_satisfies(w, task):
            return w
    return None


def _publish_task(topic: str, message: Dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False)
    n = rds.publish(topic, payload)
    logger.info("Published to topic=%s receivers=%d", topic, n)


@app.post("/api/v1/tasks")
async def submit_task(raw: Union[str, Dict[str, Any]] = Body(..., media_type="text/yaml"),
                      _: Any = Depends(require_auth)):
    if isinstance(raw, dict) and "yaml" in raw:
        yml = str(raw["yaml"])
    elif isinstance(raw, str):
        yml = raw
    else:
        raise HTTPException(status_code=400, detail='Expected YAML string or {"yaml":"..."}')

    task = validate_and_parse_task_yaml(yml)
    task_id = str(uuid.uuid4())
    rec = TaskRecord(task_id=task_id, raw_yaml=yml, parsed=task)
    TASKS[task_id] = rec

    worker = _pick_idle_worker(task)
    if not worker:
        rec.status = TaskStatus.FAILED
        rec.error = "No suitable IDLE worker"
        logger.warning("Schedule failed: %s", rec.error)
        return {"ok": False, "task_id": task_id, "error": rec.error}

    topic = f"tasks.{_safe_get(task, 'spec.taskType', 'generic')}"
    msg = {
        "task_id": task_id,
        "task": task,
        "assigned_worker": worker.worker_id,
        "dispatched_at": _now_iso(),
    }

    try:
        _publish_task(topic, msg)
        rec.status = TaskStatus.DISPATCHED
        rec.assigned_worker = worker.worker_id
        rec.topic = topic
        # mark worker RUNNING (in Redis)
        update_worker_status(worker.worker_id, "RUNNING")
        return {"ok": True, "task_id": task_id, "worker_id": worker.worker_id, "topic": topic}
    except Exception as e:
        rec.status = TaskStatus.FAILED
        rec.error = f"Publish failed: {e}"
        logger.exception("Publish failed")
        return {"ok": False, "task_id": task_id, "error": rec.error}


@app.get("/api/v1/tasks")
async def list_tasks(_: Any = Depends(require_auth)):
    return [t.model_dump() for t in TASKS.values()]

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str = ApiPath(..., min_length=1), _: Any = Depends(require_auth)):
    t = TASKS.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="task not found")
    return t.model_dump()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = _parse_int_env("PORT", 8000)
    logger.info("Starting on 0.0.0.0:%d", port)
    uvicorn.run("orchestrator.server:app", host="0.0.0.0", port=port, reload=False)
