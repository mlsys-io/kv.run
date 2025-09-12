# main.py
"""FastAPI app entrypoint for the Orchestrator service (stages + events-driven).

This app:
- Provides admin and worker introspection endpoints.
- Accepts YAML task submissions (single or staged).
- Uses TaskStore to register tasks and track dependencies.
- Runs:
    * A dependency watcher (polling) to dispatch ready tasks, and
    * A tasks.events listener to react immediately to DONE/FAILED events:
        - On DONE/FINISH/SUCCEEDED: mark DONE, try to dispatch next tasks.
        - On FAILED/ERROR: retry on a different worker (up to max_retries).
"""
from __future__ import annotations

import os
import json
import uuid
import threading
import time
from typing import Any, Dict, Optional, Union, List

import redis
from fastapi import FastAPI, HTTPException, Path as ApiPath, Depends, Request, Body
from pydantic import BaseModel, Field

from utils import parse_int_env, now_iso, get_logger, safe_get
from parser import TaskStore
from assigner import (
    Worker,
    list_workers_from_redis,
    get_worker_from_redis,
    cleanup_stale_workers,
    pick_idle_worker,
    pick_idle_worker_excluding,
    update_worker_status,
    is_stale_by_redis,  
)


# -------------------------
# Logging & Redis setup
# -------------------------
LOG_FILE = os.getenv("LOG_FILE", "orchestrator.log")
LOG_MAX_BYTES = parse_int_env("LOG_MAX_BYTES", 5_242_880)
LOG_BACKUP_COUNT = parse_int_env("LOG_BACKUP_COUNT", 5)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = get_logger(
    name="orchestrator",
    log_file=LOG_FILE,
    max_bytes=LOG_MAX_BYTES,
    backup_count=LOG_BACKUP_COUNT,
    level=LOG_LEVEL,
)

REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
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
HB_TTL_SEC = parse_int_env("HEARTBEAT_TTL_SEC", 120)

class TaskStatus(str):
    PENDING = "PENDING"      # registered, waiting for deps or resources
    DISPATCHED = "DISPATCHED"# sent to a worker
    FAILED = "FAILED"        # dispatch error or worker-reported failure
    DONE = "DONE"            # worker-reported success

class TaskRecord(BaseModel):
    task_id: str
    raw_yaml: str
    parsed: Dict[str, Any]
    status: str = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    topic: Optional[str] = None
    submitted_at: str = Field(default_factory=now_iso)
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

# In-memory bookkeeping for submission view (orthogonal to TaskStore)
TASKS: Dict[str, TaskRecord] = {}
TASKS_LOCK = threading.RLock()

# Task store: tracks parsed tasks + dependencies (supports stages)
TASK_STORE = TaskStore()

# -------------------------
# App & routes
# -------------------------
app = FastAPI(title="Orchestrator", version="1.1.0")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/workers")
async def list_workers(_: Any = Depends(require_auth)):
    return list_workers_from_redis(rds)

@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str, _: Any = Depends(require_auth)):
    w = get_worker_from_redis(rds, worker_id)
    if not w:
        raise HTTPException(status_code=404, detail="worker not found")
    stale = is_stale_by_redis(rds, worker_id)
    return {**w.model_dump(), "stale": stale}

@app.post("/admin/cleanup")
async def admin_cleanup(_: Any = Depends(require_auth)):
    removed = cleanup_stale_workers(rds)
    return {"ok": True, "removed": removed}

# -------------------------
# Publish helper
# -------------------------
def _publish_task(topic: str, message: Dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False)
    receivers = rds.publish(topic, payload)
    logger.info("Published to topic=%s receivers=%d", topic, receivers)

# -------------------------
# Dispatch helpers
# -------------------------
def _try_dispatch_one(task_id: str, task: Dict[str, Any], rec: TaskRecord,
                      exclude_worker_id: Optional[str] = None) -> None:
    """Attempt to pick a worker and dispatch a single task (idempotent per rec)."""
    with TASKS_LOCK:
        if rec.status == TaskStatus.DISPATCHED:
            return

    # Use assigner helpers
    if exclude_worker_id:
        worker = pick_idle_worker_excluding(rds, task, exclude_worker_id=exclude_worker_id)
    else:
        worker = pick_idle_worker(rds, task)

    if not worker:
        with TASKS_LOCK:
            rec.status = TaskStatus.FAILED
            rec.error = "No suitable IDLE worker"
        logger.warning("Schedule failed: %s", rec.error)
        return

    topic = "tasks"  # unified channel
    message = {
        "task_id": task_id,
        "task": task,
        "task_type": safe_get(task, "spec.taskType"),
        "assigned_worker": worker.worker_id,
        "dispatched_at": now_iso(),
    }
    try:
        _publish_task(topic, message)
        update_worker_status(rds, worker.worker_id, "RUNNING")
        with TASKS_LOCK:
            rec.status = TaskStatus.DISPATCHED
            rec.assigned_worker = worker.worker_id
            rec.topic = topic
        TASK_STORE.mark_released(task_id)  # avoid re-dispatch by watcher
    except Exception as e:
        with TASKS_LOCK:
            rec.status = TaskStatus.FAILED
            rec.error = f"Publish failed: {e}"
        logger.exception("Publish failed")

def _is_dep_satisfied(dep_task_id: str) -> bool:
    """A dependency is satisfied only when the predecessor task is DONE."""
    with TASKS_LOCK:
        dep = TASKS.get(dep_task_id)
        return bool(dep and dep.status == TaskStatus.DONE)

def _dispatch_ready_tasks() -> None:
    """Scan and dispatch tasks whose dependencies are satisfied."""
    ready_ids = TASK_STORE.ready_to_dispatch(_is_dep_satisfied)
    for tid in ready_ids:
        with TASKS_LOCK:
            rec = TASKS.get(tid)
        parsed = TASK_STORE.get_parsed(tid)
        if not rec or not parsed:
            continue
        if rec.status != TaskStatus.PENDING:
            continue
        _try_dispatch_one(tid, parsed, rec)

# -------------------------
# Background loops
# -------------------------
def _dependency_watcher_loop(interval_sec: int = 2) -> None:
    """Periodic fallback scan (keeps behavior if events are delayed)."""
    while True:
        try:
            _dispatch_ready_tasks()
        except Exception as e:
            logger.warning("dependency watcher error: %s", e)
        time.sleep(interval_sec)

def _tasks_events_loop() -> None:
    """Subscribe to tasks.events and react to DONE/FAILED to chain/retarget."""
    while True:
        try:
            sub_rds = redis.from_url(REDIS_URL, decode_responses=True)  # separate connection
            pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("tasks.events")
            logger.info("Subscribed to tasks.events")
            for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                raw = msg.get("data")
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                except Exception as e:
                    logger.warning("Bad tasks.events payload: %s", e)
                    continue

                ev_type = str(data.get("type", "")).upper()
                task_id = str(data.get("task_id") or "")
                worker_id = data.get("worker_id")

                if not task_id:
                    continue

                with TASKS_LOCK:
                    rec = TASKS.get(task_id)

                if not rec:
                    # Unknown or already GC'd; ignore
                    continue

                if ev_type == 'TASK_SUCCEEDED':
                    # Mark DONE and attempt to dispatch any newly-unblocked tasks
                    logger.info("Task succeeded: %s", task_id)
                    with TASKS_LOCK:
                        rec.status = TaskStatus.DONE
                        rec.error = None
                    # Optionally flip worker status; worker typically sets itself IDLE already
                    if worker_id:
                        try:
                            update_worker_status(rds, worker_id, "IDLE")
                        except Exception:
                            pass
                    # Immediately attempt to dispatch following stages / dependents
                    _dispatch_ready_tasks()

                elif ev_type == 'TASK_FAILED':
                    logger.info("Task failed: %s", task_id)
                    # Mark FAILED, then try retry on a different worker (bounded retries)
                    err_msg = data.get("error") or "worker failed"
                    with TASKS_LOCK:
                        rec.status = TaskStatus.FAILED
                        rec.error = str(err_msg)
                        rec.retries += 1
                        retries_left = rec.max_retries - rec.retries

                    if retries_left >= 0:
                        parsed = TASK_STORE.get_parsed(task_id)
                        if parsed:
                            _try_dispatch_one(task_id, parsed, rec, exclude_worker_id=worker_id)
                    # If no parsed or no candidates, the rec remains FAILED
                else:
                    # Other event types can be ignored or logged
                    logger.debug("tasks.events ignoring type=%s payload=%s", ev_type, data)

        except Exception as e:
            logger.warning("tasks.events listener error: %s; reconnecting soon...", e)
            time.sleep(2)  # backoff then reconnect

@app.on_event("startup")
def _start_background_threads():
    threading.Thread(target=_dependency_watcher_loop, name="deps-watcher", daemon=True).start()
    threading.Thread(target=_tasks_events_loop, name="tasks-events", daemon=True).start()

# -------------------------
# Task submission
# -------------------------
@app.post("/api/v1/tasks")
async def submit_task(raw: Union[str, Dict[str, Any]] = Body(..., media_type="text/yaml"),
                      _: Any = Depends(require_auth)):
    """Submit YAML of a single task or a staged pipeline.

    - If spec.stages exists: generates N stage tasks with auto dependencies.
    - If not: registers a single task, honoring spec.dependsOn if present.

    Returns a summary for all created task_ids.
    """
    if isinstance(raw, dict) and "yaml" in raw:
        yml = str(raw["yaml"])
    elif isinstance(raw, str):
        yml = raw
    else:
        raise HTTPException(status_code=400, detail='Expected YAML string or {"yaml":"..."}')

    entries = TASK_STORE.parse_and_register(yml)  # List of stage or single entries
    results: List[Dict[str, Any]] = []

    for entry in entries:
        task_id = entry["task_id"]
        task = entry["parsed"]
        depends_on = entry["depends_on"]

        # Keep a record visible via query APIs
        with TASKS_LOCK:
            TASKS[task_id] = TaskRecord(task_id=task_id, raw_yaml=yml, parsed=task)

        # If no dependencies, try immediate dispatch; else report as waiting
        if not depends_on:
            with TASKS_LOCK:
                rec = TASKS[task_id]
            _try_dispatch_one(task_id, task, rec)

        with TASKS_LOCK:
            rec = TASKS[task_id]
            results.append({
                "task_id": task_id,
                "status": rec.status,
                "assigned_worker": rec.assigned_worker,
                "topic": rec.topic,
                "waiting_on": depends_on or [],
                "retries": rec.retries,
                "max_retries": rec.max_retries,
            })

    return {"ok": True, "count": len(entries), "tasks": results}

# -------------------------
# Task queries
# -------------------------
@app.get("/api/v1/tasks")
async def list_tasks(_: Any = Depends(require_auth)):
    with TASKS_LOCK:
        return [t.model_dump() for t in TASKS.values()]

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str = ApiPath(..., min_length=1), _: Any = Depends(require_auth)):
    with TASKS_LOCK:
        t = TASKS.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task not found")
        return t.model_dump()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = parse_int_env("PORT", 8080)
    logger.info("Starting on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
