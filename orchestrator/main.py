# main.py
"""FastAPI Orchestrator"""

from __future__ import annotations

import os
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Set


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
    update_worker_status,
    is_stale_by_redis,
)
from retry_queue import RetryQueueManager
from worker_events import log_worker_event
from scheduler import (
    estimate_queue_length,
    choose_strategy,
    select_workers_for_task,
    make_shard_messages,
    idle_satisfying_pool,
    is_data_parallel_enabled,
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

RESULTS_DIR = Path(os.getenv("ORCHESTRATOR_RESULTS_DIR", "./orchestrator-results")).expanduser().resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
RETRY_BASE_DELAY_SEC = parse_int_env("RETRY_BASE_DELAY_SEC", 5)
RETRY_MAX_DELAY_SEC = parse_int_env("RETRY_MAX_DELAY_SEC", 300)
QUEUE_LOG_INTERVAL_SEC = parse_int_env("QUEUE_LOG_INTERVAL_SEC", 60)

class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"
    DONE = "DONE"
    WAITING = "WAITING"

class TaskRecord(BaseModel):
    task_id: str
    raw_yaml: str
    parsed: Dict[str, Any]
    status: str = TaskStatus.PENDING
    assigned_worker: Optional[str] = None  # "MULTI" for sharded parent
    topic: Optional[str] = None
    submitted_at: str = Field(default_factory=now_iso)
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    parent_task_id: Optional[str] = None
    shard_index: Optional[int] = None
    shard_total: Optional[int] = None
    next_retry_at: Optional[str] = None
    last_failed_worker: Optional[str] = None

TASKS: Dict[str, TaskRecord] = {}
TASKS_LOCK = threading.RLock()

PARENT_SHARDS: Dict[str, Dict[str, Any]] = {}
CHILD_TO_PARENT: Dict[str, str] = {}

TASK_STORE = TaskStore()
RETRY_MANAGER = RetryQueueManager(
    tasks=TASKS,
    tasks_lock=TASKS_LOCK,
    task_store=TASK_STORE,
    logger=logger,
    task_status=TaskStatus,
    base_delay_sec=RETRY_BASE_DELAY_SEC,
    max_delay_sec=RETRY_MAX_DELAY_SEC,
)


class ResultPayload(BaseModel):
    task_id: str
    result: Dict[str, Any]
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    received_at: str = Field(default_factory=now_iso)

# -------------------------
# App & routes
# -------------------------
app = FastAPI(title="Orchestrator", version="1.3.0")

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


def _result_file_path(task_id: str) -> Path:
    """Return a filesystem path for a task's saved result (sanitize task_id)."""
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
    return RESULTS_DIR / safe_id / "responses.json"

# -------------------------
# Result ingestion
# -------------------------
@app.post("/api/v1/results")
async def ingest_result(payload: ResultPayload, _: Any = Depends(require_auth)):
    task_id = payload.task_id.strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    file_path = _result_file_path(task_id)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "task_id": task_id,
        "worker_id": payload.worker_id,
        "metadata": payload.metadata,
        "received_at": payload.received_at,
        "result": payload.result,
    }

    file_path.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    logger.info("Stored result for task %s at %s", task_id, file_path)

    with TASKS_LOCK:
        rec = TASKS.get(task_id)
        if rec:
            rec.status = TaskStatus.DONE
            rec.error = None

    return {"ok": True, "path": str(file_path)}


# -------------------------
# Dispatch helpers
# -------------------------
def _try_dispatch_sharded(
    parent_task_id: str,
    base_task: Dict[str, Any],
    parent_rec: TaskRecord,
    exclude_ids: Optional[Set[str]] = None,
) -> bool:
    """Attempt data-parallel dispatch (only when inference+parallel.enabled)."""
    # ---- Switch: Enable only when taskType=='inference' and parallel.enabled==true ----
    if not is_data_parallel_enabled(base_task):
        return False

    queued = estimate_queue_length(TASK_STORE)
    pool = idle_satisfying_pool(rds, base_task, exclude_ids=exclude_ids)
    if not pool:
        return False

    strategy = choose_strategy(base_task, pool, queued)
    if strategy.mode != "data_parallel":
        return False

    workers = select_workers_for_task(pool, shard_count=strategy.shard_count, prefer_best=strategy.prefer_best)
    if len(workers) <= 1:
        return False

    child_msgs = make_shard_messages(parent_task_id, base_task, workers)

    topic = "tasks"
    created_children: List[str] = []
    for idx, (child_id, child_task, worker) in enumerate(child_msgs):
        message = {
            "task_id": child_id,
            "task": child_task,
            "task_type": safe_get(child_task, "spec.taskType"),
            "assigned_worker": worker.worker_id,
            "dispatched_at": now_iso(),
        }
        try:
            _publish_task(topic, message)
            update_worker_status(rds, worker.worker_id, "RUNNING")
            child_rec = TaskRecord(
                task_id=child_id,
                raw_yaml=parent_rec.raw_yaml,
                parsed=child_task,
                status=TaskStatus.DISPATCHED,
                assigned_worker=worker.worker_id,
                topic=topic,
                parent_task_id=parent_task_id,
                shard_index=idx,
                shard_total=len(workers),
                max_retries=parent_rec.max_retries,
            )
            with TASKS_LOCK:
                TASKS[child_id] = child_rec
                CHILD_TO_PARENT[child_id] = parent_task_id
            created_children.append(child_id)
        except Exception as e:
            with TASKS_LOCK:
                TASKS[child_id] = TaskRecord(
                    task_id=child_id,
                    raw_yaml=parent_rec.raw_yaml,
                    parsed=child_task,
                    status=TaskStatus.FAILED,
                    error=f"Publish failed: {e}",
                    parent_task_id=parent_task_id,
                    shard_index=idx,
                    shard_total=len(workers),
                )
            logger.exception("Shard publish failed")

    if not created_children:
        return False

    with TASKS_LOCK:
        PARENT_SHARDS[parent_task_id] = {
            "total": len(created_children),
            "done": 0,
            "failed": 0,
            "children": set(created_children),
        }
        parent_rec.status = TaskStatus.DISPATCHED
        parent_rec.assigned_worker = "MULTI"
        parent_rec.topic = topic

    TASK_STORE.mark_released(parent_task_id)
    return True

def _try_dispatch_one(task_id: str, task: Dict[str, Any], rec: TaskRecord,
                      exclude_worker_id: Optional[str] = None) -> None:
    """Scheduler-only dispatch: data-parallel if capacity allows, else single."""
    with TASKS_LOCK:
        if rec.status == TaskStatus.DISPATCHED:
            return

    exclude_ids: Set[str] = set()
    if exclude_worker_id:
        exclude_ids.add(exclude_worker_id)

    # Try data-parallel first
    if _try_dispatch_sharded(task_id, task, rec, exclude_ids=exclude_ids):
        return

    # Single-worker path via scheduler
    queued = estimate_queue_length(TASK_STORE)
    pool = idle_satisfying_pool(rds, task, exclude_ids=exclude_ids)
    if not pool:
        logger.info("No suitable IDLE worker for %s; scheduling retry", task_id)
        RETRY_MANAGER.schedule(task_id, rec, "No suitable IDLE worker; retrying shortly")
        return

    strategy = choose_strategy(task, pool, queued)  # decides best-fit vs first-fit
    worker_list = select_workers_for_task(pool, shard_count=1, prefer_best=strategy.prefer_best)
    if not worker_list:
        logger.info("Worker selection failed for %s; scheduling retry", task_id)
        RETRY_MANAGER.schedule(task_id, rec, "No worker passed selection; retrying shortly")
        return

    worker = worker_list[0]
    topic = "tasks"
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
            rec.next_retry_at = None
            rec.last_failed_worker = None
        TASK_STORE.mark_released(task_id)
    except Exception as e:
        logger.warning("Publish failed for %s; scheduling retry", task_id)
        RETRY_MANAGER.schedule(task_id, rec, f"Publish failed: {e}")

def _is_dep_satisfied(dep_task_id: str) -> bool:
    with TASKS_LOCK:
        dep = TASKS.get(dep_task_id)
        return bool(dep and dep.status == TaskStatus.DONE)

def _dispatch_ready_tasks() -> None:
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
    last_log = 0.0
    while True:
        try:
            _dispatch_ready_tasks()
        except Exception as e:
            logger.warning("dependency watcher error: %s", e)
        now = time.time()
        if QUEUE_LOG_INTERVAL_SEC > 0 and now - last_log >= QUEUE_LOG_INTERVAL_SEC:
            RETRY_MANAGER.log_snapshot()
            last_log = now
        time.sleep(interval_sec)

def _retry_queue_loop() -> None:
    RETRY_MANAGER.run_loop(_try_dispatch_one)

def _tasks_events_loop() -> None:
    """Subscribe to tasks.events and handle shard aggregation & retries."""
    while True:
        try:
            sub_rds = redis.from_url(REDIS_URL, decode_responses=True)
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
                err_msg = data.get("error")

                if not task_id:
                    continue

                with TASKS_LOCK:
                    rec = TASKS.get(task_id)
                parent_id = CHILD_TO_PARENT.get(task_id)

                if ev_type == 'TASK_SUCCEEDED':
                    logger.info("Task succeeded: %s", task_id)
                    if rec:
                        with TASKS_LOCK:
                            rec.status = TaskStatus.DONE
                            rec.error = None
                    if worker_id:
                        try:
                            update_worker_status(rds, worker_id, "IDLE")
                        except Exception:
                            pass

                    if parent_id:
                        with TASKS_LOCK:
                            aggr = PARENT_SHARDS.get(parent_id)
                            if aggr:
                                aggr["done"] += 1
                                if aggr["done"] >= aggr["total"]:
                                    parent_rec = TASKS.get(parent_id)
                                    if parent_rec:
                                        parent_rec.status = TaskStatus.DONE
                                        parent_rec.error = None
                                    _dispatch_ready_tasks()
                        continue
                    else:
                        _dispatch_ready_tasks()
                        continue

                elif ev_type == 'TASK_FAILED':
                    logger.info("Task failed: %s", task_id)
                    retries_left = -1
                    if rec:
                        with TASKS_LOCK:
                            rec.status = TaskStatus.FAILED
                            rec.error = str(err_msg or "worker failed")
                            rec.retries += 1
                            rec.last_failed_worker = worker_id
                            retries_left = rec.max_retries - rec.retries

                    if parent_id:
                        if rec and retries_left >= 0:
                            reason = rec.error or "Shard failed; retrying"
                            RETRY_MANAGER.schedule(task_id, rec, reason, exclude_worker_id=worker_id)
                            with TASKS_LOCK:
                                parent_rec = TASKS.get(parent_id)
                                if parent_rec:
                                    parent_rec.status = TaskStatus.WAITING
                                    parent_rec.error = f"Waiting on shard retry: {task_id}"
                            continue

                        with TASKS_LOCK:
                            parent_rec = TASKS.get(parent_id)
                            if parent_rec:
                                parent_rec.status = TaskStatus.FAILED
                                parent_rec.error = f"Child shard failed: {task_id}"
                            if rec:
                                rec.next_retry_at = None
                        continue

                    if rec and retries_left >= 0:
                        parsed = TASK_STORE.get_parsed(task_id)
                        if parsed:
                            reason = rec.error or "Task failed; retrying"
                            RETRY_MANAGER.schedule(task_id, rec, reason, exclude_worker_id=worker_id)
                        continue

                    if rec:
                        with TASKS_LOCK:
                            rec.next_retry_at = None

                else:
                    logger.debug("tasks.events ignoring type=%s payload=%s", ev_type, data)

        except Exception as e:
            logger.warning("tasks.events listener error: %s; reconnecting soon...", e)
            time.sleep(2)


def _workers_events_loop() -> None:
    """Listen to workers.events for registration/heartbeat updates."""
    while True:
        try:
            sub_rds = redis.from_url(REDIS_URL, decode_responses=True)
            pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("workers.events")
            logger.info("Subscribed to workers.events")
            for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                raw = msg.get("data")
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                except Exception as exc:
                    logger.debug("Bad workers.events payload: %s", exc)
                    continue
                log_worker_event(logger, data)
        except Exception as exc:
            logger.warning("workers.events listener error: %s; reconnecting soon...", exc)
            time.sleep(2)

@asynccontextmanager
async def _lifespan(_: FastAPI):
    threading.Thread(target=_dependency_watcher_loop, name="deps-watcher", daemon=True).start()
    threading.Thread(target=_tasks_events_loop, name="tasks-events", daemon=True).start()
    threading.Thread(target=_retry_queue_loop, name="retry-queue", daemon=True).start()
    threading.Thread(target=_workers_events_loop, name="workers-events", daemon=True).start()
    yield


app.router.lifespan_context = _lifespan

# -------------------------
# Task submission & queries
# -------------------------
@app.post("/api/v1/tasks")
async def submit_task(raw: Union[str, Dict[str, Any]] = Body(..., media_type="text/yaml"),
                      _: Any = Depends(require_auth)):
    if isinstance(raw, dict) and "yaml" in raw:
        yml = str(raw["yaml"])
    elif isinstance(raw, str):
        yml = raw
    else:
        raise HTTPException(status_code=400, detail='Expected YAML string or {"yaml":"..."}')

    entries = TASK_STORE.parse_and_register(yml)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        task_id = entry["task_id"]
        task = entry["parsed"]
        depends_on = entry["depends_on"]

        with TASKS_LOCK:
            TASKS[task_id] = TaskRecord(task_id=task_id, raw_yaml=yml, parsed=task)

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

if __name__ == "__main__":
    import uvicorn
    port = parse_int_env("PORT", 8080)
    logger.info("Starting on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
