# main.py
"""FastAPI Orchestrator Instance."""

from __future__ import annotations

import os
import json
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import redis
from fastapi import FastAPI, HTTPException, Path as ApiPath, Depends, Request, Body, UploadFile, File
from fastapi.responses import FileResponse

from utils import (
    parse_int_env,
    parse_float_env,
    get_logger,
    log_worker_event,
    now_iso,
)
from task_store import TaskStore
from worker_cls import (
    list_workers_from_redis,
    get_worker_from_redis,
    update_worker_status,
    is_stale_by_redis,
)
from task import TaskRecord, TaskStatus
from dispatch import DispatchManager
from aggregation import maybe_aggregate_parent
from results import write_result, read_result, result_file_path, ResultPayload

# -------------------------
# Settings & globals
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

# results directory
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results_host")).expanduser().resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Redis connection
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
try:
    rds = redis.from_url(REDIS_URL, decode_responses=True)
    rds.ping()
    logger.info("Connected to Redis: %s", REDIS_URL)
except Exception as e:
    logger.exception("Failed to connect to Redis: %s", e)
    raise SystemExit(1)

# -------------------------
# Authentication
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
TASK_POOL_BATCH_SIZE = parse_int_env("TASK_POOL_BATCH_SIZE", 5)
TASK_SLO_DISPATCH_THRESHOLD = max(0.0, min(1.0, parse_float_env("TASK_SLO_THRESHOLD", 0.5)))

TASKS: Dict[str, TaskRecord] = {}
TASKS_LOCK = threading.RLock()

PARENT_SHARDS: Dict[str, Dict[str, Any]] = {}
CHILD_TO_PARENT: Dict[str, str] = {}

TASK_STORE = TaskStore()
DISPATCHER = DispatchManager(
    task_store=TASK_STORE,
    redis_client=rds,
    tasks=TASKS,
    tasks_lock=TASKS_LOCK,
    parent_shards=PARENT_SHARDS,
    child_to_parent=CHILD_TO_PARENT,
    results_dir=RESULTS_DIR,
    logger=logger,
)
DISPATCHER.configure_task_store(
    batch_size=TASK_POOL_BATCH_SIZE,
    slo_fraction=TASK_SLO_DISPATCH_THRESHOLD,
    pending_status=TaskStatus.PENDING,
    done_status=TaskStatus.DONE,
)

# -------------------------
# App & routes
# -------------------------
app = FastAPI(title="Orchestrator", version="1.0.0")

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

# -------------------------
# Result ingestion
# -------------------------
@app.post("/api/v1/results")
async def ingest_result(payload: ResultPayload, _: Any = Depends(require_auth)):
    task_id = payload.task_id.strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    content = {
        "task_id": task_id,
        "worker_id": payload.worker_id,
        "metadata": payload.metadata,
        "received_at": payload.received_at,
        "result": payload.result,
    }

    file_path = write_result(RESULTS_DIR, task_id, content)
    logger.info("Stored result for task %s at %s", task_id, file_path)

    with TASKS_LOCK:
        rec = TASKS.get(task_id)
        if rec:
            rec.status = TaskStatus.DONE
            rec.error = None

    maybe_aggregate_parent(
        task_id,
        content,
        child_to_parent=CHILD_TO_PARENT,
        parent_shards=PARENT_SHARDS,
        tasks=TASKS,
        tasks_lock=TASKS_LOCK,
        results_dir=RESULTS_DIR,
        logger=logger,
    )

    return {"ok": True, "path": str(file_path)}


# -------------------------
# Result retrieval
# -------------------------
@app.get("/api/v1/results/{task_id}")
async def get_result(task_id: str, _: Any = Depends(require_auth)):
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    try:
        content = read_result(RESULTS_DIR, task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="result not found")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"task_id": task_id, "raw": content}

    return data


@app.post("/api/v1/results/{task_id}/files")
async def upload_result_file(
    task_id: str,
    file: UploadFile = File(...),
    _: Any = Depends(require_auth),
):
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    filename = Path(file.filename or "")
    if filename.name != file.filename or filename.name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir = result_file_path(RESULTS_DIR, safe_task_id).parent
    target_path = (base_dir / filename.name).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir.mkdir(parents=True, exist_ok=True)

    with target_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.info("Stored artifact for task %s at %s", safe_task_id, target_path)
    return {"ok": True, "path": str(target_path)}


@app.get("/api/v1/results/{task_id}/files/{filename}")
async def download_result_file(
    task_id: str,
    filename: str,
    _: Any = Depends(require_auth),
):
    sanitized = Path(filename)
    if sanitized.name != filename or sanitized.name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir = result_file_path(RESULTS_DIR, task_id).parent
    target_path = (base_dir / sanitized.name).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(target_path)

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
                    TASK_STORE.clear_from_pool(task_id)
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
                    TASK_STORE.flush_pool()
                    continue

                elif ev_type == 'TASK_FAILED':
                    logger.info("Task failed: %s", task_id)
                    TASK_STORE.clear_from_pool(task_id)
                    TASK_STORE.forget_worker(worker_id)
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
                            DISPATCHER.requeue_task(
                                task_id,
                                rec,
                                reason,
                                task=rec.parsed,
                                exclude_worker_id=worker_id,
                            )
                            with TASKS_LOCK:
                                parent_rec = TASKS.get(parent_id)
                                if parent_rec:
                                    parent_rec.status = TaskStatus.WAITING
                                    parent_rec.error = f"Waiting on shard retry: {task_id}"
                            TASK_STORE.flush_pool()
                            continue
                        else:
                            with TASKS_LOCK:
                                parent_rec = TASKS.get(parent_id)
                                if parent_rec:
                                    parent_rec.status = TaskStatus.FAILED
                                    parent_rec.error = f"Child shard failed: {task_id}"
                                if rec:
                                    rec.next_retry_at = None
                            _record_dead_letter(task_id, rec, worker_id, err_msg, parent_id=parent_id)
                            TASK_STORE.flush_pool()
                            continue
                    elif rec and retries_left >= 0:
                        parsed = TASK_STORE.get_parsed(task_id)
                        if parsed:
                            reason = rec.error or "Task failed; retrying"
                            DISPATCHER.requeue_task(
                                task_id,
                                rec,
                                reason,
                                task=parsed,
                                exclude_worker_id=worker_id,
                            )
                        continue

                    if rec:
                        with TASKS_LOCK:
                            rec.next_retry_at = None
                        _record_dead_letter(task_id, rec, worker_id, err_msg, parent_id=parent_id)
                    TASK_STORE.flush_pool()
                    continue

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

                ev_type = str(data.get("type", "")).upper()
                if ev_type == "UNREGISTER":
                    _reschedule_tasks_for_worker(data.get("worker_id"))
        except Exception as exc:
            logger.warning("workers.events listener error: %s; reconnecting soon...", exc)
            time.sleep(2)


def _record_dead_letter(
    task_id: str,
    rec: Optional[TaskRecord],
    worker_id: Optional[str],
    err_msg: Optional[str],
    *,
    parent_id: Optional[str] = None,
) -> None:
    """
    Persist a terminal failure entry for observability/triage.
    Called when retry budget is exhausted.
    """
    if not task_id:
        return
    message = (
        (rec.error if rec and rec.error else None)
        or (str(err_msg) if err_msg else None)
        or "task failed without error detail"
    )
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "parent_task_id": parent_id,
        "last_failed_worker": worker_id,
        "error": message,
        "retries": getattr(rec, "retries", None) if rec else None,
        "max_retries": getattr(rec, "max_retries", None) if rec else None,
        "finalized_at": now_iso(),
    }
    if rec:
        payload["load"] = rec.load
        payload["slo_seconds"] = rec.slo_seconds
    TASK_STORE.record_dead_letter(payload)

def _reschedule_tasks_for_worker(worker_id: Optional[str]) -> None:
    """Move dispatched tasks back to waiting when their worker disappears."""
    safe_worker_id = (worker_id or "").strip()
    if not safe_worker_id:
        return
    TASK_STORE.forget_worker(safe_worker_id)

    to_reschedule: List[tuple[str, Optional[str]]] = []
    with TASKS_LOCK:
        for task_id, rec in TASKS.items():
            if rec.status == TaskStatus.DISPATCHED and rec.assigned_worker == safe_worker_id:
                parent_id = CHILD_TO_PARENT.get(task_id)
                to_reschedule.append((task_id, parent_id))

    if not to_reschedule:
        return

    for task_id, parent_id in to_reschedule:
        with TASKS_LOCK:
            rec = TASKS.get(task_id)
        if not rec or rec.assigned_worker != safe_worker_id:
            continue

        DISPATCHER.requeue_task(
            task_id,
            rec,
            f"Worker {safe_worker_id} lost; retry via task pool",
            task=rec.parsed,
            exclude_worker_id=safe_worker_id,
        )
        logger.info("Rescheduling %s after worker %s unregistered", task_id, safe_worker_id)

        if parent_id:
            with TASKS_LOCK:
                parent_rec = TASKS.get(parent_id)
                if parent_rec and parent_rec.status == TaskStatus.DISPATCHED:
                    parent_rec.status = TaskStatus.WAITING
                    parent_rec.error = f"Waiting on shard retry: {task_id}"
    
@asynccontextmanager
async def _lifespan(_: FastAPI):
    try:
        TASK_STORE.flush_pool()
    except Exception as exc:
        logger.warning("Initial dispatch scan failed: %s", exc)
    threading.Thread(target=_tasks_events_loop, name="tasks-events", daemon=True).start()
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
        slo_seconds = entry.get("slo_seconds")

        with TASKS_LOCK:
            rec_obj = TaskRecord(
                task_id=task_id,
                raw_yaml=yml,
                parsed=task,
                graph_node_name=entry.get("graph_node_name"),
                load=int(entry.get("load", 0) or 0),
                slo_seconds=slo_seconds,
            )
            TASKS[task_id] = rec_obj

        DISPATCHER.enqueue_for_dispatch(task_id, task, rec_obj)

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
                "load": rec.load,
            })

    TASK_STORE.flush_pool()
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


@app.get("/api/v1/dead_letters")
async def list_dead_letters(_: Any = Depends(require_auth)):
    return TASK_STORE.list_dead_letters()

if __name__ == "__main__":
    import uvicorn
    port = parse_int_env("PORT", 8000)
    logger.info("Starting on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
