# main.py
"""FastAPI Orchestrator Instance."""

from __future__ import annotations

import atexit
import os
import json
import copy
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import redis
from fastapi import FastAPI, HTTPException, Path as ApiPath, Depends, Request, UploadFile, File
from fastapi.responses import FileResponse

from utils import (
    parse_int_env,
    parse_float_env,
    get_logger,
    log_worker_event,
    now_iso,
    parse_iso_ts,
)
from task_store import TaskStore
from worker_cls import (
    list_workers_from_redis,
    get_worker_from_redis,
    update_worker_status,
    is_stale_by_redis,
)
from task import TaskRecord, TaskStatus, categorize_task_type
from dispatch import DispatchManager
from aggregation import maybe_aggregate_parent
from results import write_result, read_result, result_file_path, ResultPayload
from state_store import StateManager
from manifest_utils import sync_manifest, ARTIFACTS_DIR
from event_schema import parse_event, TaskEvent, WorkerEvent
from metrics import MetricsRecorder

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

STATE_DIR = Path(os.getenv("ORCHESTRATOR_STATE_DIR", RESULTS_DIR.parent / "state")).expanduser().resolve()
STATE_FLUSH_INTERVAL = parse_float_env("STATE_FLUSH_INTERVAL_SEC", 5.0)
STATE_MANAGER = StateManager(
    state_dir=STATE_DIR,
    task_store=TASK_STORE,
    tasks=TASKS,
    tasks_lock=TASKS_LOCK,
    parent_shards=PARENT_SHARDS,
    child_to_parent=CHILD_TO_PARENT,
    logger=logger,
    flush_interval=STATE_FLUSH_INTERVAL,
)
METRICS_DIR = STATE_DIR / "metrics"
METRICS_RECORDER = MetricsRecorder(METRICS_DIR, logger)

def _export_metrics_on_exit() -> None:
    try:
        result = METRICS_RECORDER.export_final_report()
        report = result.get("report") or {}
        summary = METRICS_RECORDER.format_report(report)
        if summary:
            logger.info(summary)
        path = result.get("path")
        if path:
            logger.info("Metrics report saved to %s", path)
    except Exception as exc:  # noqa: broad-except
        logger.warning("Failed to export final metrics on shutdown: %s", exc)

atexit.register(_export_metrics_on_exit)

_restored_tasks = STATE_MANAGER.load_snapshot()
if _restored_tasks:
    logger.info("Restoring %d pending tasks from snapshot", len(_restored_tasks))
    restored_count = 0
    for task_id in _restored_tasks:
        payload = TASK_STORE.get_parsed(task_id)
        if not payload:
            logger.warning("Skip requeue for %s: missing parsed spec", task_id)
            continue
        with TASKS_LOCK:
            rec = TASKS.get(task_id)
        if not rec:
            logger.warning("Skip requeue for %s: missing TaskRecord", task_id)
            continue
        try:
            DISPATCHER.enqueue_for_dispatch(task_id, payload, rec)
            restored_count += 1
        except Exception as exc:
            logger.exception("Failed to enqueue restored task %s: %s", task_id, exc)
    if restored_count:
        TASK_STORE.flush_pool()
        logger.info("Requeued %d tasks after restoration", restored_count)
        STATE_MANAGER.mark_dirty()

# -------------------------
# App & routes
# -------------------------
app = FastAPI(title="Orchestrator", version="1.0.0")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/metrics")
async def get_metrics(_: Any = Depends(require_auth)):
    return METRICS_RECORDER.snapshot()

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

    content, child_payloads = _split_merged_result(task_id, content)

    file_path = write_result(RESULTS_DIR, task_id, content)
    logger.info("Stored result for task %s at %s", task_id, file_path)

    with TASKS_LOCK:
        rec = TASKS.get(task_id)
        if rec:
            rec.status = TaskStatus.DONE
            rec.error = None

    if child_payloads:
        _finalize_merge_children(task_id, child_payloads)

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

    expected_artifacts: List[str] = []
    if rec:
        expected_artifacts = ((rec.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
    sync_manifest(file_path.parent, task_id, expected_artifacts)
    STATE_MANAGER.mark_dirty()
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
    target_path = (base_dir / ARTIFACTS_DIR / filename.name).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    (base_dir / ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    with target_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.info("Stored artifact for task %s at %s", safe_task_id, target_path)
    expected_artifacts: List[str] = []
    with TASKS_LOCK:
        rec = TASKS.get(safe_task_id)
        if rec:
            expected_artifacts = ((rec.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
    sync_manifest(base_dir, safe_task_id, expected_artifacts)
    STATE_MANAGER.mark_dirty()
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

                try:
                    event = parse_event(data)
                except Exception as exc:
                    logger.warning("Failed to parse task event payload: %s (%s)", data, exc)
                    continue
                if not isinstance(event, TaskEvent):
                    logger.debug("Ignoring non-task event on tasks channel: %s", data)
                    continue

                ev_type = event.type
                task_id = event.task_id
                worker_id = event.worker_id
                err_msg = event.error
                METRICS_RECORDER.record_task_event(event)

                if not task_id:
                    continue

                with TASKS_LOCK:
                    rec = TASKS.get(task_id)
                parent_id = CHILD_TO_PARENT.get(task_id)
                state_dirty = False

                if ev_type == "TASK_STARTED":
                    with TASKS_LOCK:
                        rec = TASKS.get(task_id)
                        if rec:
                            payload = event.payload or {}
                            start_iso = payload.get("started_at") or event.ts
                            rec.started_ts = parse_iso_ts(start_iso)
                            dispatch_iso = payload.get("dispatched_at")
                            if dispatch_iso:
                                rec.dispatched_ts = parse_iso_ts(dispatch_iso)
                            elif not rec.dispatched_ts:
                                rec.dispatched_ts = rec.started_ts
                            rec.attempts = int(rec.attempts or 0) + 1
                    continue

                if ev_type == 'TASK_SUCCEEDED':
                    logger.info("Task succeeded: %s", task_id)
                    TASK_STORE.clear_from_pool(task_id)
                    state_dirty = True
                    if rec:
                        with TASKS_LOCK:
                            rec.status = TaskStatus.DONE
                            rec.error = None
                            payload = event.payload or {}
                            rec.finished_ts = parse_iso_ts(payload.get("finished_at") or event.ts)
                            rec.started_ts = parse_iso_ts(payload.get("started_at") or event.ts)
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
                    if state_dirty:
                        STATE_MANAGER.mark_dirty()
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
                            payload = event.payload or {}
                            rec.finished_ts = parse_iso_ts(payload.get("finished_at") or event.ts)
                            rec.started_ts = parse_iso_ts(payload.get("started_at") or event.ts)
                        state_dirty = True

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
                            METRICS_RECORDER.record_task_event(
                                TaskEvent(
                                    type="TASK_REQUEUED",
                                    task_id=task_id,
                                    worker_id=worker_id,
                                    error=reason,
                                )
                            )
                            with TASKS_LOCK:
                                parent_rec = TASKS.get(parent_id)
                                if parent_rec:
                                    parent_rec.status = TaskStatus.WAITING
                                    parent_rec.error = f"Waiting on shard retry: {task_id}"
                            TASK_STORE.flush_pool()
                            if state_dirty:
                                STATE_MANAGER.mark_dirty()
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
                            METRICS_RECORDER.finalize_task_failure(task_id)
                            TASK_STORE.flush_pool()
                            if state_dirty:
                                STATE_MANAGER.mark_dirty()
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
                            METRICS_RECORDER.record_task_event(
                                TaskEvent(
                                    type="TASK_REQUEUED",
                                    task_id=task_id,
                                    worker_id=worker_id,
                                    error=reason,
                                )
                            )
                            STATE_MANAGER.mark_dirty()
                        continue

                    if rec:
                        with TASKS_LOCK:
                            rec.next_retry_at = None
                        _propagate_merge_failure(rec)
                        _record_dead_letter(task_id, rec, worker_id, err_msg, parent_id=parent_id)
                        state_dirty = True
                    METRICS_RECORDER.finalize_task_failure(task_id)
                    TASK_STORE.flush_pool()
                    if state_dirty:
                        STATE_MANAGER.mark_dirty()
                    continue

                else:
                    logger.debug("tasks.events ignoring type=%s payload=%s", ev_type, data)

        except Exception as e:
            logger.warning("tasks.events listener error: %s; reconnecting soon...", e)
            time.sleep(2)


def _split_merged_result(task_id: str, content: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    slices = TASK_STORE.get_merge_slices(task_id)
    if not slices:
        return content, {}

    base_content = copy.deepcopy(content)
    base_result = copy.deepcopy(base_content.get("result") or {})
    items: List[Any] = list(base_result.get("items") or [])
    total_items = len(items)

    parent_slice = slices.get(task_id, (0, total_items))
    parent_content = copy.deepcopy(content)
    parent_content["result"] = _slice_result_section(base_result, items, parent_slice, total_items)
    parent_meta = dict(parent_content.get("metadata") or {})
    parent_meta.setdefault("merged_children", [cid for cid in slices.keys() if cid != task_id])
    parent_content["metadata"] = parent_meta

    child_payloads: Dict[str, Dict[str, Any]] = {}
    for child_id, slc in slices.items():
        if child_id == task_id:
            continue
        child_content = copy.deepcopy(content)
        child_content["task_id"] = child_id
        child_content["result"] = _slice_result_section(base_result, items, slc, total_items)
        child_meta = dict(child_content.get("metadata") or {})
        child_meta["merged_from"] = task_id
        child_meta["merged_slice"] = {"start": int(slc[0]), "end": int(slc[1])}
        child_content["metadata"] = child_meta
        child_payloads[child_id] = child_content

    return parent_content, child_payloads


def _slice_result_section(
    base_result: Dict[str, Any],
    items: List[Any],
    slc: Tuple[int, int],
    total_items: int,
) -> Dict[str, Any]:
    start = max(0, int(slc[0]))
    end = max(start, int(slc[1]))
    if total_items:
        end = min(end, total_items)

    subset = items[start:end] if items else []
    result = copy.deepcopy(base_result)
    result["items"] = subset

    usage = base_result.get("usage")
    if isinstance(usage, dict):
        result["usage"] = _scale_usage(usage, total_items, len(subset))

    if "num_requests" in result:
        result["num_requests"] = len(subset)
    return result


def _scale_usage(usage: Dict[str, Any], total_items: int, portion: int) -> Dict[str, Any]:
    if portion <= 0 or total_items <= 0:
        return copy.deepcopy(usage)
    ratio = portion / float(total_items)
    scaled: Dict[str, Any] = {}
    for key, value in usage.items():
        if key in {"num_requests", "requests"}:
            scaled[key] = portion
            continue
        if isinstance(value, (int, float)):
            computed = value * ratio
            if isinstance(value, int):
                scaled[key] = max(0, int(round(computed)))
            else:
                scaled[key] = computed
        else:
            scaled[key] = copy.deepcopy(value)
    return scaled


def _finalize_merge_children(parent_id: str, child_payloads: Dict[str, Dict[str, Any]]) -> None:
    if not child_payloads:
        with TASKS_LOCK:
            parent_rec = TASKS.get(parent_id)
            if parent_rec:
                parent_rec.merged_children = None
                parent_rec.merge_slice = None
        TASK_STORE.clear_merge(parent_id)
        return

    with TASKS_LOCK:
        parent_rec = TASKS.get(parent_id)
        if parent_rec:
            parent_rec.merged_children = None
            parent_rec.merge_slice = None

    for child_id, payload in child_payloads.items():
        try:
            path = write_result(RESULTS_DIR, child_id, payload)
            logger.info("Stored split result for merged task %s -> child %s at %s", parent_id, child_id, path)
        except Exception as exc:
            logger.warning("Failed to store split result for child %s derived from %s: %s", child_id, parent_id, exc)
            continue

        with TASKS_LOCK:
            child_rec = TASKS.get(child_id)
            if child_rec:
                child_rec.status = TaskStatus.DONE
                child_rec.error = None
                child_rec.merged_parent_id = None
                child_rec.merge_slice = None
                expected_child_artifacts = ((child_rec.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
            else:
                expected_child_artifacts = []
        sync_manifest(path.parent, child_id, expected_child_artifacts)
        TASK_STORE.mark_released(child_id)

    TASK_STORE.clear_merge(parent_id)
    STATE_MANAGER.mark_dirty()


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
                try:
                    event = parse_event(data)
                except Exception as exc:
                    logger.debug("Failed to parse worker event: %s (%s)", data, exc)
                    continue
                if not isinstance(event, WorkerEvent):
                    logger.debug("Ignoring non-worker event on workers channel: %s", data)
                    continue
                log_worker_event(logger, event)
                METRICS_RECORDER.record_worker_event(event)

                if event.type == "UNREGISTER":
                    _reschedule_tasks_for_worker(event.worker_id)
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
    STATE_MANAGER.mark_dirty()


def _propagate_merge_failure(parent_rec: TaskRecord) -> None:
    children = list(parent_rec.merged_children or [])
    if not children:
        return

    parent_id = parent_rec.task_id
    error_msg = parent_rec.error or "Merged parent failed"

    with TASKS_LOCK:
        parent_rec.merged_children = None
        parent_rec.merge_slice = None

    for child in children:
        child_id = child.get("task_id")
        if not child_id:
            continue
        with TASKS_LOCK:
            child_rec = TASKS.get(child_id)
            if child_rec:
                child_rec.status = TaskStatus.FAILED
                child_rec.error = error_msg
                child_rec.merged_parent_id = None
                child_rec.merge_slice = None
        TASK_STORE.mark_released(child_id)
        logger.info("Propagated merged failure from %s to child %s", parent_id, child_id)

    TASK_STORE.clear_merge(parent_id)
    STATE_MANAGER.mark_dirty()


def _propagate_merge_failure(parent_rec: TaskRecord) -> None:
    children = list(parent_rec.merged_children or [])
    if not children:
        return

    parent_id = parent_rec.task_id
    error_msg = parent_rec.error or "Merged parent failed"

    with TASKS_LOCK:
        parent_rec.merged_children = None
        parent_rec.merge_slice = None

    for child in children:
        child_id = child.get("task_id")
        if not child_id:
            continue
        with TASKS_LOCK:
            child_rec = TASKS.get(child_id)
            if child_rec:
                child_rec.status = TaskStatus.FAILED
                child_rec.error = error_msg
                child_rec.merged_parent_id = None
                child_rec.merge_slice = None
        TASK_STORE.mark_released(child_id)
        logger.info("Propagated merged failure from %s to child %s", parent_id, child_id)

    TASK_STORE.clear_merge(parent_id)

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

    changed = False
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
        changed = True

        if parent_id:
            with TASKS_LOCK:
                parent_rec = TASKS.get(parent_id)
                if parent_rec and parent_rec.status == TaskStatus.DISPATCHED:
                    parent_rec.status = TaskStatus.WAITING
                    parent_rec.error = f"Waiting on shard retry: {task_id}"
                    changed = True
    if changed:
        STATE_MANAGER.mark_dirty()
    
@asynccontextmanager
async def _lifespan(_: FastAPI):
    STATE_MANAGER.start()
    try:
        try:
            TASK_STORE.flush_pool()
        except Exception as exc:
            logger.warning("Initial dispatch scan failed: %s", exc)
        threading.Thread(target=_tasks_events_loop, name="tasks-events", daemon=True).start()
        threading.Thread(target=_workers_events_loop, name="workers-events", daemon=True).start()
        yield
    finally:
        STATE_MANAGER.stop()


app.router.lifespan_context = _lifespan

# -------------------------
# Task submission & queries
# -------------------------
@app.post("/api/v1/tasks")
async def submit_task(request: Request, _: Any = Depends(require_auth)):
    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=400, detail="Request body is required")

    content_type = (request.headers.get("content-type") or "").lower()
    yml: Optional[str] = None

    if "application/json" in content_type:
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}")

        if isinstance(data, dict) and "yaml" in data:
            yml = str(data["yaml"])
        elif isinstance(data, str):
            yml = data
        else:
            raise HTTPException(status_code=400, detail='Expected YAML string or {"yaml":"..."} in JSON body')
    else:
        try:
            yml = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Body must be UTF-8 encoded text")

    if yml is None or yml.strip() == "":
        raise HTTPException(status_code=400, detail="YAML payload cannot be empty")

    entries = TASK_STORE.parse_and_register(yml)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        task_id = entry["task_id"]
        task = entry["parsed"]
        depends_on = entry["depends_on"]
        slo_seconds = entry.get("slo_seconds")
        task_type = (task.get("spec") or {}).get("taskType")
        category = categorize_task_type(task_type)

        with TASKS_LOCK:
            rec_obj = TaskRecord(
                task_id=task_id,
                raw_yaml=yml,
                parsed=task,
                graph_node_name=entry.get("graph_node_name"),
                load=int(entry.get("load", 0) or 0),
                slo_seconds=slo_seconds,
                task_type=task_type,
                category=category,
            )
            rec_obj.last_queue_ts = rec_obj.submitted_ts
            TASKS[task_id] = rec_obj

        DISPATCHER.enqueue_for_dispatch(task_id, task, rec_obj)
        METRICS_RECORDER.record_task_event(
            TaskEvent(
                type="TASK_SUBMITTED",
                task_id=task_id,
                payload={
                    "taskType": task_type,
                    "sloSeconds": slo_seconds,
                },
            )
        )

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
    STATE_MANAGER.mark_dirty()
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
