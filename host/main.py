from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse
from pydantic import BaseModel

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "host"
    sys.modules.setdefault("host.main", sys.modules[__name__])

from .event_schema import TaskEvent, WorkerEvent, parse_event
from .dispatcher_factory import DEFAULT_DISPATCH_MODE, create_dispatcher
from .worker_selector import DEFAULT_WORKER_SELECTION
from .manifest_utils import ARTIFACTS_DIR, sync_manifest
from .metrics import MetricsRecorder
from .results import ResultPayload, read_result, result_file_path, write_result
from .task_metadata import extract_model_dataset_names
from .task_runtime import TaskRuntime
from .utils import get_logger, log_worker_event, now_iso, parse_bool_env, parse_float_env, parse_int_env, safe_get
from .elastic import ElasticCoordinator
from .worker_registry import (
    get_worker_from_redis,
    is_stale_by_redis,
    list_workers_from_redis,
    sort_workers,
    update_worker_status,
    record_worker_cache,
)
from .watchdog import WorkerWatchdog


# --------------------------------------------------------------------------- #
# Environment & globals
# --------------------------------------------------------------------------- #

LOG_FILE = os.getenv("LOG_FILE", "host.log")
LOG_MAX_BYTES = parse_int_env("LOG_MAX_BYTES", 5_242_880)
LOG_BACKUP_COUNT = parse_int_env("LOG_BACKUP_COUNT", 5)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = get_logger(
    name="host",
    log_file=LOG_FILE,
    max_bytes=LOG_MAX_BYTES,
    backup_count=LOG_BACKUP_COUNT,
    level=LOG_LEVEL,
)

# Result & metrics directories
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results_host")).expanduser().resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

metrics_base_default = RESULTS_DIR.parent / "metrics"
metrics_env = os.getenv("HOST_METRICS_DIR") or os.getenv("ORCHESTRATOR_METRICS_DIR")
METRICS_DIR = Path(metrics_env or metrics_base_default).expanduser().resolve()

# Redis connection
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
try:
    RDS = redis.from_url(REDIS_URL, decode_responses=True)
    RDS.ping()
    logger.info("Connected to Redis: %s", REDIS_URL)
except Exception as exc:  # pragma: no cover
    logger.exception("Failed to connect to Redis: %s", exc)
    raise SystemExit(1)

# Authentication
ORCH_BEARER = os.getenv("ORCHESTRATOR_TOKEN")


async def require_auth(request: Request):
    if not ORCH_BEARER:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if token != ORCH_BEARER:
        raise HTTPException(status_code=403, detail="Invalid token")


# Task runtime & metrics
RUNTIME = TaskRuntime(logger)
dispatch_mode = os.getenv("ORCHESTRATOR_DISPATCH_MODE", DEFAULT_DISPATCH_MODE)
worker_selection = os.getenv("ORCHESTRATOR_WORKER_SELECTION", DEFAULT_WORKER_SELECTION)
ENABLE_CONTEXT_REUSE = parse_bool_env("ENABLE_CONTEXT_REUSE", True)
ENABLE_TASK_MERGE = parse_bool_env("ENABLE_TASK_MERGE", True)
TASK_MERGE_MAX_BATCH_SIZE = max(1, parse_int_env("TASK_MERGE_MAX_BATCH_SIZE", 4))
ENABLE_ELASTIC_SCALING = parse_bool_env("ENABLE_ELASTIC_SCALING", True)
ENABLE_STAGE_WEIGHT_STICKINESS = parse_bool_env("ENABLE_STAGE_WEIGHT_STICKINESS", False)
WORKER_CACHE_TTL_SEC = max(0, parse_int_env("WORKER_CACHE_TTL_SEC", 3600))
SCHED_LAMBDA_INFERENCE = parse_float_env("SCHEDULER_LAMBDA_INFERENCE", 0.4)
SCHED_LAMBDA_TRAINING = parse_float_env("SCHEDULER_LAMBDA_TRAINING", 0.8)
SCHED_LAMBDA_OTHER = parse_float_env("SCHEDULER_LAMBDA_OTHER", 0.5)
SCHED_SELECTION_JITTER = parse_float_env("SCHEDULER_SELECTION_JITTER", 1e-3)
AUTO_DISABLE_IDLE_SEC = max(0, parse_int_env("ELASTIC_AUTO_DISABLE_IDLE_SEC", 60))
AUTO_ENABLE_QUEUE_THRESHOLD = max(0, parse_int_env("ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD", 5))
AUTO_DISABLE_QUEUE_MAX = max(0, parse_int_env("ELASTIC_AUTO_DISABLE_QUEUE_MAX", 2))
AUTO_POLL_INTERVAL_SEC = max(5, parse_int_env("ELASTIC_AUTO_POLL_INTERVAL_SEC", 30))
AUTO_TOGGLE_COOLDOWN_SEC = max(
    30,
    parse_int_env("ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC", max(60, AUTO_POLL_INTERVAL_SEC * 2)),
)
AUTO_MIN_ACTIVE_WORKERS = max(0, parse_int_env("ELASTIC_AUTO_MIN_ACTIVE_WORKERS", 1))
ENABLE_WORKER_WATCHDOG = parse_bool_env("ENABLE_WORKER_WATCHDOG", True)
WORKER_DEATH_CHECK_INTERVAL = max(5, parse_int_env("WORKER_DEATH_CHECK_INTERVAL", 30))
WORKER_DEATH_GRACE_SEC = max(0, parse_int_env("WORKER_DEATH_GRACE_SEC", 60))

PENDING_RESULT_CLONES: Dict[str, List[str]] = {}
PENDING_RESULT_CLONES_LOCK = threading.RLock()

METRICS_RECORDER = MetricsRecorder(METRICS_DIR, logger)
ELASTIC_COORDINATOR = ElasticCoordinator(
    RDS,
    METRICS_RECORDER,
    RUNTIME,
    enabled=ENABLE_ELASTIC_SCALING,
    auto_disable_idle_secs=AUTO_DISABLE_IDLE_SEC,
    auto_enable_queue_threshold=AUTO_ENABLE_QUEUE_THRESHOLD,
    auto_disable_queue_max=AUTO_DISABLE_QUEUE_MAX,
    auto_poll_interval=AUTO_POLL_INTERVAL_SEC,
    auto_toggle_cooldown=AUTO_TOGGLE_COOLDOWN_SEC,
    min_active_workers=AUTO_MIN_ACTIVE_WORKERS,
    logger=logger,
)
ELASTIC_COORDINATOR.start()

DISPATCHER = create_dispatcher(
    dispatch_mode,
    RUNTIME,
    RDS,
    RESULTS_DIR,
    logger=logger,
    worker_selection_strategy=worker_selection,
    enable_context_reuse=ENABLE_CONTEXT_REUSE,
    enable_task_merge=ENABLE_TASK_MERGE,
    task_merge_max_batch_size=TASK_MERGE_MAX_BATCH_SIZE,
    elastic_coordinator=ELASTIC_COORDINATOR,
    reuse_cache_ttl_sec=WORKER_CACHE_TTL_SEC,
    enable_stage_weight_stickiness=ENABLE_STAGE_WEIGHT_STICKINESS,
    metrics_recorder=METRICS_RECORDER,
    lambda_config={
        "inference": SCHED_LAMBDA_INFERENCE,
        "training": SCHED_LAMBDA_TRAINING,
        "other": SCHED_LAMBDA_OTHER,
    },
    selection_jitter_epsilon=SCHED_SELECTION_JITTER,
)
STOP_EVENT = threading.Event()
BACKGROUND_THREADS: List[threading.Thread] = []
REDIS_FLUSH_LOCK = threading.Lock()
REDIS_FLUSHED = False
WATCHDOG = WorkerWatchdog(
    RDS,
    RUNTIME,
    DISPATCHER,
    logger,
    enabled=ENABLE_WORKER_WATCHDOG,
    check_interval=WORKER_DEATH_CHECK_INTERVAL,
    grace_seconds=WORKER_DEATH_GRACE_SEC,
)


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
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to export metrics summary: %s", exc)


def _clear_redis_state() -> None:
    global REDIS_FLUSHED
    with REDIS_FLUSH_LOCK:
        if REDIS_FLUSHED:
            return
        try:
            RDS.flushdb()
            logger.info("Cleared Redis database at %s", REDIS_URL)
            REDIS_FLUSHED = True
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to clear Redis database on exit: %s", exc)


def _clear_redis_on_exit() -> None:
    _clear_redis_state()


atexit.register(_clear_redis_on_exit)
atexit.register(_export_metrics_on_exit)


# --------------------------------------------------------------------------- #
# Background loops
# --------------------------------------------------------------------------- #

def _dispatch_loop() -> None:
    DISPATCHER.dispatch_loop(STOP_EVENT, poll_interval=1.0)


def _tasks_events_loop() -> None:
    while not STOP_EVENT.is_set():
        try:
            sub_rds = redis.from_url(REDIS_URL, decode_responses=True)
            pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("tasks.events")
            logger.info("Subscribed to tasks.events")
            for msg in pubsub.listen():
                if STOP_EVENT.is_set():
                    break
                if msg.get("type") != "message":
                    continue
                raw = msg.get("data")
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                except Exception as exc:
                    logger.debug("Invalid tasks.events payload: %s", exc)
                    continue
                try:
                    event = parse_event(data)
                except Exception as exc:
                    logger.debug("Failed to parse task event: %s (%s)", data, exc)
                    continue
                if not isinstance(event, TaskEvent):
                    continue
                _handle_task_event(event)
            try:
                pubsub.close()
            except Exception:
                pass
        except Exception as exc:
            if STOP_EVENT.is_set():
                break
            logger.warning("tasks.events listener error: %s; reconnecting...", exc)
            time.sleep(2.0)


def _handle_task_event(event: TaskEvent) -> None:
    payload = event.payload or {}
    event_type = event.type
    if event_type == "TASK_STARTED":
        METRICS_RECORDER.record_task_event(event)
        RUNTIME.mark_started(event.task_id, event.worker_id, payload, event.ts)
    elif event_type == "TASK_SUCCEEDED":
        METRICS_RECORDER.record_task_event(event)
        ready_children, merged_children = RUNTIME.mark_succeeded(event.task_id, event.worker_id, payload, event.ts)
        try:
            queueing, dispatched, pending, done, total = RUNTIME.task_status_counts()
            summary = (
                f"QUEUEING {queueing}, DISPATCHED {dispatched}, PENDING {pending}, "
                f"DONE {done}, TOTAL {total}"
            )
        except Exception:
            summary = (
                "QUEUEING UNKNOWN, DISPATCHED UNKNOWN, PENDING UNKNOWN, "
                "DONE UNKNOWN, TOTAL UNKNOWN"
            )
        logger.info("Task %s completed; %s", event.task_id, summary)
        if merged_children:
            for child_id in merged_children:
                child_payload = dict(payload)
                child_payload["parent_task_id"] = event.task_id
                child_payload["is_child_task"] = True
                child_event = TaskEvent(
                    type="TASK_SUCCEEDED",
                    task_id=child_id,
                    worker_id=event.worker_id,
                    payload=child_payload,
                    ts=event.ts,
                )
                METRICS_RECORDER.record_task_event(child_event, is_child=True)
        if event.worker_id:
            try:
                record = RUNTIME.get_record(event.task_id)
                if record:
                    models, datasets = extract_model_dataset_names(record.parsed)
                    if models or datasets:
                        record_worker_cache(RDS, event.worker_id, models=models, datasets=datasets)
            except Exception as exc:
                logger.debug("Failed to update cache metadata for worker %s: %s", event.worker_id, exc)
            try:
                update_worker_status(RDS, event.worker_id, "IDLE")
            except Exception:
                pass
        if merged_children:
            _mirror_task_results(event.task_id, merged_children)
    elif event_type == "TASK_FAILED":
        record = RUNTIME.get_record(event.task_id)
        attempts = record.attempts if record else 0
        max_attempts = record.max_attempts if record else None
        can_retry = (
            record
            and (max_attempts is None or max_attempts < 0 or attempts < max_attempts)
        )
        if can_retry:
            limit_display = "∞" if max_attempts is None or max_attempts < 0 else max_attempts
            if event.worker_id:
                record.last_failed_worker = event.worker_id
            logger.warning(
                "Retrying task %s after worker failure (%d/%s)",
                event.task_id,
                attempts + 1,
                limit_display,
            )
            DISPATCHER._requeue_task(
                event.task_id,
                reason="worker_failed",
                front=True,
                extra_payload={
                    "error": event.error,
                    "attempt": attempts + 1,
                    "max_attempts": max_attempts,
                },
            )
            return

        METRICS_RECORDER.record_task_event(event)
        impacted, merged_children = RUNTIME.mark_failed(
            event.task_id,
            event.worker_id,
            payload,
            event.ts,
            error=event.error,
        )
        METRICS_RECORDER.finalize_task_failure(event.task_id)
        for task_id, reason in impacted:
            derived = TaskEvent(
                type="TASK_FAILED",
                task_id=task_id,
                error=reason,
                payload={"dependency_failure": event.task_id},
            )
            METRICS_RECORDER.record_task_event(derived)
            METRICS_RECORDER.finalize_task_failure(task_id)
        for child_id in merged_children:
            child_payload = dict(payload)
            child_payload["parent_task_id"] = event.task_id
            child_payload["dependency_failure"] = event.task_id
            child_payload["is_child_task"] = True
            child_event = TaskEvent(
                type="TASK_FAILED",
                task_id=child_id,
                worker_id=event.worker_id,
                error=event.error or "parent_failed",
                payload=child_payload,
            )
            METRICS_RECORDER.record_task_event(child_event, is_child=True)
            METRICS_RECORDER.finalize_task_failure(child_id)
    else:
        logger.debug("Ignoring task event type=%s payload=%s", event_type, payload)


def _mirror_task_results(parent_task_id: str, child_ids: List[str]) -> None:
    if not child_ids:
        return
    parent_dir = result_file_path(RESULTS_DIR, parent_task_id).parent
    if not parent_dir.exists():
        with PENDING_RESULT_CLONES_LOCK:
            pending = PENDING_RESULT_CLONES.setdefault(parent_task_id, [])
            for child in child_ids:
                if child not in pending:
                    pending.append(child)
        logger.debug("Deferring result mirroring for %s (waiting for artifacts)", parent_task_id)
        return

    for child_id in child_ids:
        if child_id == parent_task_id:
            continue
        dst_dir = result_file_path(RESULTS_DIR, child_id).parent
        if dst_dir.exists() and (dst_dir / "responses.json").exists():
            continue
        try:
            if dst_dir.exists():
                shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(parent_dir, dst_dir)
            record = RUNTIME.get_record(child_id)
            expected_artifacts: List[str] = []
            if record:
                expected_artifacts = ((record.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
            sync_manifest(dst_dir, child_id, expected_artifacts)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to mirror results from %s to %s: %s", parent_task_id, child_id, exc)

    with PENDING_RESULT_CLONES_LOCK:
        PENDING_RESULT_CLONES.pop(parent_task_id, None)


def _workers_events_loop() -> None:
    while not STOP_EVENT.is_set():
        try:
            sub_rds = redis.from_url(REDIS_URL, decode_responses=True)
            pubsub = sub_rds.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("workers.events")
            logger.info("Subscribed to workers.events")
            for msg in pubsub.listen():
                if STOP_EVENT.is_set():
                    break
                if msg.get("type") != "message":
                    continue
                raw = msg.get("data")
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                except Exception as exc:
                    logger.debug("Invalid workers.events payload: %s", exc)
                    continue
                try:
                    event = parse_event(data)
                except Exception as exc:
                    logger.debug("Failed to parse worker event: %s (%s)", data, exc)
                    continue
                if not isinstance(event, WorkerEvent):
                    continue
                log_worker_event(logger, event)
                METRICS_RECORDER.record_worker_event(event)
                ELASTIC_COORDINATOR.record_worker_event(event)
                if event.type == "UNREGISTER":
                    worker_id = (event.worker_id or "").strip()
                    if worker_id:
                        if WATCHDOG.enabled and WATCHDOG.is_marked_dead(worker_id):
                            logger.info(
                                "Skipping direct requeue for %s unregister; watchdog already emitted synthetic failures",
                                worker_id,
                            )
                            WATCHDOG.clear_dead_mark(worker_id)
                            continue
                        recovered = RUNTIME.recover_tasks_for_worker(worker_id)
                        if recovered:
                            logger.info(
                                "Requeued %d task(s) after worker %s unregistered: %s",
                                len(recovered),
                                worker_id,
                                ", ".join(recovered),
                            )
                            for task_id in recovered:
                                record = RUNTIME.get_record(task_id)
                                if record:
                                    record.last_failed_worker = worker_id
                                DISPATCHER._requeue_task(
                                    task_id,
                                    reason="worker_unregistered",
                                    front=True,
                                    extra_payload={"worker": worker_id},
                                )
            try:
                pubsub.close()
            except Exception:
                pass
        except Exception as exc:
            if STOP_EVENT.is_set():
                break
            logger.warning("workers.events listener error: %s; reconnecting...", exc)
            time.sleep(2.0)
def _start_background_threads() -> None:
    threads = [
        threading.Thread(target=_dispatch_loop, name="dispatch-loop", daemon=True),
        threading.Thread(target=_tasks_events_loop, name="tasks-events", daemon=True),
        threading.Thread(target=_workers_events_loop, name="workers-events", daemon=True),
    ]
    for thread in threads:
        thread.start()
        BACKGROUND_THREADS.append(thread)

    watchdog_thread = WATCHDOG.start(STOP_EVENT)
    if watchdog_thread and watchdog_thread not in BACKGROUND_THREADS:
        BACKGROUND_THREADS.append(watchdog_thread)


def _stop_background_threads() -> None:
    STOP_EVENT.set()
    RUNTIME.shutdown()
    ELASTIC_COORDINATOR.shutdown()
    for thread in BACKGROUND_THREADS:
        thread.join(timeout=2.0)
    _clear_redis_state()


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

app = FastAPI(title="Host Orchestrator", version="0.1.0")


class WorkerElasticUpdate(BaseModel):
    enabled: bool


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/metrics")
async def get_metrics(_: Any = Depends(require_auth)):
    return METRICS_RECORDER.snapshot()


@app.get("/workers")
async def list_workers(_: Any = Depends(require_auth)):
    disabled = ELASTIC_COORDINATOR.disabled_workers() if ELASTIC_COORDINATOR.enabled else None
    return list_workers_from_redis(RDS, disabled_workers=disabled)


@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str, _: Any = Depends(require_auth)):
    worker = get_worker_from_redis(RDS, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="worker not found")
    stale = is_stale_by_redis(RDS, worker.worker_id)
    elastic_disabled = ELASTIC_COORDINATOR.is_disabled(worker.worker_id) if ELASTIC_COORDINATOR.enabled else False
    return {**worker.model_dump(), "stale": stale, "elastic_disabled": elastic_disabled}


@app.post("/workers/{worker_id}/elastic")
async def update_worker_elastic_state(
    worker_id: str,
    body: WorkerElasticUpdate,
    _: Any = Depends(require_auth),
):
    if not ELASTIC_COORDINATOR.enabled:
        raise HTTPException(status_code=400, detail="Elastic scaling control is disabled")
    worker = get_worker_from_redis(RDS, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="worker not found")
    try:
        ELASTIC_COORDINATOR.set_worker_state(worker_id, enabled=body.enabled)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state = {"worker_id": worker_id, "enabled": body.enabled, "elastic_disabled": not body.enabled}
    return state


def _parse_submission_body(raw_body: bytes, content_type: str) -> str:
    if not raw_body:
        raise HTTPException(status_code=400, detail="Request body is required")

    if "application/json" in content_type:
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc
        if isinstance(data, dict) and "yaml" in data:
            return str(data["yaml"])
        if isinstance(data, str):
            return data
        raise HTTPException(status_code=400, detail='Expected YAML string or {"yaml":"..."} in JSON body')

    try:
        return raw_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Body must be UTF-8 encoded text") from exc


@app.post("/api/v1/tasks")
async def submit_task(request: Request, _: Any = Depends(require_auth)):
    raw_body = await request.body()
    content_type = (request.headers.get("content-type") or "").lower()
    yaml_text = _parse_submission_body(raw_body, content_type)
    if not yaml_text.strip():
        raise HTTPException(status_code=400, detail="YAML payload cannot be empty")

    entries = RUNTIME.register_yaml(yaml_text)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        task_id = entry["task_id"]
        info = RUNTIME.describe_task(task_id)
        if not info:
            continue
        task_type = safe_get(info, "parsed.spec.taskType")
        METRICS_RECORDER.record_task_event(
            TaskEvent(
                type="TASK_SUBMITTED",
                task_id=task_id,
                payload={
                    "taskType": task_type,
                },
            )
        )
        results.append(
            {
                "task_id": task_id,
                "status": info.get("status"),
                "assigned_worker": info.get("assigned_worker"),
                "topic": info.get("topic"),
                "waiting_on": info.get("pending_dependencies", []),
                "depends_on": info.get("depends_on", []),
                "attempts": info.get("attempts"),
                "max_attempts": info.get("max_attempts"),
                "load": info.get("load"),
            }
        )

    return {"ok": True, "count": len(entries), "tasks": results}


@app.get("/api/v1/tasks")
async def list_tasks(_: Any = Depends(require_auth)):
    return RUNTIME.list_tasks()


@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str = ApiPath(..., min_length=1), _: Any = Depends(require_auth)):
    info = RUNTIME.describe_task(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="task not found")
    return info


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

    try:
        path = write_result(RESULTS_DIR, task_id, content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store result: {exc}") from exc

    expected_artifacts: List[str] = []
    record = RUNTIME.get_record(task_id)
    if record:
        expected_artifacts = ((record.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
    sync_manifest(path.parent, task_id, expected_artifacts)
    with PENDING_RESULT_CLONES_LOCK:
        pending_children = PENDING_RESULT_CLONES.pop(task_id, [])
    if pending_children:
        _mirror_task_results(task_id, pending_children)
    return {"ok": True, "path": str(path)}


@app.get("/api/v1/results/{task_id}")
async def get_result(task_id: str, _: Any = Depends(require_auth)):
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    try:
        raw = read_result(RESULTS_DIR, task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="result not found")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"task_id": task_id, "raw": raw}


@app.post("/api/v1/results/{task_id}/files")
async def upload_result_file(task_id: str, file: UploadFile = File(...), _: Any = Depends(require_auth)):
    sanitized = Path(file.filename or "")
    if sanitized.name != file.filename or sanitized.name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir = result_file_path(RESULTS_DIR, task_id).parent
    target_path = (base_dir / ARTIFACTS_DIR / sanitized.name).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    (base_dir / ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("wb") as out:
            out.write(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store artifact: {exc}") from exc

    record = RUNTIME.get_record(task_id)
    expected_artifacts: List[str] = []
    if record:
        expected_artifacts = ((record.parsed or {}).get("spec") or {}).get("output", {}).get("artifacts", []) or []
    sync_manifest(base_dir, task_id, expected_artifacts)
    return {"ok": True, "path": str(target_path)}


@app.get("/api/v1/results/{task_id}/files/{filename}")
async def download_result_file(task_id: str, filename: str, _: Any = Depends(require_auth)):
    sanitized = Path(filename)
    if sanitized.name != filename or sanitized.name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir = result_file_path(RESULTS_DIR, task_id).parent
    artifact_dir = base_dir / ARTIFACTS_DIR
    target_path = (artifact_dir / sanitized.name).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    if not target_path.exists() or not target_path.is_file():
        fallback = (base_dir / sanitized.name).resolve()
        try:
            fallback.relative_to(base_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid filename")
        if not fallback.exists() or not fallback.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        target_path = fallback

    return FileResponse(target_path)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _start_background_threads()
    try:
        yield
    finally:
        _stop_background_threads()


app.router.lifespan_context = _lifespan


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the MLOC orchestrator service.")
    parser.add_argument(
        "--host",
        help="Bind address (defaults to HOST_APP_HOST env or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port (defaults to HOST_APP_PORT, then PORT env, else 8000).",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable auto-reload (overrides HOST_APP_RELOAD).",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload.",
    )
    parser.set_defaults(reload=None)
    parser.add_argument(
        "--log-level",
        help="Uvicorn log level (defaults to HOST_APP_LOG_LEVEL or LOG_LEVEL).",
    )

    args = parser.parse_args(argv)

    host_value = args.host or os.getenv("HOST_APP_HOST", "0.0.0.0")
    port_default = parse_int_env("PORT", 8000)
    port_value = args.port if args.port is not None else parse_int_env("HOST_APP_PORT", port_default)

    reload_default = parse_bool_env("HOST_APP_RELOAD", False)
    reload_enabled = reload_default if args.reload is None else args.reload

    log_level_env = args.log_level or os.getenv("HOST_APP_LOG_LEVEL") or os.getenv("LOG_LEVEL", "info")
    log_level = str(log_level_env).lower()

    uvicorn_app = app if not reload_enabled else "host.main:app"

    uvicorn.run(
        uvicorn_app,
        host=host_value,
        port=port_value,
        reload=reload_enabled,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
