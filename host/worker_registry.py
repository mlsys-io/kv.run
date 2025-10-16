from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from .utils import (
    WORKERS_SET,
    now_iso,
    parse_mem_to_bytes,
    safe_get,
    r_hb_key,
    r_worker_key,
)


class Worker(BaseModel):
    worker_id: str
    status: str = Field(default="UNKNOWN")
    started_at: Optional[str] = None
    pid: Optional[int] = None
    env: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    last_seen: Optional[str] = None
    cached_models: List[str] = Field(default_factory=list)
    cached_datasets: List[str] = Field(default_factory=list)


def is_stale_by_redis(rds, worker_id: str) -> bool:
    ttl = rds.ttl(r_hb_key(worker_id))
    return (ttl is None) or (ttl < 0)


def get_worker_from_redis(rds, worker_id: str) -> Optional[Worker]:
    raw = rds.hgetall(r_worker_key(worker_id))
    if not raw:
        return None

    def _loads(value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def _ensure_str_list(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                norm = item.strip()
                if norm:
                    result.append(norm)
        return result

    env = _loads(raw.get("env_json"), {})
    hardware = _loads(raw.get("hardware_json"), {})
    tags = _loads(raw.get("tags_json"), [])
    cached_models = _ensure_str_list(_loads(raw.get("cache_models_json"), []))
    cached_datasets = _ensure_str_list(_loads(raw.get("cache_datasets_json"), []))

    pid_val = raw.get("pid")
    try:
        pid = int(pid_val) if pid_val is not None else None
    except (TypeError, ValueError):
        pid = None

    return Worker(
        worker_id=raw.get("worker_id", worker_id),
        status=raw.get("status", "UNKNOWN"),
        started_at=raw.get("started_at") or None,
        pid=pid,
        env=env,
        hardware=hardware,
        tags=tags,
        last_seen=raw.get("last_seen") or None,
        cached_models=cached_models,
        cached_datasets=cached_datasets,
    )


def list_workers_from_redis(rds) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for worker_id in list(rds.smembers(WORKERS_SET)):
        worker = get_worker_from_redis(rds, worker_id)
        if not worker:
            continue
        stale = is_stale_by_redis(rds, worker_id)
        if not worker.last_seen:
            hb_ts = rds.get(r_hb_key(worker_id))
            if hb_ts:
                worker.last_seen = hb_ts
        results.append({**worker.model_dump(), "stale": stale})
    return results


def update_worker_status(rds, worker_id: str, status: str) -> None:
    payload = {
        "type": "STATUS",
        "worker_id": worker_id,
        "status": status,
        "ts": now_iso(),
    }
    with rds.pipeline() as pipe:
        pipe.hset(r_worker_key(worker_id), mapping={"status": status, "last_seen": payload["ts"]})
        pipe.publish("workers.events", json.dumps(payload, ensure_ascii=False))
        pipe.execute()


def _normalize_cache_value(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _merge_unique(existing: List[str], additions: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for value in existing:
        normalized = _normalize_cache_value(value)
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        merged.append(normalized)
    for value in additions:
        normalized = _normalize_cache_value(value)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged


def record_worker_cache(
    rds,
    worker_id: str,
    *,
    models: Optional[Iterable[str]] = None,
    datasets: Optional[Iterable[str]] = None,
) -> None:
    models_list = [v for v in (models or []) if _normalize_cache_value(v)]
    datasets_list = [v for v in (datasets or []) if _normalize_cache_value(v)]
    if not models_list and not datasets_list:
        return

    key = r_worker_key(worker_id)
    if not rds.exists(key):
        return

    try:
        current_models_json, current_datasets_json = rds.hmget(
            key,
            "cache_models_json",
            "cache_datasets_json",
        )
    except Exception:
        return

    try:
        current_models = json.loads(current_models_json) if current_models_json else []
        if not isinstance(current_models, list):
            current_models = []
    except Exception:
        current_models = []

    try:
        current_datasets = json.loads(current_datasets_json) if current_datasets_json else []
        if not isinstance(current_datasets, list):
            current_datasets = []
    except Exception:
        current_datasets = []

    updated_models = _merge_unique(current_models, models_list)
    updated_datasets = _merge_unique(current_datasets, datasets_list)

    mapping: Dict[str, str] = {}
    if updated_models != current_models:
        mapping["cache_models_json"] = json.dumps(updated_models, ensure_ascii=False)
    if updated_datasets != current_datasets:
        mapping["cache_datasets_json"] = json.dumps(updated_datasets, ensure_ascii=False)
    if not mapping:
        return

    try:
        rds.hset(key, mapping=mapping)
    except Exception:
        return


def idle_satisfying_pool(rds, task: Dict[str, Any]) -> List[Worker]:
    available: List[Worker] = []
    for worker_id in rds.smembers(WORKERS_SET):
        worker = get_worker_from_redis(rds, worker_id)
        if not worker or worker.status != "IDLE":
            continue
        if is_stale_by_redis(rds, worker.worker_id):
            continue
        if _hw_satisfies(worker, task):
            available.append(worker)
    return available


def sort_workers(workers: List[Worker]) -> List[Worker]:
    decorated = []
    for worker in workers:
        gpus = safe_get(worker.hardware, "gpu.gpus", []) or safe_get(worker.hardware, "gpus", []) or []
        gpu_count = len(gpus) if isinstance(gpus, list) else 0
        total_vram = 0
        if gpu_count > 0:
            for gpu in gpus:
                mem = (
                    safe_get(gpu, "memory.total_bytes")
                    or safe_get(gpu, "memory_bytes")
                    or safe_get(gpu, "vram_bytes")
                    or safe_get(gpu, "memory.total")
                    or safe_get(gpu, "vram")
                )
                bytes_val = parse_mem_to_bytes(mem) if isinstance(mem, str) else mem
                try:
                    total_vram += int(bytes_val or 0)
                except Exception:
                    total_vram += 0
        sys_ram = int(safe_get(worker.hardware, "memory.total_bytes", 0) or 0)
        cpu_cores = int(safe_get(worker.hardware, "cpu.logical_cores", 0) or 0)
        decorated.append((worker, gpu_count, total_vram, sys_ram, cpu_cores))

    decorated.sort(key=lambda item: (item[1] > 0, item[2], item[3], item[4]), reverse=True)
    return [item[0] for item in decorated]


def _hw_satisfies(worker: Worker, task: Dict[str, Any]) -> bool:
    requirements = safe_get(task, "spec.resources.hardware", {}) or {}
    if not requirements:
        return True

    if not isinstance(requirements, dict):
        return True

    hw = worker.hardware or {}
    cpu_needed = requirements.get("cpu")
    mem_needed = requirements.get("memory")
    gpu_req = requirements.get("gpu") or {}

    if cpu_needed is not None:
        try:
            cpu_needed = int(cpu_needed)
        except Exception:
            cpu_needed = None
        cpu_cores = safe_get(hw, "cpu.logical_cores")
        try:
            cpu_cores = int(cpu_cores) if cpu_cores is not None else None
        except Exception:
            cpu_cores = None
        if cpu_needed is not None and (cpu_cores is None or cpu_cores < cpu_needed):
            return False

    if mem_needed:
        required_bytes = parse_mem_to_bytes(str(mem_needed)) or 0
        available = safe_get(hw, "memory.total_bytes")
        try:
            available = int(available) if available is not None else 0
        except Exception:
            available = 0
        if available < required_bytes:
            return False

    if isinstance(gpu_req, dict) and gpu_req:
        required_count = gpu_req.get("count")
        if required_count is not None:
            try:
                required_count = int(required_count)
            except Exception:
                required_count = None
        required_type = str(gpu_req.get("type") or "").strip().lower()
        if required_type in {"", "any", "auto", "*"}:
            required_type = ""
        gpu_info = safe_get(hw, "gpu.gpus", []) or safe_get(hw, "gpus", [])
        if isinstance(gpu_info, list) and gpu_info:
            if required_count is not None and len(gpu_info) < required_count:
                return False
            if required_type:
                pattern = re.compile(re.escape(required_type), re.IGNORECASE)
                if not any(pattern.search(str(gpu.get("name") or gpu.get("type") or "")) for gpu in gpu_info):
                    return False
        else:
            count = safe_get(hw, "gpu.count") or safe_get(hw, "gpu_count") or safe_get(hw, "gpu.info.count")
            try:
                count = int(count)
            except Exception:
                count = 0
            if required_count is not None and count < required_count:
                return False
            type_value = (
                safe_get(hw, "gpu.type")
                or safe_get(hw, "gpu_info.type")
                or safe_get(hw, "gpuType")
            )
            if required_type and not str(type_value or "").strip().lower().startswith(required_type):
                return False

    return True
