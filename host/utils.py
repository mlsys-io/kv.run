from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from .event_schema import WorkerEvent


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(str(value).replace("_", ""))
    except Exception:
        return default


def parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_ts(value: Optional[str]) -> float:
    if not value:
        return time.time()
    try:
        v = value
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return time.time()


def safe_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_mem_to_bytes(mem: str) -> Optional[int]:
    if not isinstance(mem, str):
        return None
    match = re.match(r"^\s*([0-9]+)\s*([KkMmGgTt][Ii]?[Bb]?)?\s*$", mem)
    if not match:
        return None
    qty = int(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit in {"k", "kb", "kib"}:
        return qty * 1024
    if unit in {"m", "mb", "mib"}:
        return qty * 1024 ** 2
    if unit in {"g", "gb", "gib"}:
        return qty * 1024 ** 3
    if unit in {"t", "tb", "tib"}:
        return qty * 1024 ** 4
    if unit == "":
        return qty
    return None


def get_logger(
    name: str,
    log_file: str,
    *,
    max_bytes: int,
    backup_count: int,
    level: str,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fh = RotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)
    return logger


def log_worker_event(logger, event: WorkerEvent) -> None:
    event_type = event.type
    worker_id = event.worker_id
    if event_type == "REGISTER":
        logger.info(
            "Worker registered: id=%s status=%s tags=%s",
            worker_id,
            event.status,
            event.tags,
        )
    elif event_type == "UNREGISTER":
        logger.info("Worker unregistered: id=%s", worker_id)
    elif event_type == "STATUS":
        logger.debug("Worker status: id=%s status=%s", worker_id, event.status)
    elif event_type == "HEARTBEAT":
        logger.debug("Worker heartbeat: id=%s", worker_id)


WORKERS_SET = "workers:ids"


def r_worker_key(worker_id: str) -> str:
    return f"worker:{worker_id}"


def r_hb_key(worker_id: str) -> str:
    return f"worker:{worker_id}:hb"
