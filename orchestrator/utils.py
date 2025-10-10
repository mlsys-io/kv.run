# utils.py
"""Utility helpers for the Orchestrator service.

This module centralizes small reusable helpers such as environment parsing,
ISO timestamp generation, safe dictionary access, memory string parsing,
logger configuration, and Redis key helpers.
"""
from __future__ import annotations

import os
import re
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from event_schema import WorkerEvent


# -------------------------
# Environment & time helpers
# -------------------------

def parse_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable.

    Supports underscores in numeric literals (e.g. "5_242_880"). If the value
    is missing or invalid, returns the provided default.
    """
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val.replace("_", ""))
    except Exception:
        return default

def parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)

def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()

def parse_iso_ts(value: str) -> float:
    """Best-effort parse of ISO-8601 timestamp into epoch seconds."""
    if not value:
        return time.time()
    try:
        # Support strings ending with "Z" by replacing with explicit UTC offset.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return time.time()


# -------------------------
# Dict helpers
# -------------------------

def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely access nested dict using dot-path (e.g. "spec.resources.cpu")."""
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# Memory parsing
# -------------------------

def parse_mem_to_bytes(mem_str: str) -> Optional[int]:
    """Parse memory quantity strings into a byte count.

    Accepts forms like "1024", "256MB", "4GiB", "16 gb", case-insensitive.
    Returns None when parsing fails.
    """
    if not isinstance(mem_str, str):
        return None
    m = re.match(r"^\s*([0-9]+)\s*([KkMmGgTt][Ii]?[Bb]?)?\s*$", mem_str)
    if not m:
        return None
    qty = int(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("k", "kb", "kib"):
        return qty * 1024
    if unit in ("m", "mb", "mib"):
        return qty * 1024 ** 2
    if unit in ("g", "gb", "gib"):
        return qty * 1024 ** 3
    if unit in ("t", "tb", "tib"):
        return qty * 1024 ** 4
    if unit == "":
        return qty
    return None


# -------------------------
# Logging
# -------------------------

def get_logger(
    name: str = "orchestrator",
    log_file: str = "orchestrator.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """Return a configured logger with a rotating file handler and console output."""
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

    # File handler (rotating)
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

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger

def log_worker_event(logger, event: WorkerEvent) -> None:
    ev_type = event.type
    worker_id = event.worker_id
    if ev_type == "REGISTER":
        logger.info(
            "Worker registered: id=%s status=%s tags=%s",
            worker_id,
            event.status,
            event.tags,
        )
    elif ev_type == "UNREGISTER":
        logger.info("Worker unregistered: id=%s", worker_id)
    elif ev_type == "STATUS":
        logger.debug("Worker status: id=%s status=%s", worker_id, event.status)
    elif ev_type == "HEARTBEAT":
        logger.debug("Worker heartbeat: id=%s", worker_id)
        
# -------------------------
# Redis key helpers
# -------------------------

def r_worker_key(worker_id: str) -> str:
    """Redis hash key for a worker."""
    return f"worker:{worker_id}"


def r_hb_key(worker_id: str) -> str:
    """Redis key for a worker heartbeat value (with TTL)."""
    return f"worker:{worker_id}:hb"


# Constants
WORKERS_SET = "workers:ids"
