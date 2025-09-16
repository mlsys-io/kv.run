# utils.py
"""Utility helpers for the Orchestrator service.

This module centralizes small reusable helpers such as environment parsing,
ISO timestamp generation, safe dictionary access, memory string parsing,
logger configuration, and Redis key helpers.
"""
from __future__ import annotations

import os
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, Optional


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


def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


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
        # Already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler (rotating)
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger


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

