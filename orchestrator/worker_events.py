"""Helpers for logging worker events."""

from __future__ import annotations

from typing import Any, Dict


def log_worker_event(logger, event: Dict[str, Any]) -> None:
    ev_type = str(event.get("type", "")).upper()
    worker_id = event.get("worker_id")
    if ev_type == "REGISTER":
        logger.info(
            "Worker registered: id=%s status=%s tags=%s",
            worker_id,
            event.get("status"),
            event.get("tags"),
        )
    elif ev_type == "UNREGISTER":
        logger.info("Worker unregistered: id=%s", worker_id)
    elif ev_type == "STATUS":
        logger.debug("Worker status: id=%s status=%s", worker_id, event.get("status"))
    elif ev_type == "HEARTBEAT":
        logger.debug("Worker heartbeat: id=%s", worker_id)
