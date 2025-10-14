# worker/config.py
"""Configuration loader for the Worker process.

This module encapsulates all environment-derived configuration so the rest of
the worker code can depend on a structured config object.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


DEFAULT_TOPIC = "tasks"


@dataclass(frozen=True)
class WorkerConfig:
    redis_url: str
    topic: str
    results_dir: Path
    hb_interval_sec: int
    hb_ttl_sec: int
    worker_id: str
    tags: List[str]
    log_level: str
    cost_per_hour: float

    @staticmethod
    def from_env() -> "WorkerConfig":
        redis_url = os.getenv("REDIS_URL")

        topic = os.getenv("TASK_TOPIC", DEFAULT_TOPIC)

        results_dir = Path(os.getenv("RESULTS_DIR", "./results_workers")).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)

        hb_interval = int(os.getenv("HEARTBEAT_INTERVAL_SEC", "30"))
        hb_ttl = max(hb_interval * 4, 120)

        worker_id = os.getenv("WORKER_ID", "").strip() or os.urandom(8).hex()
        tags = [t.strip() for t in os.getenv("WORKER_TAGS", "").split(',') if t.strip()]

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        cost_raw = (os.getenv("WORKER_COST_PER_HOUR") or "").strip() or "1.0"
        try:
            cost_per_hour = float(cost_raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid WORKER_COST_PER_HOUR value: {cost_raw}") from exc
        if cost_per_hour < 0:
            raise SystemExit("WORKER_COST_PER_HOUR must be non-negative")

        return WorkerConfig(
            redis_url=redis_url,
            topic=topic,
            results_dir=results_dir,
            hb_interval_sec=hb_interval,
            hb_ttl_sec=hb_ttl,
            worker_id=worker_id,
            tags=tags,
            log_level=log_level,
            cost_per_hour=cost_per_hour,
        )
