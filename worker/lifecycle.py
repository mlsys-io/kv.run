# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redis_worker import RedisWorker


class Lifecycle:
    def __init__(self, rworker: RedisWorker, hb_sec: int, hb_ttl_sec: int):
        self.rworker = rworker
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.stop = threading.Event()

    def _metrics(self) -> Dict[str, Any]:
        try:
            la = os.getloadavg()
            return {"loadavg": {"1m": la[0], "5m": la[1], "15m": la[2]}}
        except Exception:
            return {}

    def start(self, env: Dict[str, Any], hardware: Dict[str, Any], tags: List[str]):
        self.rworker.register(
            status="STARTING",
            started_at=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
            env=env,
            hardware=hardware,
            tags=tags,
        )
        self.rworker.set_status("IDLE")
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _hb_loop(self):
        while not self.stop.is_set():
            try:
                self.rworker.heartbeat(ttl_sec=self.hb_ttl_sec, metrics=self._metrics())
            except Exception:
                pass
            self.stop.wait(self.hb_sec)

    def set_running(self, task_id: str):
        try:
            self.rworker.set_status("RUNNING", {"task_id": task_id})
        except Exception:
            pass

    def set_idle(self, task_id: str):
        try:
            self.rworker.set_status("IDLE", {"last_task": task_id})
        except Exception:
            pass
        
    def set_failed(self, task_id: str, error: Optional[str] = None):
        try:
            self.rworker.task_failed(task_id, error=error)
        except Exception:
            pass
    
    def set_succeeded(self, task_id: str):
        try:
            self.rworker.task_succeeded(task_id)
        except Exception:
            pass
    def shutdown(self):
        self.stop.set()
        try:
            self.rworker.unregister()
        except Exception:
            pass