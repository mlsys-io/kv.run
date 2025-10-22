# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .power import PowerMonitor
from .redis_worker import RedisWorker


class Lifecycle:
    def __init__(
        self,
        rworker: RedisWorker,
        hb_sec: int,
        hb_ttl_sec: int,
        *,
        cost_per_hour: float,
        power_monitor: Optional[PowerMonitor] = None,
    ):
        self.rworker = rworker
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.cost_per_hour = cost_per_hour
        self.power_monitor = power_monitor or PowerMonitor()
        self.stop = threading.Event()
        self._started_ts: Optional[float] = None

    def _metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
            metrics["uptime_sec"] = uptime
            metrics["accrued_cost_usd"] = (self.cost_per_hour / 3600.0) * uptime
        try:
            la = os.getloadavg()
            metrics["loadavg"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
        except Exception:
            pass
        try:
            power_sample = self.power_monitor.sample()
        except Exception:
            power_sample = None
        if power_sample:
            metrics["power"] = power_sample
        try:
            power_summary = self.power_monitor.summary()
        except Exception:
            power_summary = None
        if power_summary:
            metrics["power_summary"] = power_summary
            energy_total = power_summary.get("estimated_energy_kwh")
            if isinstance(energy_total, (int, float)):
                metrics["estimated_energy_kwh"] = energy_total
        return metrics

    def start(self, env: Dict[str, Any], hardware: Dict[str, Any], tags: List[str]):
        self._started_ts = time.time()
        try:
            initial_power = self.power_monitor.sample()
        except Exception:
            initial_power = None
        self.rworker.register(
            status="STARTING",
            started_at=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
            env=env,
            hardware=hardware,
            tags=tags,
            cost_per_hour=self.cost_per_hour,
            power_metrics=initial_power,
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
        
    def set_failed(self, task_id: str, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        try:
            self.rworker.task_failed(task_id, error=error, metadata=metadata)
        except Exception:
            pass
    
    def set_succeeded(self, task_id: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            self.rworker.task_succeeded(task_id, metadata=metadata)
        except Exception:
            pass

    def notify_task_started(
        self,
        task_id: str,
        task_type: Optional[str],
        dispatched_at: Optional[str],
        started_at: str,
    ) -> None:
        try:
            self.rworker.task_started(
                task_id,
                task_type=task_type,
                dispatched_at=dispatched_at,
                started_at=started_at,
            )
        except Exception:
            pass

    def shutdown(self):
        self.stop.set()
        try:
            self.power_monitor.sample()
        except Exception:
            pass
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
        accrued_cost = (self.cost_per_hour / 3600.0) * uptime if uptime is not None else None
        summary = self.power_monitor.summary()
        try:
            self.rworker.unregister(
                cost_per_hour=self.cost_per_hour,
                uptime_sec=uptime,
                accrued_cost_usd=accrued_cost,
                power_summary=summary,
            )
        except Exception:
            pass
