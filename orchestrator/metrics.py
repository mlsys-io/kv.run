from __future__ import annotations

"""简单的本地指标记录器，用于根据 Redis 事件生成度量。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Set

from event_schema import TaskEvent, WorkerEvent, Event, serialize_event


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricsRecorder:
    def __init__(self, base_dir: Path, logger) -> None:
        self._dir = Path(base_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._dir / "metrics.json"
        self._events_log = self._dir / "events.log"
        self._logger = logger
        self._lock = RLock()
        self._active_workers: Set[str] = set()
        self._counters: Dict[str, int] = {
            "tasks_submitted": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_requeued": 0,
            "workers_registered": 0,
            "workers_unregistered": 0,
            "worker_heartbeats": 0,
        }
        self._write_metrics()

    # -------------------------
    # Recording
    # -------------------------
    def record_event(self, event: Event) -> None:
        if isinstance(event, TaskEvent):
            self.record_task_event(event)
        else:
            self.record_worker_event(event)

    def record_task_event(self, event: TaskEvent) -> None:
        with self._lock:
            ev_type = event.type
            if ev_type == "TASK_SUCCEEDED":
                self._counters["tasks_succeeded"] += 1
            elif ev_type == "TASK_FAILED":
                self._counters["tasks_failed"] += 1
            elif ev_type == "TASK_REQUEUED":
                self._counters["tasks_requeued"] += 1
            elif ev_type == "TASK_SUBMITTED":
                self._counters["tasks_submitted"] += 1
            self._append_event(event)
            self._write_metrics()

    def record_worker_event(self, event: WorkerEvent) -> None:
        with self._lock:
            ev_type = event.type
            worker_id = (event.worker_id or "").strip()
            if ev_type == "REGISTER" and worker_id:
                self._active_workers.add(worker_id)
                self._counters["workers_registered"] += 1
            elif ev_type == "UNREGISTER" and worker_id:
                self._active_workers.discard(worker_id)
                self._counters["workers_unregistered"] += 1
            elif ev_type == "HEARTBEAT":
                self._counters["worker_heartbeats"] += 1
            self._append_event(event)
            self._write_metrics()

    # -------------------------
    # Utilities
    # -------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_snapshot()

    def _write_metrics(self) -> None:
        payload = self._build_snapshot()
        try:
            self._metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._logger.warning("Failed to write metrics snapshot: %s", exc)

    def _append_event(self, event: Event) -> None:
        try:
            with self._events_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(serialize_event(event), ensure_ascii=False))
                fh.write("\n")
        except Exception as exc:
            self._logger.debug("Failed to append event log: %s", exc)

    def _build_snapshot(self) -> Dict[str, Any]:
        return {
            "last_updated": _now_iso(),
            "counters": dict(self._counters),
            "active_workers": sorted(self._active_workers),
            "active_workers_count": len(self._active_workers),
        }
