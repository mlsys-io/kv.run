# worker/redis_worker.py
"""Redis worker protocol helpers.

Implements the registration, heartbeat, status, and unregister operations
against Redis as described by the Orchestrator contract.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .event_schema import WorkerEvent, TaskEvent, serialize_event


class RedisWorker:
    WORKERS_SET = "workers:ids"

    def __init__(self, rds, worker_id: str):
        self.rds = rds
        self.worker_id = worker_id

    def _key(self) -> str:
        return f"worker:{self.worker_id}"

    def _hb_key(self) -> str:
        return f"worker:{self.worker_id}:hb"

    def register(
        self,
        status: str,
        started_at: str,
        pid: int,
        env: Dict[str, Any],
        hardware: Dict[str, Any],
        tags: List[str],
        *,
        cost_per_hour: float,
        power_metrics: Optional[Dict[str, Any]] = None,
    ):
        data = {
            "worker_id": self.worker_id,
            "status": status,
            "started_at": started_at,
            "pid": str(pid),
            "env_json": json.dumps(env, ensure_ascii=False),
            "hardware_json": json.dumps(hardware, ensure_ascii=False),
            "tags_json": json.dumps(tags, ensure_ascii=False),
            "last_seen": started_at,
            "cost_per_hour": f"{cost_per_hour}",
        }
        with self.rds.pipeline() as p:
            p.sadd(self.WORKERS_SET, self.worker_id)
            p.hset(self._key(), mapping=data)
            payload = {
                "env": env,
                "hardware": hardware,
                "cost_per_hour": cost_per_hour,
            }
            if power_metrics:
                payload["power_metrics"] = power_metrics
            evt = WorkerEvent(
                type="REGISTER",
                worker_id=self.worker_id,
                status=status,
                ts=started_at,
                tags=tags,
                payload=payload,
            )
            p.publish("workers.events", json.dumps(serialize_event(evt), ensure_ascii=False))
            p.execute()

    def heartbeat(self, ts: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None, ttl_sec: int = 120):
        from datetime import datetime, timezone
        ts = ts or datetime.now(timezone.utc).isoformat()
        with self.rds.pipeline() as p:
            p.setex(self._hb_key(), ttl_sec, ts)
            p.hset(self._key(), mapping={"last_seen": ts})
            evt = WorkerEvent(
                type="HEARTBEAT",
                worker_id=self.worker_id,
                ts=ts,
                metrics=metrics or {},
            )
            p.publish("workers.events", json.dumps(serialize_event(evt), ensure_ascii=False))
            p.execute()

    def set_status(self, status: str, extra: Optional[Dict[str, Any]] = None):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        mapping = {"status": status, "last_seen": now}
        if extra:
            mapping.update({f"extra_{k}": str(v) for k, v in extra.items()})
        with self.rds.pipeline() as p:
            p.hset(self._key(), mapping=mapping)
            evt = WorkerEvent(
                type="STATUS",
                worker_id=self.worker_id,
                status=status,
                ts=now,
                payload=extra or {},
            )
            p.publish("workers.events", json.dumps(serialize_event(evt), ensure_ascii=False))
            p.execute()

    def unregister(
        self,
        *,
        cost_per_hour: Optional[float] = None,
        uptime_sec: Optional[float] = None,
        accrued_cost_usd: Optional[float] = None,
        power_summary: Optional[Dict[str, Any]] = None,
    ):
        with self.rds.pipeline() as p:
            p.srem(self.WORKERS_SET, self.worker_id)
            p.delete(self._key())
            p.delete(self._hb_key())
            payload: Dict[str, Any] = {}
            if cost_per_hour is not None:
                payload["cost_per_hour"] = cost_per_hour
            if uptime_sec is not None:
                payload["uptime_sec"] = uptime_sec
            if accrued_cost_usd is not None:
                payload["accrued_cost_usd"] = accrued_cost_usd
            if power_summary is not None:
                payload["power_summary"] = power_summary
            evt = WorkerEvent(type="UNREGISTER", worker_id=self.worker_id, payload=payload)
            p.publish("workers.events", json.dumps(serialize_event(evt), ensure_ascii=False))
            p.execute()

    def task_failed(self, task_id: str, error: str, metadata: Optional[Dict[str, Any]] = None):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        evt = TaskEvent(
            type="TASK_FAILED",
            worker_id=self.worker_id,
            task_id=task_id,
            error=error,
            payload=metadata or {},
            ts=now,
        )
        self.rds.publish("tasks.events", json.dumps(serialize_event(evt), ensure_ascii=False))

    def task_succeeded(self, task_id: str, metadata: Optional[Dict[str, Any]] = None):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        evt = TaskEvent(
            type="TASK_SUCCEEDED",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=metadata or {},
            ts=now,
        )
        self.rds.publish("tasks.events", json.dumps(serialize_event(evt), ensure_ascii=False))

    def task_started(
        self,
        task_id: str,
        *,
        task_type: Optional[str] = None,
        dispatched_at: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {}
        if task_type is not None:
            payload["taskType"] = task_type
        if dispatched_at is not None:
            payload["dispatched_at"] = dispatched_at
        if started_at is not None:
            payload["started_at"] = started_at
        evt = TaskEvent(
            type="TASK_STARTED",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=payload,
            ts=now,
        )
        self.rds.publish("tasks.events", json.dumps(serialize_event(evt), ensure_ascii=False))
