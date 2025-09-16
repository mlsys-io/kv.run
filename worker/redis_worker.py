# worker/redis_worker.py
"""Redis worker protocol helpers.

Implements the registration, heartbeat, status, and unregister operations
against Redis as described by the Orchestrator contract.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class RedisWorker:
    WORKERS_SET = "workers:ids"

    def __init__(self, rds, worker_id: str):
        self.rds = rds
        self.worker_id = worker_id

    def _key(self) -> str:
        return f"worker:{self.worker_id}"

    def _hb_key(self) -> str:
        return f"worker:{self.worker_id}:hb"

    def register(self, status: str, started_at: str, pid: int,
                 env: Dict[str, Any], hardware: Dict[str, Any], tags: List[str]):
        data = {
            "worker_id": self.worker_id,
            "status": status,
            "started_at": started_at,
            "pid": str(pid),
            "env_json": json.dumps(env, ensure_ascii=False),
            "hardware_json": json.dumps(hardware, ensure_ascii=False),
            "tags_json": json.dumps(tags, ensure_ascii=False),
            "last_seen": started_at,
        }
        with self.rds.pipeline() as p:
            p.sadd(self.WORKERS_SET, self.worker_id)
            p.hset(self._key(), mapping=data)
            evt = {
                "type": "REGISTER",
                "worker_id": self.worker_id,
                "status": status,
                "ts": started_at,
                "tags": tags,
            }
            p.publish("workers.events", json.dumps(evt, ensure_ascii=False))
            p.execute()

    def heartbeat(self, ts: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None, ttl_sec: int = 120):
        from datetime import datetime, timezone
        ts = ts or datetime.now(timezone.utc).isoformat()
        with self.rds.pipeline() as p:
            p.setex(self._hb_key(), ttl_sec, ts)
            p.hset(self._key(), mapping={"last_seen": ts})
            evt = {
                "type": "HEARTBEAT",
                "worker_id": self.worker_id,
                "ts": ts,
                "metrics": metrics or {},
            }
            p.publish("workers.events", json.dumps(evt, ensure_ascii=False))
            p.execute()

    def set_status(self, status: str, extra: Optional[Dict[str, Any]] = None):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        mapping = {"status": status, "last_seen": now}
        if extra:
            mapping.update({f"extra_{k}": str(v) for k, v in extra.items()})
        with self.rds.pipeline() as p:
            p.hset(self._key(), mapping=mapping)
            evt = {
                "type": "STATUS",
                "worker_id": self.worker_id,
                "status": status,
                "ts": now,
                "extra": extra or {},
            }
            p.publish("workers.events", json.dumps(evt, ensure_ascii=False))
            p.execute()

    def unregister(self):
        with self.rds.pipeline() as p:
            p.srem(self.WORKERS_SET, self.worker_id)
            p.delete(self._key())
            p.delete(self._hb_key())
            evt = {"type": "UNREGISTER", "worker_id": self.worker_id}
            p.publish("workers.events", json.dumps(evt, ensure_ascii=False))
            p.execute()
            
    def task_failed(self, task_id: str, error: str):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        evt = {
            "type": "TASK_FAILED",
            "worker_id": self.worker_id,
            "task_id": task_id,
            "error": error,
            "ts": now,
        }
        self.rds.publish("tasks.events", json.dumps(evt, ensure_ascii=False))

    def task_succeeded(self, task_id: str, result: Optional[Dict[str, Any]] = None):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        evt = {
            "type": "TASK_SUCCEEDED",
            "worker_id": self.worker_id,
            "task_id": task_id,
            "result": result or {},
            "ts": now,
        }
        self.rds.publish("tasks.events", json.dumps(evt, ensure_ascii=False))