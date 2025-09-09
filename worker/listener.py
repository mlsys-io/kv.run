#!/usr/bin/env python3
"""
Worker Process

- Subscribes to Redis Pub/Sub topics
- Executes task payloads via a pluggable executor
- Writes results to RESULTS_DIR/<task_id>/responses.json

ENV:
  REDIS_URL              e.g. redis://localhost:6379/0 (required)
  TASK_TOPICS            default "tasks.inference"
  RESULTS_DIR            default ./results
  HEARTBEAT_INTERVAL_SEC default 30
  WORKER_ID              optional fixed id
  WORKER_TAGS            optional tags (comma-separated)
  LOG_LEVEL              default INFO
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
import socket
import platform
import subprocess
import sys
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

# executors
from worker.executors import VLLMExecutor, PPOExecutor

# -------------------------
# Helpers
# -------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")


logger = logging.getLogger("worker")


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def collect_hw() -> Dict[str, Any]:
    # CPU
    cpu = {"logical_cores": os.cpu_count() or 0, "model": platform.processor() or platform.machine()}
    # Memory
    mem = {"total_bytes": None}
    if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem["total_bytes"] = int(line.split()[1]) * 1024
                    break
    # GPU
    full = _run(["nvidia-smi"])  # presence indicates NVIDIA stack
    cuda = None
    drv = None
    m = re.search(r"CUDA Version:\s*([\w\.\-]+)", full)
    cuda = m.group(1) if m else None
    m = re.search(r"Driver Version:\s*([\w\.\-]+)", full)
    drv = m.group(1) if m else None
    lst = _run(["nvidia-smi", "-L"]) if full else ""
    gpus: List[Dict[str, Any]] = []
    for line in lst.splitlines():
        m = re.match(r"GPU\s+(\d+):\s+(.+?)\s+\(UUID:\s*([^\)]+)\)", line.strip())
        if m:
            gpus.append({"index": int(m.group(1)), "name": m.group(2), "uuid": m.group(3)})
    gpu = {"driver_version": drv, "cuda_version": cuda, "gpus": gpus}

    # Network
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = None

    return {"cpu": cpu, "memory": mem, "gpu": gpu, "network": {"ip": ip}}


# -------------------------
# Redis worker protocol helpers
# -------------------------

class RedisWorker:
    WORKERS_SET = "workers:ids"

    def __init__(self, rds: "redis.Redis", worker_id: str):
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
        logger.info("[orchestrator] registered worker=%s status=%s", self.worker_id, status)

    def heartbeat(self, ts: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None, ttl_sec: int = 120):
        ts = ts or _now_iso()
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
        logger.debug("[orchestrator] heartbeat sent worker=%s ts=%s", self.worker_id, ts)

    def set_status(self, status: str, extra: Optional[Dict[str, Any]] = None):
        now = _now_iso()
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
        logger.info("[orchestrator] status updated worker=%s status=%s extra=%s", self.worker_id, status, extra)

    def unregister(self):
        with self.rds.pipeline() as p:
            p.srem(self.WORKERS_SET, self.worker_id)
            p.delete(f"worker:{self.worker_id}")
            p.delete(f"worker:{self.worker_id}:hb")
            evt = {
                "type": "UNREGISTER",
                "worker_id": self.worker_id,
                "ts": _now_iso(),
            }
            p.publish("workers.events", json.dumps(evt, ensure_ascii=False))
            p.execute()
        logger.info("[orchestrator] unregistered worker=%s", self.worker_id)

# -------------------------
# Lifecycle + Runner
# -------------------------

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
            started_at=_now_iso(),
            pid=os.getpid(),
            env=env,
            hardware=hardware,
            tags=tags,
        )
        self.rworker.set_status("IDLE")
        threading.Thread(target=self._hb_loop, daemon=True).start()
        logger.info("worker %s ready", self.rworker.worker_id)

    def _hb_loop(self):
        while not self.stop.is_set():
            try:
                self.rworker.heartbeat(ts=_now_iso(), metrics=self._metrics(), ttl_sec=self.hb_ttl_sec)
            except Exception as e:
                logger.warning("heartbeat failed: %s", e)
            self.stop.wait(self.hb_sec)

    def set_running(self, task_id: str):
        try:
            self.rworker.set_status("RUNNING", {"task_id": task_id})
        except Exception as e:
            logger.warning("status RUNNING failed: %s", e)

    def set_idle(self, task_id: str):
        try:
            self.rworker.set_status("IDLE", {"last_task": task_id})
        except Exception as e:
            logger.warning("status IDLE failed: %s", e)

    def shutdown(self):
        self.stop.set()
        try:
            # 主动注销 worker，避免 Redis 残留僵尸
            self.rworker.unregister()
        except Exception as e:
            logger.warning("unregister failed: %s", e)


class Runner:
    def __init__(self, lifecycle: Lifecycle, rds: "redis.Redis", topics: List[str], results_dir: Path):
        self.lifecycle = lifecycle
        self.redis = rds
        self.topics = topics
        self.results_dir = results_dir

        # Initialize available executors
        self.executors = {
            "inference": VLLMExecutor(),
            "ppo": PPOExecutor(),
        }
        
        # Default executor for backward compatibility
        self.executor = self.executors.get("inference")

    def start(self):
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(*self.topics)
        logger.info("subscribed to topics: %s", self.topics)
        for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue

            # 打印所有收到的订阅消息（原始内容）
            logger.info("pubsub msg channel=%s data=%s", msg.get("channel"), msg.get("data"))

            data_raw = msg.get("data")
            if isinstance(data_raw, bytes):
                data_raw = data_raw.decode("utf-8", errors="ignore")
            try:
                data = json.loads(data_raw)
            except Exception as e:
                logger.warning("invalid task message: %s raw=%s", e, data_raw)
                continue

            if data.get("assigned_worker") != self.lifecycle.rworker.worker_id:
                logger.debug("skip task_id=%s assigned=%s my_id=%s",
                             data.get("task_id"),
                             data.get("assigned_worker"),
                             self.lifecycle.rworker.worker_id)
                continue

            task_id = str(data.get("task_id"))
            task = data.get("task") or {}
            task_type = (task.get("spec") or {}).get("taskType")

            logger.info("accepted task_id=%s type=%s", task_id, task_type)
            self.lifecycle.set_running(task_id)
            try:
                # Select appropriate executor based on task type
                executor = self.executors.get(task_type, self.executor)
                logger.info("Selected executor: %s for task_type: %s", executor.__class__.__name__ if executor else "None", task_type)
                
                if executor:
                    logger.info("Starting task execution for task_id=%s", task_id)
                    out_dir = self.results_dir / task_id
                    out = executor.run(task, out_dir)
                    logger.info("task %s finished successfully", task_id)
                else:
                    logger.warning("no executor configured for taskType=%s", task_type)
            except Exception as e:
                logger.exception("Task %s failed with error: %s", task_id, e)
            finally:
                self.lifecycle.set_idle(task_id)


# -------------------------
# Main
# -------------------------

def main():
    _setup_logging()

    REDIS_URL = os.getenv("REDIS_URL")
    if not REDIS_URL:
        logger.error("REDIS_URL is required")
        raise SystemExit(1)
    rds = redis.from_url(REDIS_URL, decode_responses=True)
    rds.ping()

    worker_id = os.getenv("WORKER_ID") or str(uuid.uuid4())
    tags = [t.strip() for t in (os.getenv("WORKER_TAGS") or "").split(",") if t.strip()]

    hb_interval = int(os.getenv("HEARTBEAT_INTERVAL_SEC", "30"))
    hb_ttl = max(hb_interval * 4, 120)

    rworker = RedisWorker(rds, worker_id)
    lifecycle = Lifecycle(rworker, hb_interval, hb_ttl)
    lifecycle.start(env={}, hardware=collect_hw(), tags=tags)

    task_topics_env = os.getenv("TASK_TOPICS")
    default_topics = "tasks.inference,tasks.ppo"
    topics_str = task_topics_env or default_topics
    topics = [t.strip() for t in topics_str.split(",") if t.strip()]
    
    logger.info("TASK_TOPICS environment variable: %s", task_topics_env)
    logger.info("Using topics: %s", topics)
    results_dir = Path(os.getenv("RESULTS_DIR", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        Runner(lifecycle, rds, topics, results_dir).start()
    except KeyboardInterrupt:
        pass
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()
