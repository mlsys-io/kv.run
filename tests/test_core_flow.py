from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORCH_DIR = ROOT / "orchestrator"
if str(ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(ORCH_DIR))

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str | None = None):
            super().__init__(detail or "HTTPException")
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.HTTPException = _HTTPException
    sys.modules["fastapi"] = fastapi_stub

import utils  # noqa: F401  确保顶层模块可被后续相对导入使用
from event_schema import TaskEvent, WorkerEvent, parse_event, serialize_event
from manifest_utils import prepare_output_dir, sync_manifest
from orchestrator.aggregation import maybe_aggregate_parent
from orchestrator.metrics import MetricsRecorder
from orchestrator.results import write_result, read_result
from orchestrator.task_store import TaskPool, PoolEntry
from task import TaskRecord, TaskStatus


def _sample_yaml() -> str:
    return """
apiVersion: mloc/v1
kind: InferenceTask
metadata:
  name: demo
spec:
  taskType: inference
  resources:
    replicas: 1
    hardware:
      cpu: "2"
      memory: "4Gi"
  output:
    destination:
      type: local
"""


def test_task_pool_dependency_resolution():
    pool = TaskPool(batch_size=2, slo_fraction=0.5)
    record_a = TaskRecord(task_id="a", raw_yaml="", parsed={}, status=TaskStatus.PENDING)
    record_b = TaskRecord(task_id="b", raw_yaml="", parsed={}, status=TaskStatus.PENDING)

    assert pool.add(PoolEntry(task_id="a", task={}, record=record_a)) == []
    batch = pool.add(PoolEntry(task_id="b", task={}, record=record_b))
    assert {entry.task_id for entry in batch} == {"a", "b"}


def test_manifest_sync_and_upload(tmp_path: Path):
    task_id = "task-demo"
    base_dir = tmp_path / task_id
    prepare_output_dir(base_dir)

    content = {"task_id": task_id, "result": {"items": [1, 2, 3]}}
    write_result(tmp_path, task_id, content)

    manifest = sync_manifest(base_dir, task_id, ["artifacts/model.pt", "logs"])
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert entries["responses.json"]["status"] == "present"
    assert entries["artifacts/model.pt"]["status"] == "missing"


def test_aggregation_merges(tmp_path: Path):
    parent_id = "parent"
    child_id = "child"
    tasks = {
        parent_id: TaskRecord(task_id=parent_id, raw_yaml="", parsed={"spec": {"taskType": "inference"}}),
    }
    child_payload = {
        "task_id": child_id,
        "result": {
            "items": [{"output": "hello"}],
            "usage": {"tokens": 10},
        },
    }
    child_to_parent = {child_id: parent_id}
    parent_shards = {
        parent_id: {
            "total": 1,
            "done": 1,
            "children": {child_id},
            "order": {},
            "results": {child_id: child_payload},
        }
    }

    maybe_aggregate_parent(
        child_id,
        child_payload,
        child_to_parent=child_to_parent,
        parent_shards=parent_shards,
        tasks=tasks,
        tasks_lock=threading.RLock(),
        results_dir=tmp_path,
        logger=logging.getLogger("test"),
    )

    aggregated = json.loads(read_result(tmp_path, parent_id))
    assert aggregated["result"]["merged"]["items"]


def test_event_schema_and_metrics(tmp_path: Path):
    metrics = MetricsRecorder(tmp_path, logging.getLogger("metrics"))
    task_event = TaskEvent(type="TASK_SUCCEEDED", task_id="abc", worker_id="worker-1")
    worker_event = WorkerEvent(type="REGISTER", worker_id="worker-1", status="IDLE")

    metrics.record_task_event(task_event)
    metrics.record_worker_event(worker_event)

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["tasks_succeeded"] == 1
    assert snapshot["active_workers_count"] == 1

    raw = serialize_event(task_event)
    parsed = parse_event(raw)
    assert isinstance(parsed, TaskEvent)
    assert parsed.task_id == "abc"
