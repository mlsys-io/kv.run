from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from orchestrator.aggregation import maybe_aggregate_parent
from orchestrator.task import TaskRecord, TaskStatus
from manifest_utils import sync_manifest


def _make_child_payload(text: str) -> dict[str, object]:
    return {
        "result": {
            "items": [text],
            "usage": {"tokens": 4},
        }
    }


def test_data_parallel_merges_and_manifests(tmp_path: Path):
    parent_id = "parent-task"
    child_ids = ["child-a", "child-b"]

    tasks = {
        parent_id: TaskRecord(
            task_id=parent_id,
            raw_yaml="",
            parsed={
                "spec": {
                    "taskType": "inference",
                    "output": {"destination": {"type": "local", "path": "./"}},
                }
            },
            status=TaskStatus.DISPATCHED,
        )
    }

    parent_shards = {
        parent_id: {
            "total": 2,
            "done": 0,
            "failed": 0,
            "children": set(child_ids),
            "order": {child_ids[0]: 0, child_ids[1]: 1},
            "results": {},
        }
    }
    child_to_parent = {cid: parent_id for cid in child_ids}

    tasks_lock = threading.RLock()

    logger = logging.getLogger("test.parallel")

    for idx, cid in enumerate(child_ids):
        payload = _make_child_payload(f"response-{idx}")
        payload["task_id"] = cid
        maybe_aggregate_parent(
            cid,
            payload,
            child_to_parent=child_to_parent,
            parent_shards=parent_shards,
            tasks=tasks,
            tasks_lock=tasks_lock,
            results_dir=tmp_path,
            logger=logger,
        )

    parent_path = tmp_path / parent_id / "responses.json"
    assert parent_path.exists()
    merged = json.loads(parent_path.read_text())
    assert merged["result"]["merged"]["items"] == ["response-0", "response-1"]

    manifest = sync_manifest(parent_path.parent, parent_id, ["responses.json", "logs", "artifacts"])
    statuses = {entry["name"]: entry["status"] for entry in manifest["entries"]}
    assert statuses["responses.json"] == "present"
    assert statuses["logs"] == "present"
