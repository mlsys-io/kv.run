from __future__ import annotations

import json
from pathlib import Path

from orchestrator.manifest_utils import prepare_output_dir, sync_manifest
from orchestrator.results import write_result


def test_http_artifact_manifest(tmp_path: Path):
    task_id = "http-123"
    base_dir = tmp_path / task_id
    prepare_output_dir(base_dir)

    write_result(tmp_path, task_id, {"task_id": task_id, "result": {"ok": True}})

    expected = ["responses.json", "artifacts/model.bin"]
    manifest = sync_manifest(base_dir, task_id, expected)
    statuses = {entry["name"]: entry["status"] for entry in manifest["entries"]}
    assert statuses["responses.json"] == "present"
    assert statuses["artifacts/model.bin"] == "missing"

    artifact_path = base_dir / "artifacts" / "model.bin"
    artifact_path.write_text("checkpoint", encoding="utf-8")

    manifest = sync_manifest(base_dir, task_id, expected)
    statuses = {entry["name"]: entry["status"] for entry in manifest["entries"]}
    assert statuses["artifacts/model.bin"] == "present"
