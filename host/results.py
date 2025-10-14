from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .utils import now_iso
from .manifest_utils import prepare_output_dir


class ResultPayload(BaseModel):
    task_id: str
    result: Dict[str, Any]
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    received_at: str = Field(default_factory=now_iso)


def _sanitize_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)


def result_file_path(base_dir: Path, task_id: str) -> Path:
    return base_dir / _sanitize_task_id(task_id) / "responses.json"


def write_result(base_dir: Path, task_id: str, content: Dict[str, Any]) -> Path:
    path = result_file_path(base_dir, task_id)
    prepare_output_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    return path


def read_result(base_dir: Path, task_id: str) -> str:
    path = result_file_path(base_dir, task_id)
    return path.read_text(encoding="utf-8")
