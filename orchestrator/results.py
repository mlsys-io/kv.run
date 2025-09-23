from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _sanitize_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)


def result_file_path(base_dir: Path, task_id: str) -> Path:
    return base_dir / _sanitize_task_id(task_id) / "responses.json"


def write_result(base_dir: Path, task_id: str, content: Dict[str, Any]) -> Path:
    path = result_file_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    return path


def read_result(base_dir: Path, task_id: str) -> str:
    path = result_file_path(base_dir, task_id)
    return path.read_text(encoding="utf-8")
