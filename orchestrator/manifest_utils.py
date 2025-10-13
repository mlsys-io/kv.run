from __future__ import annotations

"""Manifest helpers scoped to orchestrator output management."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MANIFEST_NAME = "manifest.json"
RESPONSES_NAME = "responses.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"


def prepare_output_dir(base_dir: Path) -> None:
    """Ensure the base directory and standard sub-directories exist."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / LOGS_DIR).mkdir(parents=True, exist_ok=True)
    (base_dir / ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)


def sync_manifest(base_dir: Path, task_id: str, expected: Iterable[str]) -> Dict[str, Any]:
    """
    Build a manifest by reconciling expected versus actual files.

    expected comes from spec.output.artifacts and is interpreted relative to base_dir.
    """
    prepare_output_dir(base_dir)
    expected_set = {_normalize_artifact_name(item) for item in expected or [] if item}
    expected_set.update({RESPONSES_NAME, LOGS_DIR, ARTIFACTS_DIR})

    entries: List[Dict[str, Any]] = []
    added: set[str] = set()

    for name in sorted(expected_set):
        rel_path = Path(name)
        entry = _describe_path(base_dir, rel_path, required=True)
        entries.append(entry)
        added.add(_path_key(rel_path))

    # Capture additional files/directories that exist but were not declared.
    for item in base_dir.iterdir():
        key = _path_key(item.relative_to(base_dir))
        if key in added or item.name == MANIFEST_NAME:
            continue
        entry = _describe_path(base_dir, item.relative_to(base_dir), required=False)
        entries.append(entry)

    manifest = {
        "task_id": task_id,
        "generated_at": _now_iso(),
        "entries": entries,
    }
    (base_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# -------------------------
# Helpers
# -------------------------

def _path_key(path: Path) -> str:
    return str(path.as_posix())


def _infer_type(rel_path: Path) -> str:
    normalized = rel_path.as_posix()
    if normalized == RESPONSES_NAME:
        return "result"
    if normalized.startswith(f"{LOGS_DIR}/") or normalized == LOGS_DIR:
        return "logs"
    if normalized.startswith(f"{ARTIFACTS_DIR}/") or normalized == ARTIFACTS_DIR:
        return "artifact"
    if rel_path.suffix:
        return "artifact"
    return "directory"


def _describe_path(base_dir: Path, rel_path: Path, *, required: bool) -> Dict[str, Any]:
    target = base_dir / rel_path
    entry_type = _infer_type(rel_path)
    entry: Dict[str, Any] = {
        "name": rel_path.as_posix(),
        "path": rel_path.as_posix(),
        "type": entry_type,
        "required": required,
    }

    if target.exists():
        entry["status"] = "present"
        entry["updated_at"] = _now_iso()
        if target.is_file():
            stat = target.stat()
            entry["size"] = stat.st_size
            entry["sha256"] = _sha256_file(target)
        else:
            size, count = _directory_stats(target)
            entry["size"] = size
            entry["file_count"] = count
    else:
        entry["status"] = "missing"
    return entry


def _normalize_artifact_name(name: str) -> str:
    value = name.strip()
    if value.endswith("/"):
        value = value.rstrip("/")
    if value.startswith("./"):
        value = value[2:]
    return value or name


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _directory_stats(path: Path) -> Tuple[int, int]:
    total_size = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            total_size += stat.st_size
            file_count += 1
    return total_size, file_count


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
