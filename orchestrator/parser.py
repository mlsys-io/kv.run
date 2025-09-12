# parser.py
"""Task parsing and in-memory dependency tracking (supports stages).

This module provides:
- YAML parsing and validation for tasks.
- An in-memory TaskStore that records tasks, auto-generates stage tasks,
  and tracks dependencies.
- For staged tasks (spec.stages), it generates one task per stage, assigns
  task IDs, auto-chains dependencies (stage i depends on stage i-1), and
  stores all tasks atomically in the store.

Schema expectations (minimal and backward compatible):
- REQUIRED_FIELDS must exist for the base task.
- Optional single-task dependencies via: spec.dependsOn: [<task_id>, ...]
- Optional staged pipeline via:
    spec:
      stages:
        - name: "prepare"
          spec: {...}        # optional overrides merged into base spec
        - name: "train"
          spec: {...}
        - name: "infer"
          spec: {...}

Notes:
- This store is process-local (in-memory). Persist to Redis/DB if durability
  is required.
- Dependency satisfaction is evaluated externally by the orchestrator via
  a callback (e.g., check predecessor status == DONE).
"""
from __future__ import annotations

import copy
import threading
import uuid
from typing import Any, Dict, List, Set, Callable

from fastapi import HTTPException
import yaml

from utils import safe_get


REQUIRED_FIELDS = [
    "apiVersion",
    "kind",
    "metadata.name",
    "spec.taskType",
    "spec.resources.replicas",
    "spec.resources.hardware.cpu",
    "spec.resources.hardware.memory",
]


def _validate_yaml_to_dict(yaml_text: str) -> Dict[str, Any]:
    """Parse YAML and validate required fields. Raises HTTP 400 on error."""
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # Validate required fields on the base document
    for path in REQUIRED_FIELDS:
        cur = data
        for key in path.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                raise HTTPException(status_code=400, detail=f"Missing required field: {path}")
            cur = cur[key]

    # Optional GPU block validation
    gpu = safe_get(data, "spec.resources.hardware.gpu", {})
    if gpu:
        if safe_get(gpu, "count") is None:
            raise HTTPException(status_code=400, detail="Missing spec.resources.hardware.gpu.count")
        if safe_get(gpu, "type") is None:
            raise HTTPException(status_code=400, detail="Missing spec.resources.hardware.gpu.type")

    return data


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (src overrides dst). Returns a new dict."""
    out = copy.deepcopy(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class TaskStore:
    """In-memory registry for parsed tasks and dependency relations.

    Data structures:
      - _parsed:   task_id -> parsed YAML dict
      - _depends:  task_id -> set of predecessor task_ids
      - _released: set of task_ids already handed to the dispatcher (avoid duplicates)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._parsed: Dict[str, Dict[str, Any]] = {}
        self._depends: Dict[str, Set[str]] = {}
        self._released: Set[str] = set()

    # -------------------------
    # Registration / Parsing
    # -------------------------
    def parse_and_register(self, yaml_text: str) -> List[Dict[str, Any]]:
        """Parse YAML and register one or more tasks in the store.

        Behavior:
          - If spec.stages is present and is a non-empty list:
                * Generate a task per stage with its own task_id
                * Merge base spec with stage.spec overrides
                * Auto-chain dependencies: stage[i] depends on stage[i-1]
                * metadata.name is suffixed with ':<stage_name_or_index>'
          - Else (no stages):
                * Register a single task with optional spec.dependsOn

        Returns:
            List[{"task_id": str, "parsed": dict, "depends_on": List[str]}]
        """
        base = _validate_yaml_to_dict(yaml_text)
        stages = safe_get(base, "spec.stages", None)

        # Base copy for children to inherit from
        base_clean = copy.deepcopy(base)
        if isinstance(safe_get(base_clean, "spec", {}), dict):
            base_clean["spec"].pop("stages", None)  # remove stages from child tasks

        results: List[Dict[str, Any]] = []

        if isinstance(stages, list) and len(stages) > 0:
            # Staged pipeline
            prev_task_id: str | None = None
            base_name = str(safe_get(base, "metadata.name", "task"))

            for idx, stage in enumerate(stages):
                stage_name = str(stage.get("name") or f"stage-{idx+1}")
                stage_overrides = stage.get("spec") or {}

                # Effective parsed task = deep-merge(base_clean, {"spec": stage_overrides})
                effective = _deep_merge(base_clean, {"spec": stage_overrides})

                # Give each stage task a unique and descriptive name
                eff = copy.deepcopy(effective)
                eff_md = eff.setdefault("metadata", {})
                eff_md["name"] = f"{base_name}:{stage_name}"

                # Generate task_id and dependencies (chain to previous stage)
                tid = str(uuid.uuid4())
                depends_on: List[str] = []
                if prev_task_id:
                    depends_on.append(prev_task_id)

                with self._lock:
                    self._parsed[tid] = eff
                    self._depends[tid] = set(depends_on)

                results.append({"task_id": tid, "parsed": eff, "depends_on": depends_on})
                prev_task_id = tid

            return results

        # Single-task path (no stages)
        depends_on_raw = safe_get(base, "spec.dependsOn", []) or []
        if not isinstance(depends_on_raw, list):
            raise HTTPException(status_code=400, detail="spec.dependsOn must be a list of task_ids")
        depends_on = [str(x).strip() for x in depends_on_raw if str(x).strip()]

        tid = str(uuid.uuid4())
        with self._lock:
            self._parsed[tid] = base
            self._depends[tid] = set(depends_on)

        results.append({"task_id": tid, "parsed": base, "depends_on": depends_on})
        return results

    def mark_released(self, task_id: str) -> None:
        """Mark a task as already handed to dispatcher (prevents re-dispatch)."""
        with self._lock:
            self._released.add(task_id)

    # -------------------------
    # Queries
    # -------------------------
    def get_parsed(self, task_id: str) -> Dict[str, Any] | None:
        with self._lock:
            return self._parsed.get(task_id)

    def get_dependencies(self, task_id: str) -> List[str]:
        with self._lock:
            return list(self._depends.get(task_id, set()))

    def list_waiting_tasks(self) -> List[str]:
        with self._lock:
            return [tid for tid in self._parsed.keys() if tid not in self._released]

    def ready_to_dispatch(self, is_dep_satisfied: Callable[[str], bool]) -> List[str]:
        """Return task_ids that are not released and have all deps satisfied."""
        ready: List[str] = []
        with self._lock:
            for tid, deps in self._depends.items():
                if tid in self._released:
                    continue
                if all(is_dep_satisfied(did) for did in deps):
                    ready.append(tid)
        return ready
