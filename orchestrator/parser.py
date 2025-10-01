# parser.py
"""Task parsing and in-memory dependency tracking (supports stages and DAG graphs).

Unchanged core behavior:
- REQUIRED_FIELDS, single-task validation, optional staged pipelines.
- The scheduler may choose to shard at dispatch time; those child shards
  are internal to the orchestrator and are NOT registered in TaskStore,
  so downstream dependencies continue to track the parent logical task.
"""
from __future__ import annotations

import copy
import threading
import uuid
from typing import Any, Dict, List, Set, Callable

from fastapi import HTTPException
import yaml

from utils import safe_get
from task_load import compute_task_load, DEFAULT_TASK_LOAD


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
        self._load: Dict[str, int] = {}

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
        base_load = compute_task_load(base)
        stages = safe_get(base, "spec.stages", None)
        graph_nodes = safe_get(base, "spec.graph.nodes", None)

        # Base copy for children to inherit from
        base_clean = copy.deepcopy(base)
        if isinstance(safe_get(base_clean, "spec", {}), dict):
            base_clean["spec"].pop("stages", None)
            base_clean["spec"].pop("graph", None)

        results: List[Dict[str, Any]] = []

        if isinstance(graph_nodes, list) and graph_nodes:
            return self._register_graph(base, base_clean, graph_nodes)

        if isinstance(stages, list) and stages:
            return self._register_linear_stages(base, base_clean, stages)

        # Single-task path (no stages)
        depends_on_raw = safe_get(base, "spec.dependsOn", []) or []
        if not isinstance(depends_on_raw, list):
            raise HTTPException(status_code=400, detail="spec.dependsOn must be a list of task_ids")
        depends_on = [str(x).strip() for x in depends_on_raw if str(x).strip()]

        tid = str(uuid.uuid4())
        with self._lock:
            self._parsed[tid] = base
            self._depends[tid] = set(depends_on)
            self._load[tid] = base_load

        results.append({
            "task_id": tid,
            "parsed": base,
            "depends_on": depends_on,
            "graph_node_name": None,
            "load": base_load,
        })
        return results

    def _register_linear_stages(
        self,
        base: Dict[str, Any],
        base_clean: Dict[str, Any],
        stages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        prev_task_id: str | None = None
        base_name = str(safe_get(base, "metadata.name", "task"))

        for idx, stage in enumerate(stages):
            stage_name = str(stage.get("name") or f"stage-{idx+1}")
            stage_overrides = stage.get("spec") or {}

            effective = _deep_merge(base_clean, {"spec": stage_overrides})
            eff = copy.deepcopy(effective)
            eff_md = eff.setdefault("metadata", {})
            eff_md["name"] = f"{base_name}:{stage_name}"
            load = compute_task_load(eff)

            depends_on_names = stage.get("dependsOn", []) or []
            if not isinstance(depends_on_names, list):
                raise HTTPException(status_code=400, detail="stage.dependsOn must be a list")

            depends_on: List[str] = [str(dep).strip() for dep in depends_on_names if str(dep).strip()]
            if prev_task_id:
                depends_on.insert(0, prev_task_id)

            tid = str(uuid.uuid4())
            with self._lock:
                self._parsed[tid] = eff
                self._depends[tid] = set(depends_on)
                self._load[tid] = load

            results.append({
                "task_id": tid,
                "parsed": eff,
                "depends_on": depends_on,
                "graph_node_name": stage_name,
                "load": load,
            })
            prev_task_id = tid

        return results

    def _register_graph(
        self,
        base: Dict[str, Any],
        base_clean: Dict[str, Any],
        nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not nodes:
            return []

        name_to_node: Dict[str, Dict[str, Any]] = {}
        base_name = str(safe_get(base, "metadata.name", "task"))

        for idx, node in enumerate(nodes):
            name = node.get("name") or node.get("id") or f"node-{idx+1}"
            name = str(name).strip()
            if not name:
                raise HTTPException(status_code=400, detail="Each graph node requires a non-empty name")
            if name in name_to_node:
                raise HTTPException(status_code=400, detail=f"Duplicate graph node name '{name}'")
            name_to_node[name] = dict(node)

        unresolved = dict(name_to_node)
        resolved_ids: Dict[str, str] = {}
        results: List[Dict[str, Any]] = []

        while unresolved:
            progressed = False
            for name, node in list(unresolved.items()):
                depends_raw = node.get("dependsOn") or []
                if not isinstance(depends_raw, list):
                    raise HTTPException(status_code=400, detail=f"Node '{name}' dependsOn must be a list")

                internal_deps = [d for d in depends_raw if d in unresolved]
                if internal_deps:
                    continue  # wait for dependencies to resolve

                if any(d not in resolved_ids and d in name_to_node for d in depends_raw):
                    continue

                depends_on_task_ids: List[str] = []
                for dep in depends_raw:
                    dep = str(dep).strip()
                    if not dep:
                        continue
                    if dep in resolved_ids:
                        depends_on_task_ids.append(resolved_ids[dep])
                    elif dep in name_to_node:
                        # unresolved internal dependency; skip for now
                        break
                    else:
                        depends_on_task_ids.append(dep)
                else:
                    stage_overrides = node.get("spec") or {}
                    effective = _deep_merge(base_clean, {"spec": stage_overrides})
                    eff = copy.deepcopy(effective)
                    eff_md = eff.setdefault("metadata", {})
                    eff_md["name"] = f"{base_name}:{name}"

                    load = compute_task_load(eff)
                    tid = str(uuid.uuid4())
                    with self._lock:
                        self._parsed[tid] = eff
                        self._depends[tid] = set(depends_on_task_ids)
                        self._load[tid] = load

                    results.append({
                        "task_id": tid,
                        "parsed": eff,
                        "depends_on": depends_on_task_ids,
                        "graph_node_name": name,
                        "load": load,
                    })
                    resolved_ids[name] = tid
                    unresolved.pop(name)
                    progressed = True
            if not progressed:
                unresolved_names = ", ".join(unresolved.keys())
                raise HTTPException(status_code=400, detail=f"Graph contains cycles or unresolved deps: {unresolved_names}")

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

    def get_load(self, task_id: str) -> int:
        with self._lock:
            return int(self._load.get(task_id, DEFAULT_TASK_LOAD))

    def list_waiting_tasks(self) -> List[str]:
        """Tasks not yet 'released' to dispatcher (pending capacity/deps)."""
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
