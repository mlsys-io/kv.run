from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from .utils import safe_get


@dataclass
class ParsedTaskSpec:
    spec: Dict[str, Any]
    depends_on: List[str]
    local_name: Optional[str]
    graph_node_name: Optional[str]
    load: int


def parse_task_yaml(yaml_text: str) -> List[Dict[str, Any]]:
    specs = _expand_specs(yaml_text)
    results: List[Dict[str, Any]] = []
    local_ids: Dict[str, str] = {}
    for spec in specs:
        task_id = str(uuid.uuid4())
        depends_on = [
            local_ids.get(dep.strip(), dep.strip())
            for dep in spec.depends_on
            if dep and dep.strip()
        ]
        results.append(
            {
                "task_id": task_id,
                "parsed": copy.deepcopy(spec.spec),
                "depends_on": depends_on,
                "graph_node_name": spec.graph_node_name,
                "load": spec.load,
            }
        )
        if spec.local_name:
            local_ids[spec.local_name] = task_id
    return results


def _expand_specs(yaml_text: str) -> List[ParsedTaskSpec]:
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict")

    base_clean = copy.deepcopy(data)
    if isinstance(base_clean.get("spec"), dict):
        base_clean["spec"].pop("stages", None)
        base_clean["spec"].pop("graph", None)

    stages = safe_get(data, "spec.stages")
    graph_nodes = safe_get(data, "spec.graph.nodes")
    if stages and graph_nodes:
        raise ValueError("spec.stages cannot be combined with spec.graph")
    if isinstance(stages, list) and stages:
        return _expand_stages(data, base_clean, stages)
    if isinstance(graph_nodes, list) and graph_nodes:
        return _expand_graph_nodes(data, base_clean, graph_nodes)

    depends_on_raw = safe_get(data, "spec.dependsOn", []) or []
    depends_on = _normalize_dep_list(depends_on_raw)
    return [
        ParsedTaskSpec(
            spec=data,
            depends_on=depends_on,
            local_name=None,
            graph_node_name=None,
            load=_compute_task_load(data),
        )
    ]


def _expand_graph_nodes(
    base: Dict[str, Any],
    base_clean: Dict[str, Any],
    nodes: List[Dict[str, Any]],
) -> List[ParsedTaskSpec]:
    indexed: Dict[str, Dict[str, Any]] = {}
    base_name = str(safe_get(base, "metadata.name", "task"))

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"graph.nodes[{idx}] must be a mapping/dict")
        name = node.get("name") or node.get("id") or f"node-{idx+1}"
        name = str(name).strip()
        if not name:
            raise ValueError("Graph node requires a non-empty name")
        if name in indexed:
            raise ValueError(f"Duplicate graph node '{name}'")
        deps = node.get("dependsOn") or []
        if any(str(dep).strip() == name for dep in deps if dep):
            raise ValueError(f"Node '{name}' cannot depend on itself")
        indexed[name] = dict(node)

    unresolved = dict(indexed)
    results: List[ParsedTaskSpec] = []

    while unresolved:
        progressed = False
        for name, node in list(unresolved.items()):
            depends_raw = node.get("dependsOn") or []
            if not isinstance(depends_raw, list):
                raise ValueError(f"Node '{name}' dependsOn must be a list")
            pending = []
            unresolved_internal = False
            for dep in depends_raw:
                dep = str(dep).strip()
                if not dep:
                    continue
                if dep in unresolved:
                    unresolved_internal = True
                    break
                pending.append(dep)
            if unresolved_internal:
                continue

            stage_overrides = node.get("spec") or {}
            if not isinstance(stage_overrides, dict):
                raise ValueError(f"Node '{name}'.spec must be a mapping/dict")

            effective = _deep_merge(base_clean, {"spec": stage_overrides})
            eff = copy.deepcopy(effective)
            eff_md = eff.setdefault("metadata", {})
            eff_md["name"] = f"{base_name}:{name}"

            results.append(
                ParsedTaskSpec(
                    spec=eff,
                    depends_on=pending,
                    local_name=name,
                    graph_node_name=name,
                    load=_compute_task_load(eff),
                )
            )
            unresolved.pop(name)
            progressed = True

        if not progressed:
            raise ValueError(f"Graph contains cycles or unresolved dependencies: {', '.join(unresolved.keys())}")

    return results


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dst)
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _normalize_dep_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("dependsOn must be a list")
    return [str(dep).strip() for dep in raw if str(dep).strip()]


def _expand_stages(
    base: Dict[str, Any],
    base_clean: Dict[str, Any],
    stages: List[Dict[str, Any]],
) -> List[ParsedTaskSpec]:
    indexed: Dict[str, Dict[str, Any]] = {}
    base_name = str(safe_get(base, "metadata.name", "task"))

    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage[{idx}] must be a mapping/dict")
        name = stage.get("name") or stage.get("id") or f"stage-{idx+1}"
        name = str(name).strip()
        if not name:
            raise ValueError(f"Stage[{idx}] requires a non-empty name")
        if name in indexed:
            raise ValueError(f"Duplicate stage '{name}'")
        indexed[name] = dict(stage)

    results: List[ParsedTaskSpec] = []
    ordered_names = list(indexed.keys())
    for idx, name in enumerate(ordered_names):
        stage = indexed[name]
        if "spec" in stage and not isinstance(stage["spec"], dict):
            raise ValueError(f"Stage '{name}'.spec must be a mapping/dict")

        stage_spec = stage.get("spec") or {}
        effective = _deep_merge(base_clean, {"spec": stage_spec})
        eff = copy.deepcopy(effective)
        eff_md = eff.setdefault("metadata", {})
        eff_md["name"] = f"{base_name}:{name}"

        depends_raw = stage.get("dependsOn")
        depends_on = _normalize_dep_list(depends_raw) if depends_raw is not None else []
        if not depends_on and idx > 0:
            depends_on = [ordered_names[idx - 1]]

        results.append(
            ParsedTaskSpec(
                spec=eff,
                depends_on=depends_on,
                local_name=name,
                graph_node_name=name,
                load=_compute_task_load(eff),
            )
        )

    return results


def _compute_task_load(task: Dict[str, Any]) -> int:
    load_value = safe_get(task, "spec.resources.estimatedLoad")
    try:
        return int(load_value)
    except Exception:
        return 0
