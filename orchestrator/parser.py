from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import HTTPException

from load_func import compute_task_load
from results import read_result
from utils import safe_get


# Required minimal fields for a task spec.
REQUIRED_FIELDS = [
    "apiVersion",
    "kind",
    "metadata.name",
    "spec.taskType",
    "spec.resources.replicas",
    "spec.resources.hardware.cpu",
    "spec.resources.hardware.memory",
]

# Matches ${path.to.value[0]} placeholders in strings.
# We allow escaping via $${...} (handled in _replace_placeholders).
_PLACEHOLDER_RE = re.compile(r"(?<!\$)\$\{([^}]+)\}")


@dataclass
class ParsedTaskSpec:
    """Flattened logical task spec produced from a single YAML."""
    spec: Dict[str, Any]
    depends_on: List[str]
    local_name: Optional[str]
    graph_node_name: Optional[str]
    load: int
    slo_seconds: Optional[float]


def validate_yaml_to_dict(yaml_text: str) -> Dict[str, Any]:
    """
    Parse YAML into a dict and validate presence of required fields.
    Raises HTTP 400 with actionable error messages on failure.

    Notes:
      - If 'gpu' block exists under spec.resources.hardware.gpu, we require 'count'.
        'type' is optional to align with runtime matching fallback.
    """
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="YAML root must be a mapping/dict")

    # Required path presence checks
    for path in REQUIRED_FIELDS:
        cur: Any = data
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                raise HTTPException(status_code=400, detail=f"Missing required field: {path}")
            cur = cur[key]

    # Basic type sanity (best-effort, do not over-validate)
    _ensure_is_dict(safe_get(data, "metadata"), "metadata")
    _ensure_is_dict(safe_get(data, "spec"), "spec")
    _ensure_is_dict(safe_get(data, "spec.resources"), "spec.resources")
    _ensure_is_dict(safe_get(data, "spec.resources.hardware"), "spec.resources.hardware")

    # GPU requirements: require 'count' if gpu section exists; 'type' is optional
    gpu = safe_get(data, "spec.resources.hardware.gpu", {}) or {}
    if gpu:
        if safe_get(gpu, "count") is None:
            raise HTTPException(status_code=400, detail="Missing spec.resources.hardware.gpu.count")

    return data


def _ensure_is_dict(val: Any, path: str) -> None:
    if val is not None and not isinstance(val, dict):
        raise HTTPException(status_code=400, detail=f"Field '{path}' must be a mapping/dict")


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge (src overrides dst). Returns a new dict.
    Safe for nested structures by copying leaf nodes.
    """
    out = copy.deepcopy(dst)
    if not isinstance(src, dict):
        return out
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_task_specs(yaml_text: str) -> List[ParsedTaskSpec]:
    """
    Expand a YAML document into one or more logical task specs:

      1) Graph mode:    spec.graph.nodes[*]
      2) Stages mode:   spec.stages[*]
      3) Single-task:   base spec as-is

    Each produced spec includes computed load and SLO seconds.
    """
    base = validate_yaml_to_dict(yaml_text)

    # Base without graph/stages for merging per-node/stage overrides.
    base_clean = copy.deepcopy(base)
    if isinstance(base_clean.get("spec"), dict):
        base_clean["spec"].pop("stages", None)
        base_clean["spec"].pop("graph", None)

    stages = safe_get(base, "spec.stages")
    graph_nodes = safe_get(base, "spec.graph.nodes")

    if isinstance(graph_nodes, list) and graph_nodes:
        return _parse_graph_nodes(base, base_clean, graph_nodes)

    if isinstance(stages, list) and stages:
        return _parse_linear_stages(base, base_clean, stages)

    depends_on_raw = safe_get(base, "spec.dependsOn", []) or []
    if not isinstance(depends_on_raw, list):
        raise HTTPException(status_code=400, detail="spec.dependsOn must be a list of task_ids")
    depends_on = [str(dep).strip() for dep in depends_on_raw if str(dep).strip()]

    load = compute_task_load(base)
    slo = extract_slo_seconds(base)

    return [ParsedTaskSpec(
        spec=base,
        depends_on=depends_on,
        local_name=None,
        graph_node_name=None,
        load=load,
        slo_seconds=slo,
    )]


def extract_slo_seconds(task: Dict[str, Any]) -> Optional[float]:
    """
    Pull SLO target (in seconds) from common spec locations. Returns None if absent.
    Only positive numeric values are accepted.
    """
    candidates = (
        safe_get(task, "spec.sloSeconds"),
        safe_get(task, "spec.slo.seconds"),
        safe_get(task, "spec.latency.sloSeconds"),
        safe_get(task, "spec.runtime.sloSeconds"),
        safe_get(task, "metadata.annotations.sloSeconds"),
        safe_get(task, "spec.deadlineSeconds"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _parse_linear_stages(
    base: Dict[str, Any],
    base_clean: Dict[str, Any],
    stages: List[Dict[str, Any]],
) -> List[ParsedTaskSpec]:
    """
    Expand spec.stages[] linearly. Each stage:
      - Overrides base via deep_merge({'spec': stage.spec})
      - metadata.name becomes "<base>:<stage_name>"
      - dependsOn = stage.dependsOn + previous stage (implicit chaining)
    """
    results: List[ParsedTaskSpec] = []
    prev_stage_name: Optional[str] = None
    base_name = str(safe_get(base, "metadata.name", "task"))

    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise HTTPException(status_code=400, detail=f"spec.stages[{idx}] must be a mapping/dict")

        stage_name = str(stage.get("name") or f"stage-{idx+1}")
        stage_overrides = stage.get("spec") or {}
        if not isinstance(stage_overrides, dict):
            raise HTTPException(status_code=400, detail=f"spec.stages[{idx}].spec must be a mapping/dict")

        effective = deep_merge(base_clean, {"spec": stage_overrides})
        eff = copy.deepcopy(effective)
        eff_md = eff.setdefault("metadata", {})
        eff_md["name"] = f"{base_name}:{stage_name}"

        load = compute_task_load(eff)
        slo = extract_slo_seconds(eff)

        depends_raw = stage.get("dependsOn") or []
        if not isinstance(depends_raw, list):
            raise HTTPException(status_code=400, detail="stage.dependsOn must be a list")
        depends_on = [str(dep).strip() for dep in depends_raw if str(dep).strip()]
        if prev_stage_name:
            # Implicit linear dependency: current depends on the previous stage
            depends_on.insert(0, prev_stage_name)

        results.append(ParsedTaskSpec(
            spec=eff,
            depends_on=depends_on,
            local_name=stage_name,
            graph_node_name=stage_name,
            load=load,
            slo_seconds=slo,
        ))
        prev_stage_name = stage_name

    return results


def _parse_graph_nodes(
    base: Dict[str, Any],
    base_clean: Dict[str, Any],
    nodes: List[Dict[str, Any]],
) -> List[ParsedTaskSpec]:
    """
    Topologically expand spec.graph.nodes[] into runnable nodes.
    Detects cycles / unresolved dependencies and raises HTTP 400.
    """
    if not nodes:
        return []

    name_to_node: Dict[str, Dict[str, Any]] = {}
    base_name = str(safe_get(base, "metadata.name", "task"))

    # Normalize and index nodes by name
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise HTTPException(status_code=400, detail=f"graph.nodes[{idx}] must be a mapping/dict")
        name = node.get("name") or node.get("id") or f"node-{idx+1}"
        name = str(name).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Each graph node requires a non-empty name")
        if name in name_to_node:
            raise HTTPException(status_code=400, detail=f"Duplicate graph node name '{name}'")
        # Quick self-dependency guard
        deps = node.get("dependsOn") or []
        if any(str(d).strip() == name for d in deps if d is not None):
            raise HTTPException(status_code=400, detail=f"Node '{name}' cannot depend on itself")
        name_to_node[name] = dict(node)

    unresolved = dict(name_to_node)
    results: List[ParsedTaskSpec] = []

    # Kahn-like loop: pick nodes whose deps are all resolved
    while unresolved:
        progressed = False
        for name, node in list(unresolved.items()):
            depends_raw = node.get("dependsOn") or []
            if not isinstance(depends_raw, list):
                raise HTTPException(status_code=400, detail=f"Node '{name}' dependsOn must be a list")

            unresolved_internal = False
            depends_on: List[str] = []

            for dep in depends_raw:
                dep = str(dep).strip()
                if not dep:
                    continue
                if dep in unresolved:
                    unresolved_internal = True
                    break
                depends_on.append(dep)

            if unresolved_internal:
                continue  # still waiting for its deps

            stage_overrides = node.get("spec") or {}
            if not isinstance(stage_overrides, dict):
                raise HTTPException(status_code=400, detail=f"Node '{name}'.spec must be a mapping/dict")

            effective = deep_merge(base_clean, {"spec": stage_overrides})
            eff = copy.deepcopy(effective)
            eff_md = eff.setdefault("metadata", {})
            eff_md["name"] = f"{base_name}:{name}"

            load = compute_task_load(eff)
            slo = extract_slo_seconds(eff)

            results.append(ParsedTaskSpec(
                spec=eff,
                depends_on=depends_on,
                local_name=name,
                graph_node_name=name,
                load=load,
                slo_seconds=slo,
            ))
            unresolved.pop(name)
            progressed = True

        if not progressed:
            unresolved_names = ", ".join(unresolved.keys())
            raise HTTPException(status_code=400, detail=f"Graph contains cycles or unresolved deps: {unresolved_names}")

    return results


def resolve_graph_templates(
    task: Dict[str, Any],
    task_id: str,
    rec,
    task_store,
    tasks: Dict[str, Any],
    results_dir: Path,
    logger,
) -> Dict[str, Any]:
    """
    Resolve ${...} placeholders in the task by injecting upstream results.

    Steps:
      1) Ask task_store for dependency ids for this task_id.
      2) Build a context dict keyed by upstream node name.
      3) Attach context under spec._upstreamResults for optional direct access.
      4) Walk the task dict; for any string containing ${...}, replace it
         with the referenced value from context (dict/list traversal only).

    Safety:
      - No eval; only structural traversal by keys and numeric indexes.
      - If a path is missing or invalid, we log at debug and substitute "".
      - Escaped placeholders $${...} are left intact (becomes ${...}).
    """
    deps = task_store.get_dependencies(task_id)
    if not deps:
        return task

    context: Dict[str, Any] = {}
    for dep_id in deps:
        dep_rec = tasks.get(dep_id)
        if not dep_rec:
            continue
        node_name = getattr(dep_rec, "graph_node_name", None) or _derive_node_name(dep_rec)
        if not node_name:
            continue
        try:
            raw = read_result(results_dir, dep_id)
            data = json.loads(raw)
        except FileNotFoundError:
            logger.debug("Dependency %s result not found for task %s", dep_id, task_id)
            continue
        except json.JSONDecodeError as exc:
            logger.debug("Failed to parse result for %s: %s", dep_id, exc)
            continue

        # Normalize: ensure {'result': ...} exists for consistent addressing
        if isinstance(data, dict) and "result" not in data:
            data = {"result": data}
        context[node_name] = data

    resolved = copy.deepcopy(task)

    if deps:
        spec = resolved.setdefault("spec", {})
        # Make upstream results accessible to the spec (optional consumption).
        spec["_upstreamResults"] = context

    if context:
        _walk_and_replace(resolved, context, logger)

    return resolved


def _derive_node_name(dep_rec) -> str:
    """
    Derive a node name from dep_rec.parsed.metadata.name.
    Expected format: "<base_name>:<node_name>" -> returns "<node_name>".
    """
    meta_name = safe_get(dep_rec.parsed, "metadata.name")
    if not isinstance(meta_name, str):
        return ""
    if ":" in meta_name:
        return meta_name.split(":", 1)[1]
    return meta_name


def _walk_and_replace(value: Any, context: Dict[str, Any], logger) -> Any:
    """
    Recursively traverse the task structure, replacing string placeholders.
    Lists and dicts are processed in place; scalars returned as-is.
    """
    if isinstance(value, dict):
        for key, val in list(value.items()):
            value[key] = _walk_and_replace(val, context, logger)
        return value
    if isinstance(value, list):
        for idx, item in enumerate(value):
            value[idx] = _walk_and_replace(item, context, logger)
        return value
    if isinstance(value, str) and "${" in value:
        return _replace_placeholders(value, context, logger)
    return value


def _replace_placeholders(text: str, context: Dict[str, Any], logger) -> str:
    """
    Replace ${expr} with resolved values from context.
    Supports escaping with $${...} -> becomes ${...} (no substitution).
    """

    # Unescape: $${foo}  -> ${foo}
    text = text.replace("$${", "${")

    def repl(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        resolved = _evaluate_expr(expr, context, logger)
        return "" if resolved is None else str(resolved)

    return _PLACEHOLDER_RE.sub(repl, text)


def _evaluate_expr(expr: str, context: Dict[str, Any], logger) -> Any:
    """
    Evaluate a dotted path with optional list indexes against the context.
    Example: nodeA.result.items[0].id

    Semantics:
      - The first token is the root key into context (node name).
      - Each following token may be "attr" or "attr[0][1]".
      - For dicts, we require exact keys; for lists, integer indexes.
      - On any missing/invalid step, log at debug and return None.
    """
    if not expr:
        return None
    parts = expr.split(".")
    if not parts:
        return None

    root_name = parts[0]
    data = context.get(root_name)
    if data is None:
        logger.debug("Template placeholder root '%s' not found in context", root_name)
        return None

    value: Any = data
    for token in parts[1:]:
        if not token:
            continue
        attr, indexes = _split_indexes(token, logger, full_expr=expr)

        if attr:
            if isinstance(value, dict) and attr in value:
                value = value[attr]
            else:
                logger.debug("Template path '%s' missing attr '%s'", expr, attr)
                return None

        for idx in indexes:
            if isinstance(value, list) and 0 <= idx < len(value):
                value = value[idx]
            else:
                logger.debug("Template path '%s' invalid index %s", expr, idx)
                return None

    return value


def _split_indexes(token: str, logger=None, full_expr: str = "") -> Tuple[str, List[int]]:
    """
    Split a token like 'field[0][1]' into ('field', [0, 1]).
    Non-integer indexes are treated as invalid and reported via debug logs.
    """
    parts = token.split("[")
    attr = parts[0]
    idx_list: List[int] = []
    for part in parts[1:]:
        part = part.rstrip("]")
        if not part:
            continue
        try:
            idx_list.append(int(part))
        except ValueError:
            if logger:
                logger.debug("Template path '%s' has non-integer index '%s'", full_expr, part)
            return attr, [-1]  # propagate invalid index sentinel
    return attr, idx_list
