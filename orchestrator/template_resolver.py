from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from results import read_result

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def resolve_graph_templates(
    task: Dict[str, Any],
    task_id: str,
    rec,
    task_store,
    tasks: Dict[str, Any],
    results_dir: Path,
    logger,
) -> Dict[str, Any]:
    deps = task_store.get_dependencies(task_id)
    if not deps:
        return task

    context: Dict[str, Any] = {}
    for dep_id in deps:
        dep_rec = tasks.get(dep_id)
        if not dep_rec:
            continue
        node_name = getattr(dep_rec, "graph_node_name", None)
        if not node_name:
            node_name = _derive_node_name(dep_rec)
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
        context[node_name] = data

    resolved = copy.deepcopy(task)

    if deps:
        spec = resolved.setdefault("spec", {})
        spec["_upstreamResults"] = context

    if context:
        _walk_and_replace(resolved, context, logger)

    return resolved


def _derive_node_name(dep_rec) -> str:
    meta_name = safe_get(dep_rec.parsed, "metadata.name")
    if not isinstance(meta_name, str):
        return ""
    if ":" in meta_name:
        return meta_name.split(":", 1)[1]
    return meta_name


def _walk_and_replace(value: Any, context: Dict[str, Any], logger) -> Any:
    if isinstance(value, dict):
        for k, v in value.items():
            value[k] = _walk_and_replace(v, context, logger)
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i] = _walk_and_replace(item, context, logger)
        return value
    if isinstance(value, str) and "${" in value:
        return _replace_placeholders(value, context, logger)
    return value


def _replace_placeholders(text: str, context: Dict[str, Any], logger) -> str:
    def repl(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        resolved = _evaluate_expr(expr, context, logger)
        return "" if resolved is None else str(resolved)

    return _PLACEHOLDER_RE.sub(repl, text)


def _evaluate_expr(expr: str, context: Dict[str, Any], logger) -> Any:
    if not expr:
        return None
    parts = expr.split('.')
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
        attr, indexes = _split_indexes(token)
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


def _split_indexes(token: str) -> (str, List[int]):
    parts = token.split('[')
    attr = parts[0]
    idx_list: List[int] = []
    for part in parts[1:]:
        part = part.rstrip(']')
        if part:
            try:
                idx_list.append(int(part))
            except ValueError:
                idx_list.append(-1)
    return attr, idx_list


from utils import safe_get  # placed at end to avoid circular import
