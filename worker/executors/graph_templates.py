"""Utilities for building inference prompts that combine upstream graph outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .base_executor import ExecutionError


def build_prompts_from_graph_template(data_cfg: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    """Render prompt strings from a `graph_template` data configuration.

    Args:
        data_cfg: The `spec.data` section for the task.
        spec:     The full `spec` dictionary (expected to contain `_upstreamResults`).

    Returns:
        A list of prompts ready to be consumed by the inference executor.
    """

    if not isinstance(data_cfg, dict):
        raise ExecutionError("spec.data must be a mapping when type == 'graph_template'.")

    template_cfg = data_cfg.get("template") or {}
    if not isinstance(template_cfg, dict):
        raise ExecutionError("spec.data.template must be a mapping for graph templates.")

    context = (spec or {}).get("_upstreamResults") or {}
    if not context:
        raise ExecutionError("No upstream results available for graph template prompts.")

    columns_cfg = template_cfg.get("columns") or []
    columns = _resolve_columns(columns_cfg, context)

    name = str(template_cfg.get("name") or "two_column_briefing")
    options = template_cfg.get("options") or {}

    renderer = _TEMPLATE_REGISTRY.get(name)
    if renderer is None:
        text = template_cfg.get("text")
        if isinstance(text, str) and text.strip():
            return [_render_inline_text(text, columns, options)]
        raise ExecutionError(f"Unknown graph template '{name}'.")

    prompt = renderer(columns, options)
    return [prompt]


def _resolve_columns(columns_cfg: Any, context: Dict[str, Any]) -> List[Dict[str, str]]:
    if not isinstance(columns_cfg, list) or not columns_cfg:
        raise ExecutionError("graph_template.template.columns must be a non-empty list.")

    columns: List[Dict[str, str]] = []
    for idx, raw in enumerate(columns_cfg):
        if not isinstance(raw, dict):
            raise ExecutionError("Each column definition must be a mapping.")

        label = str(raw.get("label") or f"Column {idx + 1}").strip()

        expr = raw.get("expr")
        if not expr:
            node = raw.get("node")
            path = raw.get("path")
            if node and path:
                expr = f"{node}.{path}"
        if not isinstance(expr, str) or not expr.strip():
            raise ExecutionError(f"Column '{label}' is missing an expr/node+path definition.")

        value = _evaluate_expr(expr.strip(), context)
        if value is None:
            if "default" in raw:
                value = raw.get("default")
            else:
                raise ExecutionError(f"Column '{label}' expression '{expr}' resolved to null.")

        columns.append(
            {
                "label": label,
                "value": _coerce_to_string(value),
                "expr": expr,
            }
        )

    return columns


def _coerce_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _render_inline_text(text: str, columns: List[Dict[str, str]], _: Dict[str, Any]) -> str:
    mapping: Dict[str, str] = {}
    for idx, column in enumerate(columns):
        mapping[f"col{idx}_label"] = column["label"]
        mapping[f"col{idx}_value"] = column["value"]

    class _SafeDict(dict):
        def __missing__(self, key):  # type: ignore[override]
            return "{" + key + "}"

    return text.format_map(_SafeDict(mapping))


def _render_two_column_briefing(columns: List[Dict[str, str]], options: Dict[str, Any]) -> str:
    if len(columns) < 2:
        raise ExecutionError("two_column_briefing template expects at least two columns.")

    role = str(options.get("role") or "energy strategist")
    intro_lines = options.get("intro")
    closing_lines = options.get("closing")

    lines: List[str] = []

    if intro_lines:
        if isinstance(intro_lines, list):
            lines.extend(str(x) for x in intro_lines)
        else:
            lines.append(str(intro_lines))
    else:
        article = "an" if role[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        lines.append(f"You are {article} {role}. Combine the following factor analyses side by side")
        lines.append("and produce a concise, comparative briefing:")

    for column in columns:
        lines.append(_format_column_line(column["label"], column["value"]))

    if closing_lines:
        if isinstance(closing_lines, list):
            lines.extend(str(x) for x in closing_lines)
        else:
            lines.append(str(closing_lines))
    else:
        left_label = columns[0]["label"].lower()
        right_label = columns[1]["label"].lower()
        lines.append("Present them as a two-column style summary with actionable recommendations that")
        lines.append(f"weigh the {left_label} against the {right_label}.")

    return "\n".join(lines)


def _format_column_line(label: str, value: str) -> str:
    cleaned = value.replace("\r\n", "\n").strip()
    if "\n" in cleaned:
        indented = cleaned.replace("\n", "\n  ")
    else:
        indented = cleaned
    return f"• {label}: {indented}"


def _evaluate_expr(expr: str, context: Dict[str, Any]) -> Any:
    if not expr:
        return None

    parts = expr.split('.')
    root = parts[0]
    data = context.get(root)
    if data is None:
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
                return None
        for idx in indexes:
            if isinstance(value, list) and 0 <= idx < len(value):
                value = value[idx]
            else:
                return None
    return value


def _split_indexes(token: str) -> Tuple[str, List[int]]:
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


_TEMPLATE_REGISTRY = {
    "two_column_briefing": _render_two_column_briefing,
}
