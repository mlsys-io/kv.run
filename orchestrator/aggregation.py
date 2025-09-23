from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import urllib.request
import urllib.error

from utils import now_iso, safe_get
from results import write_result


def maybe_aggregate_parent(
    child_task_id: str,
    child_payload: Dict[str, Any],
    *,
    child_to_parent: Dict[str, str],
    parent_shards: Dict[str, Dict[str, Any]],
    tasks: Dict[str, Any],
    tasks_lock,
    results_dir: Path,
    logger,
) -> None:
    parent_id = child_to_parent.get(child_task_id)
    if not parent_id:
        return

    with tasks_lock:
        parent_info = parent_shards.get(parent_id)
        parent_rec = tasks.get(parent_id)
        if not parent_info or not parent_rec or parent_info.get("aggregated"):
            return
        results = parent_info.setdefault("results", {})
        results[child_task_id] = child_payload
        collected = len(results)
        total = parent_info.get("total", 0)
        order_map = dict(parent_info.get("order", {}))

    if total <= 0 or collected < total:
        return

    shard_payloads = [payload for _, payload in sorted(results.items(), key=lambda kv: order_map.get(kv[0], 0))]
    aggregated_result = _build_aggregated_result(parent_rec, shard_payloads)

    aggregated_content = {
        "task_id": parent_id,
        "worker_id": "MULTI",
        "metadata": {"aggregated": True, "shards": total},
        "received_at": now_iso(),
        "result": aggregated_result,
    }

    path = write_result(results_dir, parent_id, aggregated_content)
    logger.info("Aggregated result stored for parent %s (%d shards) at %s", parent_id, total, path)

    try:
        _deliver_parent_output(parent_rec, aggregated_content, logger)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Failed to deliver aggregated result for parent %s: %s", parent_id, exc)

    with tasks_lock:
        parent_info["aggregated"] = True
        parent_info.pop("results", None)
        for cid in list(parent_info.get("children", [])):
            child_to_parent.pop(cid, None)


def _build_aggregated_result(parent_rec: Any, shard_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    parent_task = parent_rec.parsed
    task_type = str(safe_get(parent_task, "spec.taskType") or "").lower()

    aggregated: Dict[str, Any] = {"shards": shard_payloads}

    if task_type == "inference":
        merged_items: List[Any] = []
        usage_totals: Dict[str, float] = {}
        models: List[str] = []

        for payload in shard_payloads:
            shard_result = payload.get("result") or {}
            items = shard_result.get("items") or []
            merged_items.extend(items)

            usage = shard_result.get("usage") or {}
            for key, value in usage.items():
                try:
                    usage_totals[key] = usage_totals.get(key, 0.0) + float(value)
                except Exception:
                    pass

            model = shard_result.get("model")
            if model:
                models.append(str(model))

        merged: Dict[str, Any] = {"items": merged_items}
        if usage_totals:
            merged["usage"] = {
                key: int(val) if float(val).is_integer() else val
                for key, val in usage_totals.items()
            }
        if models:
            merged["model"] = models[0] if all(m == models[0] for m in models) else models
        aggregated["merged"] = merged

    return aggregated


def _deliver_parent_output(parent_rec: Any, payload: Dict[str, Any], logger) -> None:
    destination = safe_get(parent_rec.parsed, "spec.output.destination") or {}
    dest_type = str(destination.get("type") or "local").lower()
    if dest_type != "http":
        return

    url = destination.get("url")
    if not url:
        raise RuntimeError("spec.output.destination.url is required when type is 'http'")

    method = str(destination.get("method") or "POST").upper()
    headers = destination.get("headers") or {}
    timeout = float(destination.get("timeoutSec") or 15)

    hdrs = {str(k): str(v) for k, v in headers.items()}
    if not any(k.lower() == "content-type" for k in hdrs):
        hdrs["Content-Type"] = "application/json"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)

    with urllib.request.urlopen(request, timeout=timeout) as resp:
        status = resp.getcode()
        body = resp.read(200).decode("utf-8", "ignore")
    if status >= 400:
        raise RuntimeError(f"HTTP delivery failed ({status}): {body}")

    logger.info("Aggregated result for %s delivered to %s (%s)", parent_rec.task_id, url, status)
