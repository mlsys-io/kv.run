from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional
import json
import threading
import time
import urllib.request
import urllib.error

from utils import now_iso, safe_get
from results import write_result


def maybe_aggregate_parent(
    child_task_id: str,
    child_payload: Dict[str, Any],
    *,
    child_to_parent: MutableMapping[str, str],
    parent_shards: MutableMapping[str, Dict[str, Any]],
    tasks: MutableMapping[str, Any],
    tasks_lock: threading.RLock,
    results_dir: Path,
    logger,
) -> None:
    """
    Collect a shard result and, when all shards have arrived, aggregate the
    parent result exactly once and (optionally) deliver it to an HTTP endpoint.

    Concurrency model:
      1) Under the lock, we persist the arriving child result into the shared
         results map (idempotently).
      2) If and only if collected == total, we atomically flip an 'aggregating'
         flag under the same lock, then release the lock and perform I/O.
      3) Other threads arriving while aggregation is in progress see the flag
         and return early; no data is lost because the result is recorded
         before the flag check.

    Data structures (per parent_id, in parent_shards[parent_id]):
      {
        "total": int,                         # total shards
        "order": {child_id: rank, ...},       # deterministic order for merge
        "children": set(child_ids),           # bookkeeping
        "results": {child_id: payload, ...},  # collected child payloads
        "aggregating": bool,                  # in-progress aggregation
        "aggregated": bool,                   # finalization done
      }
    """
    parent_id = child_to_parent.get(child_task_id)
    if not parent_id:
        return

    # ---- Record the child payload (idempotent) under the lock ----
    with tasks_lock:
        info = parent_shards.get(parent_id)
        parent_rec = tasks.get(parent_id)
        if not info or not parent_rec:
            return

        # If already finalized or in-flight aggregation, no need to proceed.
        if info.get("aggregated") or info.get("aggregating"):
            return

        results: Dict[str, Any] = info.setdefault("results", {})

        # Idempotency: avoid churn if identical content already recorded.
        prev = results.get(child_task_id)
        if prev is None or not _json_equal(prev, child_payload):
            results[child_task_id] = child_payload

        collected = len(results)
        total = int(info.get("total") or 0)
        order_map: Dict[str, int] = dict(info.get("order") or {})

        # Not all shards present yet; nothing else to do now.
        if total <= 0 or collected < total:
            return

        # Exactly one thread flips to "aggregating" and proceeds to I/O.
        info["aggregating"] = True

    # ---- Outside the lock: deterministic ordering, merge, write, deliver ----
    default_rank = len(order_map) + 999  # missing keys drift to the tail
    ordered_items = sorted(
        results.items(),
        key=lambda kv: order_map.get(kv[0], default_rank),
    )
    shard_payloads = [payload for _, payload in ordered_items]

    aggregated_result = _build_aggregated_result(parent_rec, shard_payloads)

    aggregated_content = {
        "task_id": parent_id,
        "worker_id": "MULTI",
        "metadata": {"aggregated": True, "shards": total},
        "received_at": now_iso(),
        "result": aggregated_result,
    }

    path = write_result(results_dir, parent_id, aggregated_content)
    logger.info(
        "Aggregated result stored for parent %s (%d shards) at %s",
        parent_id, total, path
    )

    try:
        _deliver_parent_output(parent_rec, aggregated_content, logger)
    except Exception as exc:  # pragma: no cover (best-effort delivery)
        logger.error(
            "Failed to deliver aggregated result for parent %s: %s",
            parent_id, exc
        )

    # ---- Cleanup and finalize under the lock ----
    with tasks_lock:
        info = parent_shards.get(parent_id)
        if not info:
            return
        info["aggregated"] = True
        info.pop("aggregating", None)
        info.pop("results", None)
        for cid in list(info.get("children", [])):
            child_to_parent.pop(cid, None)
        parent_shards.pop(parent_id, None)


def _json_equal(a: Any, b: Any) -> bool:
    """
    Structural idempotency check that is resilient to key-order differences.
    Falls back to False on serialization errors.
    """
    try:
        return json.dumps(a, sort_keys=True, ensure_ascii=False) == \
               json.dumps(b, sort_keys=True, ensure_ascii=False)
    except Exception:
        return False


def _build_aggregated_result(
    parent_rec: Any,
    shard_payloads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build the aggregated 'result' object placed under the parent payload.

    Generic structure:
      {
        "shards": [child_payload_0, child_payload_1, ...],
        # Optional 'merged' for known task types (e.g., inference)
      }

    For taskType == "inference":
      - items: concatenated list of per-shard result.items
      - usage: per-key numeric sum across shards
      - model: single value if all equal, else list of observed values
    """
    parent_task = parent_rec.parsed
    task_type = str(safe_get(parent_task, "spec.taskType") or "").lower()

    aggregated: Dict[str, Any] = {"shards": shard_payloads}

    if task_type == "inference":
        merged_items: List[Any] = []
        usage_totals: Dict[str, float] = {}
        models: List[str] = []

        for payload in shard_payloads:
            shard_result = payload.get("result") or {}

            # Merge list items
            items = shard_result.get("items") or []
            if isinstance(items, list):
                merged_items.extend(items)

            # Sum numeric usage metrics per key
            usage = shard_result.get("usage") or {}
            if isinstance(usage, dict):
                for key, value in usage.items():
                    try:
                        usage_totals[key] = usage_totals.get(key, 0.0) + float(value)
                    except Exception:
                        # Non-numeric values are ignored in the sum
                        continue

            # Track model identity across shards
            model = shard_result.get("model")
            if model is not None:
                models.append(str(model))

        merged: Dict[str, Any] = {"items": merged_items}
        if usage_totals:
            # Render integers without decimal points if possible
            merged["usage"] = {
                key: int(val) if float(val).is_integer() else val
                for key, val in usage_totals.items()
            }
        if models:
            merged["model"] = models[0] if all(m == models[0] for m in models) else models

        aggregated["merged"] = merged

    return aggregated


def _deliver_parent_output(parent_rec: Any, payload: Dict[str, Any], logger) -> None:
    """
    Best-effort HTTP delivery for aggregated parent output.
    Uses standard library (urllib) and a small retry loop for robustness.

    spec.output.destination example:
      output:
        destination:
          type: http
          url: https://example.com/callback
          method: POST
          headers: {"Authorization": "Bearer ..."}
          timeoutSec: 15
          retries: 2
          retryBackoffSec: 0.8
    """
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

    # Minimal retry/backoff for transient failures
    max_attempts = int(destination.get("retries", 2))
    backoff_sec = float(destination.get("retryBackoffSec", 0.8))

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 2):  # attempts = retries + 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                status = resp.getcode()
                body = resp.read(2048).decode("utf-8", "ignore")
            if status >= 400:
                raise RuntimeError(f"HTTP delivery failed ({status}): {body}")
            logger.info(
                "Aggregated result for %s delivered to %s (%s)",
                parent_rec.task_id, url, status
            )
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            if attempt <= max_attempts:
                sleep_dur = backoff_sec * attempt
                logger.warning(
                    "Delivery attempt %d failed: %s; retrying in %.1fs",
                    attempt, exc, sleep_dur
                )
                time.sleep(sleep_dur)
            else:
                break

    raise RuntimeError(f"HTTP delivery failed after retries: {last_err}")
