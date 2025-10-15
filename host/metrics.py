from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Set, Tuple

from .event_schema import Event, TaskEvent, WorkerEvent, serialize_event
from .task_models import categorize_task_type
from .utils import parse_iso_ts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_timing_bucket() -> Dict[str, Dict[str, float]]:
    return {
        "queue": {"sum": 0.0, "count": 0},
        "execution": {"sum": 0.0, "count": 0},
        "total": {"sum": 0.0, "count": 0},
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsRecorder:
    """Aggregates task/worker metrics and persists snapshots to disk."""

    def __init__(self, base_dir: Path, logger) -> None:
        self._dir = Path(base_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._dir / "metrics.json"
        self._events_log = self._dir / "events.log"
        self._final_report_path = self._dir / "final_report.json"
        self._logger = logger
        self._lock = RLock()

        self._active_workers: Set[str] = set()
        self._counters: Dict[str, int] = {
            "tasks_submitted": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_requeued": 0,
            "workers_registered": 0,
            "workers_unregistered": 0,
            "worker_heartbeats": 0,
        }

        self._task_meta: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: Set[str] = set()

        self._timings: Dict[str, Dict[str, Dict[str, float]]] = {
            "overall": _fresh_timing_bucket(),
            "inference": _fresh_timing_bucket(),
            "training": _fresh_timing_bucket(),
            "other": _fresh_timing_bucket(),
        }

        self._worker_meta: Dict[str, Dict[str, Any]] = {}

        self._write_metrics()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_event(self, event: Event) -> None:
        if isinstance(event, TaskEvent):
            self.record_task_event(event)
        else:
            self.record_worker_event(event)

    def record_task_event(self, event: TaskEvent, *, is_child: bool = False) -> None:
        if is_child or self._is_child_task(event):
            self._append_event(event)
            self._write_metrics()
            return
        with self._lock:
            ev_type = event.type
            if ev_type == "TASK_SUCCEEDED":
                self._counters["tasks_succeeded"] += 1
            elif ev_type == "TASK_FAILED":
                self._counters["tasks_failed"] += 1
            elif ev_type == "TASK_REQUEUED":
                self._counters["tasks_requeued"] += 1
            elif ev_type == "TASK_SUBMITTED":
                self._counters["tasks_submitted"] += 1

            meta = self._ensure_task_meta(event)
            if ev_type == "TASK_SUBMITTED":
                self._on_task_submitted(meta, event)
            elif ev_type == "TASK_STARTED":
                self._on_task_started(meta, event)
            elif ev_type == "TASK_SUCCEEDED":
                self._on_task_succeeded(meta, event)
            elif ev_type == "TASK_FAILED":
                self._on_task_failed(meta, event)
            elif ev_type == "TASK_REQUEUED":
                self._on_task_requeued(meta, event)

            self._append_event(event)
        self._write_metrics()

    def _is_child_task(self, event: TaskEvent) -> bool:
        payload = event.payload or {}
        if payload.get("is_child_task") is True:
            return True
        if payload.get("parent_task_id"):
            return True
        shard_index = payload.get("shard_index")
        if shard_index is None:
            shard_index = payload.get("shard", {}).get("index") if isinstance(payload.get("shard"), dict) else None
        try:
            shard_index = int(shard_index) if shard_index is not None else None
        except (TypeError, ValueError):
            shard_index = None
        return shard_index not in (None, 0)

    def record_worker_event(self, event: WorkerEvent) -> None:
        with self._lock:
            ev_type = event.type
            worker_id = (event.worker_id or "").strip()

            if ev_type == "REGISTER" and worker_id:
                self._active_workers.add(worker_id)
                self._counters["workers_registered"] += 1
                self._on_worker_register(worker_id, event)
            elif ev_type == "UNREGISTER" and worker_id:
                self._active_workers.discard(worker_id)
                self._counters["workers_unregistered"] += 1
                self._on_worker_unregister(worker_id, event)
            elif ev_type == "HEARTBEAT":
                self._counters["worker_heartbeats"] += 1
                self._on_worker_heartbeat(worker_id, event)
            elif ev_type == "STATUS":
                self._on_worker_status(worker_id, event)

            self._append_event(event)
            self._write_metrics()

    def finalize_task_failure(self, task_id: str) -> None:
        with self._lock:
            meta = self._task_meta.get(task_id)
            if not meta or meta.get("finalized"):
                return

            failure = meta.get("pending_failure")
            finished_ts = meta.get("finished_ts")
            if failure:
                queue_time = failure.get("queue")
                exec_time = failure.get("execution")
                total_time = failure.get("total")
            else:
                queue_time, exec_time, total_time = self._compute_durations(
                    meta,
                    finished_ts or time.time(),
                    None,
                )

            self._record_timing(meta, queue_time, exec_time, total_time)

            meta["finalized"] = True
            meta["pending_failure"] = None
            meta["last_durations"] = {
                "queue": queue_time,
                "execution": exec_time,
                "total": total_time,
                "status": "failed",
            }
            self._completed_tasks.add(task_id)
            self._write_metrics()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_snapshot()

    def export_final_report(self) -> Dict[str, Any]:
        with self._lock:
            report = self._build_snapshot()
            report["finalized_at"] = _now_iso()
            try:
                self._final_report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                path = str(self._final_report_path)
            except Exception as exc:  # noqa: broad-except
                path = ""
                self._logger.warning("Failed to write final metrics report: %s", exc)
            return {"path": path, "report": report}

    def format_report(self, report: Dict[str, Any]) -> str:
        counters = report.get("counters", {})
        timings = report.get("task_timings", {})
        workers = report.get("workers", {})

        def _fmt(avg: Optional[float]) -> str:
            return f"{avg:.2f}s" if isinstance(avg, (int, float)) else "n/a"

        overall = timings.get("overall", {})
        lines = [
            "==== Final Metrics ====",
            f"Tasks: submitted={counters.get('tasks_submitted', 0)}, "
            f"succeeded={counters.get('tasks_succeeded', 0)}, "
            f"failed={counters.get('tasks_failed', 0)}, "
            f"requeued={counters.get('tasks_requeued', 0)}",
            "Average Durations (overall): "
            f"queue={_fmt(overall.get('queue_avg_sec'))}, "
            f"execution={_fmt(overall.get('execution_avg_sec'))}, "
            f"total={_fmt(overall.get('total_avg_sec'))}",
        ]

        totals = workers.get("totals", {})
        lines.append(
            "Workers: active=%d, total_cost=$%.2f"
            % (
                report.get("active_workers_count", 0),
                totals.get("accrued_cost_usd", 0.0) or 0.0,
            )
        )
        energy_total = totals.get("estimated_energy_kwh")
        cpu_energy = totals.get("cpu_energy_kwh")
        gpu_energy = totals.get("gpu_energy_kwh")
        reporters = int(totals.get("workers_with_energy", 0) or 0)
        uptime_hours = (totals.get("uptime_sec") or 0.0) / 3600.0

        def _fmt_energy(value: Optional[float]) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.3f}kWh"
            return "n/a"

        lines.append(
            "Energy: total=%s, cpu=%s, gpu=%s, uptime=%.2fh, reporters=%d"
            % (
                _fmt_energy(energy_total),
                _fmt_energy(cpu_energy),
                _fmt_energy(gpu_energy),
                uptime_hours,
                reporters,
            )
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Task helpers
    # ------------------------------------------------------------------ #

    def _ensure_task_meta(self, event: TaskEvent) -> Dict[str, Any]:
        task_id = event.task_id
        meta = self._task_meta.get(task_id)
        if not meta:
            meta = {
                "task_type": None,
                "category": "other",
                "submitted_ts": None,
                "last_queue_ts": None,
                "dispatched_ts": None,
                "start_ts": None,
                "finished_ts": None,
                "attempts": 0,
                "pending_failure": None,
                "finalized": False,
                "last_durations": None,
            }
            self._task_meta[task_id] = meta

        payload = event.payload or {}
        if not meta["task_type"] and payload.get("taskType"):
            meta["task_type"] = payload["taskType"]
            meta["category"] = categorize_task_type(meta["task_type"])
            self._ensure_category(meta["category"])
        return meta

    def _ensure_category(self, category: str) -> None:
        if category not in self._timings:
            self._timings[category] = _fresh_timing_bucket()

    def _on_task_submitted(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        ts = parse_iso_ts(event.ts)
        meta["submitted_ts"] = ts
        meta["last_queue_ts"] = ts
        payload = event.payload or {}
        task_type = payload.get("taskType")
        if task_type:
            meta["task_type"] = task_type
            meta["category"] = categorize_task_type(task_type)
            self._ensure_category(meta["category"])
        meta["finalized"] = False
        meta["attempts"] = 0

    def _on_task_started(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        start_iso = payload.get("started_at") or event.ts
        start_ts = parse_iso_ts(start_iso)
        dispatched_iso = payload.get("dispatched_at")

        meta["start_ts"] = start_ts
        if dispatched_iso:
            meta["dispatched_ts"] = parse_iso_ts(dispatched_iso)
        elif meta["dispatched_ts"] is None:
            meta["dispatched_ts"] = start_ts

        meta["attempts"] = int(meta.get("attempts") or 0) + 1
        meta["pending_failure"] = None

        task_type = payload.get("taskType")
        if task_type and not meta.get("task_type"):
            meta["task_type"] = task_type
            meta["category"] = categorize_task_type(task_type)
            self._ensure_category(meta["category"])

    def _on_task_succeeded(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        finish_iso = payload.get("finished_at") or event.ts
        finished_ts = parse_iso_ts(finish_iso)
        meta["finished_ts"] = finished_ts

        runtime_override = _safe_float(payload.get("runtime_sec"))
        queue_time, exec_time, total_time = self._compute_durations(
            meta,
            finished_ts,
            runtime_override,
        )

        self._record_timing(meta, queue_time, exec_time, total_time)

        meta["finalized"] = True
        meta["last_durations"] = {
            "queue": queue_time,
            "execution": exec_time,
            "total": total_time,
            "status": "succeeded",
        }
        self._completed_tasks.add(event.task_id)

    def _on_task_failed(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        finish_iso = payload.get("finished_at") or event.ts
        finished_ts = parse_iso_ts(finish_iso)
        meta["finished_ts"] = finished_ts
        runtime_override = _safe_float(payload.get("runtime_sec"))
        queue_time, exec_time, total_time = self._compute_durations(
            meta,
            finished_ts,
            runtime_override,
        )
        meta["pending_failure"] = {
            "queue": queue_time,
            "execution": exec_time,
            "total": total_time,
        }

    def _on_task_requeued(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        ts = parse_iso_ts(event.ts)
        meta["last_queue_ts"] = ts
        meta["start_ts"] = None
        meta["dispatched_ts"] = None
        meta["pending_failure"] = None

    def _compute_durations(
        self,
        meta: Dict[str, Any],
        finished_ts: float,
        runtime_override: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if finished_ts is None:
            finished_ts = time.time()

        queue_anchor = (
            meta.get("last_queue_ts")
            or meta.get("submitted_ts")
            or finished_ts
        )
        start_ts = meta.get("start_ts")

        queue_time = None
        if start_ts is not None and queue_anchor is not None:
            queue_time = max(0.0, start_ts - queue_anchor)
        elif queue_anchor is not None:
            queue_time = max(0.0, finished_ts - queue_anchor)

        exec_time = None
        if runtime_override is not None:
            exec_time = max(0.0, runtime_override)
        elif start_ts is not None:
            exec_time = max(0.0, finished_ts - start_ts)

        total_time = None
        if queue_time is not None and exec_time is not None:
            total_time = queue_time + exec_time
        elif queue_anchor is not None:
            total_time = max(0.0, finished_ts - queue_anchor)

        return queue_time, exec_time, total_time

    def _record_timing(
        self,
        meta: Dict[str, Any],
        queue_time: Optional[float],
        exec_time: Optional[float],
        total_time: Optional[float],
    ) -> None:
        category = meta.get("category") or "other"
        self._ensure_category(category)
        for bucket_name in {"overall", category}:
            bucket = self._timings[bucket_name]
            if queue_time is not None:
                bucket["queue"]["sum"] += queue_time
                bucket["queue"]["count"] += 1
            if exec_time is not None:
                bucket["execution"]["sum"] += exec_time
                bucket["execution"]["count"] += 1
            if total_time is not None:
                bucket["total"]["sum"] += total_time
                bucket["total"]["count"] += 1

    # ------------------------------------------------------------------ #
    # Worker helpers
    # ------------------------------------------------------------------ #

    def _on_worker_register(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["registered_at"] = event.ts
        payload = event.payload or {}
        meta["cost_per_hour"] = _safe_float(payload.get("cost_per_hour"))
        power = payload.get("power_metrics") or {}
        if isinstance(power, dict):
            meta["power_samples"] = [power]

    def _on_worker_unregister(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["unregistered_at"] = event.ts
        payload = event.payload or {}
        meta["cost_per_hour"] = meta.get("cost_per_hour") or _safe_float(payload.get("cost_per_hour"))
        meta["uptime_sec"] = _safe_float(payload.get("uptime_sec"))
        meta["accrued_cost_usd"] = _safe_float(payload.get("accrued_cost_usd"))
        summary = payload.get("power_summary")
        if isinstance(summary, dict):
            meta["power_summary"] = summary

    def _on_worker_heartbeat(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        metrics = event.metrics or {}
        uptime = _safe_float(metrics.get("uptime_sec"))
        if uptime is not None:
            meta["uptime_sec"] = uptime
        accrued_cost = _safe_float(metrics.get("accrued_cost_usd"))
        if accrued_cost is not None:
            meta["accrued_cost_usd"] = accrued_cost
        energy_total = _safe_float(metrics.get("estimated_energy_kwh"))
        if energy_total is not None:
            meta["estimated_energy_kwh"] = energy_total
        power_summary = metrics.get("power_summary")
        if isinstance(power_summary, dict):
            meta["power_summary"] = power_summary
        power = metrics.get("power")
        if power and isinstance(power, dict):
            samples = meta.setdefault("power_samples", [])
            samples.append(power)

    def _on_worker_status(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["status"] = event.status

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _append_event(self, event: Event) -> None:
        try:
            serialized = serialize_event(event)
            with self._events_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(serialized, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Failed to log event %s: %s", event, exc)

    def _write_metrics(self) -> None:
        snapshot = self._build_snapshot()
        try:
            self._metrics_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Failed to write metrics snapshot: %s", exc)

    def _build_snapshot(self) -> Dict[str, Any]:
        counters = dict(self._counters)
        timings = {
            bucket: self._summarize_timing(values)
            for bucket, values in self._timings.items()
        }
        workers = self._summarize_workers()
        return {
            "generated_at": _now_iso(),
            "counters": counters,
            "task_timings": timings,
            "workers": workers,
            "active_workers_count": len(self._active_workers),
            "completed_tasks": sorted(self._completed_tasks),
        }

    def _summarize_timing(self, bucket: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for name, payload in bucket.items():
            count = payload.get("count", 0)
            total = payload.get("sum", 0.0)
            summary[f"{name}_avg_sec"] = (total / count) if count else None
            summary[f"{name}_count"] = count
            summary[f"{name}_total_sec"] = total
        return summary

    def _summarize_workers(self) -> Dict[str, Any]:
        totals: Dict[str, Any] = {
            "accrued_cost_usd": 0.0,
            "uptime_sec": 0.0,
            "estimated_energy_kwh": 0.0,
            "cpu_energy_kwh": 0.0,
            "gpu_energy_kwh": 0.0,
            "workers_with_energy": 0,
        }
        detailed = {}
        for worker_id, meta in self._worker_meta.items():
            info = dict(meta)
            cost = _safe_float(info.get("cost_per_hour"))
            uptime = _safe_float(info.get("uptime_sec"))
            accrued_cost = _safe_float(info.get("accrued_cost_usd"))
            if uptime is not None:
                totals["uptime_sec"] += uptime
            if accrued_cost is not None:
                totals["accrued_cost_usd"] += accrued_cost
            elif cost is not None and uptime is not None:
                totals["accrued_cost_usd"] += (cost / 3600.0) * uptime
            summary = info.get("power_summary")
            if isinstance(summary, dict):
                energy_total = _safe_float(summary.get("estimated_energy_kwh"))
                breakdown = summary.get("estimated_energy_breakdown") or {}
                if energy_total is not None:
                    totals["estimated_energy_kwh"] += energy_total
                    totals["workers_with_energy"] += 1
                    info["estimated_energy_kwh"] = energy_total
                if isinstance(breakdown, dict):
                    cpu_energy = _safe_float(breakdown.get("cpu_kwh"))
                    gpu_energy = _safe_float(breakdown.get("gpu_kwh"))
                    if cpu_energy is not None:
                        totals["cpu_energy_kwh"] += cpu_energy
                    if gpu_energy is not None:
                        totals["gpu_energy_kwh"] += gpu_energy
                    info["estimated_energy_breakdown"] = breakdown
            detailed[worker_id] = info
        if totals["workers_with_energy"] == 0:
            totals["estimated_energy_kwh"] = None
            totals["cpu_energy_kwh"] = None
            totals["gpu_energy_kwh"] = None
        return {
            "totals": totals,
            "workers": detailed,
        }
