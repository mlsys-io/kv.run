from __future__ import annotations

"""增强版指标记录器：跟踪任务耗时、SLO 达成率与 worker 成本/功耗。"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Set, Tuple

from event_schema import Event, TaskEvent, WorkerEvent, serialize_event
from task import categorize_task_type
from utils import parse_iso_ts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_timing_bucket() -> Dict[str, Dict[str, float]]:
    return {
        "queue": {"sum": 0.0, "count": 0},
        "execution": {"sum": 0.0, "count": 0},
        "total": {"sum": 0.0, "count": 0},
    }


def _fresh_slo_bucket() -> Dict[str, int]:
    return {"with_slo": 0, "met": 0, "breached": 0, "missing": 0}


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
        self._slo: Dict[str, Dict[str, int]] = {
            bucket: _fresh_slo_bucket() for bucket in self._timings
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

    def record_task_event(self, event: TaskEvent) -> None:
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
            self._record_slo(meta, total_time, outcome="failed")

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
        slo = report.get("slo", {})
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

        slo_overall = slo.get("overall", {})
        with_slo = slo_overall.get("with_slo", 0)
        if with_slo:
            rate = slo_overall.get("completion_rate")
            rate_str = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(
                f"SLO: with_slo={with_slo}, met={slo_overall.get('met', 0)}, "
                f"breached={slo_overall.get('breached', 0)}, rate={rate_str}"
            )
        else:
            lines.append("SLO: no tasks declared SLO targets")

        totals = workers.get("totals", {})
        lines.append(
            "Workers: active=%d, total_cost=$%.2f"
            % (
                report.get("active_workers_count", 0),
                totals.get("accrued_cost_usd", 0.0) or 0.0,
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
                "slo_seconds": None,
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
            self._slo[category] = _fresh_slo_bucket()

    def _on_task_submitted(self, meta: Dict[str, Any], event: TaskEvent) -> None:
        ts = parse_iso_ts(event.ts)
        meta["submitted_ts"] = ts
        meta["last_queue_ts"] = ts
        payload = event.payload or {}
        slo = _safe_float(payload.get("sloSeconds"))
        if slo is not None:
            meta["slo_seconds"] = slo
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
        self._record_slo(meta, total_time, outcome="success")

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

        if runtime_override is not None:
            exec_time = max(0.0, runtime_override)
        elif start_ts is not None:
            exec_time = max(0.0, finished_ts - start_ts)
        else:
            exec_time = None

        total_time = None
        if queue_anchor is not None:
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
        for bucket in ("overall", category):
            timings = self._timings[bucket]
            if queue_time is not None:
                timings["queue"]["sum"] += queue_time
                timings["queue"]["count"] += 1
            if exec_time is not None:
                timings["execution"]["sum"] += exec_time
                timings["execution"]["count"] += 1
            if total_time is not None:
                timings["total"]["sum"] += total_time
                timings["total"]["count"] += 1

    def _record_slo(self, meta: Dict[str, Any], total_time: Optional[float], *, outcome: str) -> None:
        category = meta.get("category") or "other"
        self._ensure_category(category)

        slo_value = meta.get("slo_seconds")
        for bucket in ("overall", category):
            stats = self._slo[bucket]
            if slo_value is None:
                stats["missing"] += 1
                continue
            stats["with_slo"] += 1
            if outcome == "success" and total_time is not None and total_time <= slo_value:
                stats["met"] += 1
            else:
                stats["breached"] += 1

    # ------------------------------------------------------------------ #
    # Worker helpers
    # ------------------------------------------------------------------ #

    def _ensure_worker_meta(self, worker_id: str) -> Dict[str, Any]:
        meta = self._worker_meta.get(worker_id)
        if meta:
            return meta
        meta = {
            "cost_per_hour": None,
            "start_ts": None,
            "end_ts": None,
            "uptime_override": None,
            "power": {
                "cpu_sum": 0.0,
                "cpu_count": 0,
                "gpu_sum": 0.0,
                "gpu_count": 0,
                "per_gpu": {},
            },
            "reported_summary": None,
        }
        self._worker_meta[worker_id] = meta
        return meta

    def _on_worker_register(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._ensure_worker_meta(worker_id)
        meta["start_ts"] = parse_iso_ts(event.ts)
        payload = event.payload or {}
        cost = _safe_float(payload.get("cost_per_hour"))
        if cost is not None:
            meta["cost_per_hour"] = cost
        power_sample = payload.get("power_metrics")
        self._update_worker_power(meta, power_sample)

    def _on_worker_unregister(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._ensure_worker_meta(worker_id)
        meta["end_ts"] = parse_iso_ts(event.ts)
        payload = event.payload or {}
        uptime = _safe_float(payload.get("uptime_sec"))
        if uptime is not None:
            meta["uptime_override"] = max(0.0, uptime)
        summary = payload.get("power_summary")
        if isinstance(summary, dict):
            meta["reported_summary"] = summary

    def _on_worker_heartbeat(self, worker_id: str, event: WorkerEvent) -> None:
        metrics = event.metrics or {}
        self._update_worker_power(self._ensure_worker_meta(worker_id), metrics.get("power"))

    def _on_worker_status(self, worker_id: str, event: WorkerEvent) -> None:
        self._on_worker_heartbeat(worker_id, event)

    def _update_worker_power(self, meta: Dict[str, Any], sample: Any) -> None:
        if not isinstance(sample, dict):
            return
        power_meta = meta["power"]

        cpu_power = _safe_float(sample.get("cpu_watts"))
        if cpu_power is not None:
            power_meta["cpu_sum"] += cpu_power
            power_meta["cpu_count"] += 1

        gpu = sample.get("gpu_watts")
        if isinstance(gpu, dict):
            total_gpu = _safe_float(gpu.get("total"))
            if total_gpu is not None:
                power_meta["gpu_sum"] += total_gpu
                power_meta["gpu_count"] += 1

            per_gpu = gpu.get("per_gpu")
            if isinstance(per_gpu, list):
                for entry in per_gpu:
                    if not isinstance(entry, dict):
                        continue
                    idx = str(entry.get("index"))
                    val = _safe_float(entry.get("power_w"))
                    if idx is None or val is None:
                        continue
                    bucket = power_meta["per_gpu"].setdefault(idx, {"sum": 0.0, "count": 0})
                    bucket["sum"] += val
                    bucket["count"] += 1

    # ------------------------------------------------------------------ #
    # Snapshot assembly
    # ------------------------------------------------------------------ #

    def _write_metrics(self) -> None:
        payload = self._build_snapshot()
        try:
            self._metrics_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: broad-except
            self._logger.warning("Failed to write metrics snapshot: %s", exc)

    def _append_event(self, event: Event) -> None:
        try:
            with self._events_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(serialize_event(event), ensure_ascii=False))
                fh.write("\n")
        except Exception as exc:  # noqa: broad-except
            self._logger.debug("Failed to append event log: %s", exc)

    def _build_snapshot(self) -> Dict[str, Any]:
        worker_entries, worker_totals = self._build_worker_snapshot()
        return {
            "last_updated": _now_iso(),
            "counters": dict(self._counters),
            "active_workers": sorted(self._active_workers),
            "active_workers_count": len(self._active_workers),
            "task_timings": self._build_timing_summary(),
            "slo": self._build_slo_summary(),
            "workers": {
                "entries": worker_entries,
                "totals": worker_totals,
            },
            "completed_tasks": len(self._completed_tasks),
        }

    def _build_timing_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for category, bucket in self._timings.items():
            summary[category] = {
                "queue_avg_sec": self._avg(bucket["queue"]),
                "queue_samples": bucket["queue"]["count"],
                "execution_avg_sec": self._avg(bucket["execution"]),
                "execution_samples": bucket["execution"]["count"],
                "total_avg_sec": self._avg(bucket["total"]),
                "total_samples": bucket["total"]["count"],
            }
        return summary

    def _build_slo_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for category, stats in self._slo.items():
            with_slo = stats.get("with_slo", 0)
            rate = (stats.get("met", 0) / with_slo) if with_slo else None
            summary[category] = {
                "with_slo": with_slo,
                "met": stats.get("met", 0),
                "breached": stats.get("breached", 0),
                "missing": stats.get("missing", 0),
                "completion_rate": rate,
            }
        return summary

    def _build_worker_snapshot(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        now = time.time()
        entries: Dict[str, Any] = {}
        total_cost = 0.0

        for worker_id, meta in self._worker_meta.items():
            start_ts = meta.get("start_ts")
            end_ts = meta.get("end_ts") or now
            uptime_override = meta.get("uptime_override")
            uptime_sec = (
                uptime_override
                if uptime_override is not None
                else (max(0.0, end_ts - start_ts) if start_ts is not None else None)
            )

            cost_per_hour = meta.get("cost_per_hour")
            accrued_cost = None
            if uptime_sec is not None and cost_per_hour is not None:
                accrued_cost = (uptime_sec / 3600.0) * cost_per_hour
                total_cost += accrued_cost

            power = meta.get("power", {})
            cpu_avg = self._avg(
                {"sum": power.get("cpu_sum", 0.0), "count": power.get("cpu_count", 0)}
            )
            gpu_avg = self._avg(
                {"sum": power.get("gpu_sum", 0.0), "count": power.get("gpu_count", 0)}
            )

            summary = meta.get("reported_summary") or {}
            if isinstance(summary, dict):
                cpu_avg = summary.get("avg_cpu_watts", cpu_avg)
                gpu_avg = summary.get("avg_gpu_watts", gpu_avg)

            per_gpu_avg = {}
            per_gpu_meta = power.get("per_gpu", {})
            for idx, stats in per_gpu_meta.items():
                per_gpu_avg[idx] = self._avg(stats)

            if isinstance(summary.get("per_gpu_avg_watts"), dict):
                per_gpu_avg.update(summary["per_gpu_avg_watts"])

            entries[worker_id] = {
                "status": "active" if worker_id in self._active_workers else "offline",
                "cost_per_hour": cost_per_hour,
                "uptime_sec": uptime_sec,
                "accrued_cost_usd": accrued_cost,
                "avg_cpu_watts": cpu_avg,
                "avg_gpu_watts": gpu_avg,
                "per_gpu_avg_watts": per_gpu_avg,
                "power_samples": {
                    "cpu": power.get("cpu_count", 0),
                    "gpu": power.get("gpu_count", 0),
                },
            }

        totals = {
            "accrued_cost_usd": total_cost,
            "worker_count": len(entries),
        }
        return entries, totals

    @staticmethod
    def _avg(bucket: Dict[str, float]) -> Optional[float]:
        count = bucket.get("count", 0)
        if not count:
            return None
        return bucket.get("sum", 0.0) / count
