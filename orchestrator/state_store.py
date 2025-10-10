from __future__ import annotations

"""任务状态持久化与恢复工具。"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from task import TaskRecord, TaskStatus
from task_store import TaskStore
from utils import now_iso


class StateManager:
    """负责周期性写入 TaskStore/TaskRecord 快照，并支持重启恢复。"""

    def __init__(
        self,
        *,
        state_dir: Path,
        task_store: TaskStore,
        tasks: Dict[str, TaskRecord],
        tasks_lock: threading.RLock,
        parent_shards: Dict[str, Dict[str, Any]],
        child_to_parent: Dict[str, str],
        logger,
        flush_interval: float = 5.0,
    ) -> None:
        self._state_dir = Path(state_dir).expanduser().resolve()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "task_state.json"
        self._task_store = task_store
        self._tasks = tasks
        self._tasks_lock = tasks_lock
        self._parent_shards = parent_shards
        self._child_to_parent = child_to_parent
        self._logger = logger
        self._flush_interval = max(1.0, float(flush_interval))

        self._stop_event = threading.Event()
        self._dirty_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------------------------
    # Snapshot lifecycle
    # -------------------------

    def load_snapshot(self) -> List[str]:
        """从磁盘读取快照并恢复内存结构，返回需重新调度的 task_id 列表。"""
        if not self._state_path.exists():
            return []
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._logger.warning("Failed to read state snapshot: %s", exc)
            return []
        return self._apply_snapshot(payload)

    def start(self) -> None:
        """启动后台线程，周期写入快照。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="state-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程并执行一次最终写入。"""
        self._stop_event.set()
        self._dirty_event.set()
        if self._thread:
            self._thread.join(timeout=self._flush_interval + 1.0)
        try:
            self.flush()
        except Exception as exc:
            self._logger.warning("Final state flush failed: %s", exc)

    def mark_dirty(self) -> None:
        """标记状态已变化，触发近期刷新。"""
        self._dirty_event.set()

    def flush(self) -> None:
        """立即写入快照。"""
        snapshot = self._collect_snapshot()
        if snapshot is None:
            return
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._state_path)
        self._logger.debug("State snapshot persisted at %s", self._state_path)

    # -------------------------
    # Internals
    # -------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._dirty_event.wait(self._flush_interval)
            if self._stop_event.is_set():
                break
            try:
                if triggered:
                    self._dirty_event.clear()
                self.flush()
            except Exception as exc:
                self._logger.warning("State snapshot flush failed: %s", exc)

    def _collect_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._tasks_lock:
            tasks_payload = {
                task_id: self._serialize_task_record(record)
                for task_id, record in self._tasks.items()
            }
            parent_payload: Dict[str, Dict[str, Any]] = {}
            for parent_id, data in self._parent_shards.items():
                entry = dict(data)
                children = entry.get("children", set())
                if isinstance(children, set):
                    entry["children"] = sorted(children)
                order = entry.get("order")
                if isinstance(order, dict):
                    entry["order"] = {str(k): int(v) for k, v in order.items()}
                parent_payload[parent_id] = entry

            snapshot = {
                "generated_at": now_iso(),
                "tasks": tasks_payload,
                "task_store": self._task_store.export_state(),
                "parent_shards": parent_payload,
                "child_to_parent": dict(self._child_to_parent),
            }
            return snapshot

    def _apply_snapshot(self, payload: Dict[str, Any]) -> List[str]:
        tasks_to_requeue: List[str] = []
        tasks_payload = payload.get("tasks") or {}
        parent_payload = payload.get("parent_shards") or {}
        child_payload = payload.get("child_to_parent") or {}

        with self._tasks_lock:
            self._tasks.clear()
            for task_id, record in tasks_payload.items():
                try:
                    rec = TaskRecord.model_validate(record)
                except Exception as exc:
                    self._logger.warning("Skip invalid task snapshot %s: %s", task_id, exc)
                    continue
                status = str(rec.status or "").upper()
                if status not in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.PENDING, TaskStatus.DISPATCHED, TaskStatus.WAITING}:
                    status = TaskStatus.PENDING
                if status != TaskStatus.DONE:
                    rec.status = TaskStatus.PENDING
                    rec.assigned_worker = None
                    rec.error = None
                    rec.next_retry_at = None
                    tasks_to_requeue.append(task_id)
                self._tasks[task_id] = rec

            self._parent_shards.clear()
            for parent_id, data in parent_payload.items():
                entry = dict(data)
                children = entry.get("children", [])
                if isinstance(children, list):
                    entry["children"] = set(children)
                order = entry.get("order", {})
                if isinstance(order, dict):
                    entry["order"] = {str(k): int(v) for k, v in order.items()}
                self._parent_shards[parent_id] = entry

            self._child_to_parent.clear()
            if isinstance(child_payload, dict):
                self._child_to_parent.update({str(k): str(v) for k, v in child_payload.items()})

        self._task_store.load_state(payload.get("task_store") or {})
        for task_id in tasks_to_requeue:
            self._task_store.reset_release(task_id)
        self._logger.info("Restored %d tasks from snapshot", len(tasks_payload))
        return tasks_to_requeue

    @staticmethod
    def _serialize_task_record(record: TaskRecord) -> Dict[str, Any]:
        payload = record.model_dump(mode="python")
        payload["status"] = str(payload.get("status"))
        payload["assigned_worker"] = payload.get("assigned_worker") or None
        return payload

