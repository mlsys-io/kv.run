from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .utils import now_iso

TRAINING_TASK_TYPES = {"sft", "lora_sft", "ppo", "dpo", "training"}


def categorize_task_type(task_type: Optional[str]) -> str:
    if not task_type:
        return "other"
    normalized = task_type.strip().lower()
    if normalized == "inference":
        return "inference"
    if normalized in TRAINING_TASK_TYPES:
        return "training"
    return "other"


class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"
    DONE = "DONE"


class TaskRecord(BaseModel):
    task_id: str
    raw_yaml: str
    parsed: Dict[str, Any]
    status: str = TaskStatus.PENDING
    submission_id: Optional[str] = None
    task_type: Optional[str] = None
    category: Optional[str] = None
    assigned_worker: Optional[str] = None
    topic: Optional[str] = None
    submitted_at: str = Field(default_factory=now_iso)
    submitted_ts: float = Field(default_factory=time.time)
    last_queue_ts: float = Field(default_factory=time.time)
    dispatched_ts: Optional[float] = None
    started_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    attempts: int = 0
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    parent_task_id: Optional[str] = None
    shard_index: Optional[int] = None
    shard_total: Optional[int] = None
    next_retry_at: Optional[str] = None
    last_failed_worker: Optional[str] = None
    graph_node_name: Optional[str] = None
    load: int = 0
    merged_children: Optional[List[Dict[str, Any]]] = None
    merged_parent_id: Optional[str] = None
    merge_slice: Optional[Dict[str, int]] = None
    merge_key: Optional[str] = None
