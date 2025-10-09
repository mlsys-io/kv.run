from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import time
from utils import now_iso

class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"
    DONE = "DONE"
    WAITING = "WAITING"

class TaskRecord(BaseModel):
    task_id: str
    raw_yaml: str
    parsed: Dict[str, Any]
    status: str = TaskStatus.PENDING
    assigned_worker: Optional[str] = None  # "MULTI" for sharded parent
    topic: Optional[str] = None
    submitted_at: str = Field(default_factory=now_iso)
    submitted_ts: float = Field(default_factory=time.time)
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
    slo_seconds: Optional[float] = None
    merged_children: Optional[List[Dict[str, Any]]] = None
    merged_parent_id: Optional[str] = None
    merge_slice: Optional[Dict[str, int]] = None
