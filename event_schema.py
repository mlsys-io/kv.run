from __future__ import annotations

"""Redis 事件统一 schema，供 orchestrator 与 worker 共用。"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="事件类型，需为大写枚举值。")
    ts: str = Field(default_factory=_now_iso, description="事件时间戳（ISO8601）。")
    worker_id: Optional[str] = Field(default=None, description="关联的 worker_id。")

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("event type must not be empty")
        return value


class TaskEvent(BaseEvent):
    task_id: str = Field(..., description="关联的任务 ID。")
    status: Optional[str] = Field(default=None, description="任务状态。")
    error: Optional[str] = Field(default=None, description="错误信息（若有）。")
    payload: Dict[str, Any] = Field(default_factory=dict, description="附加数据。")

    @field_validator("task_id")
    @classmethod
    def _trim_task_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("task_id must not be empty")
        return value


class WorkerEvent(BaseEvent):
    status: Optional[str] = Field(default=None, description="Worker 状态（IDLE/RUNNING 等）。")
    tags: Optional[list[str]] = Field(default=None, description="Worker 标签。")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="心跳上报的指标。")
    payload: Dict[str, Any] = Field(default_factory=dict, description="额外上下文信息。")


Event = TaskEvent | WorkerEvent


def parse_event(data: Dict[str, Any]) -> Event:
    event_type = str(data.get("type", "")).upper()
    if event_type.startswith("TASK"):
        return TaskEvent.model_validate(data)
    return WorkerEvent.model_validate(data)


def serialize_event(event: Event) -> Dict[str, Any]:
    return event.model_dump(mode="python")
