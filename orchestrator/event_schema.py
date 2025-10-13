from __future__ import annotations

"""Event schema definitions scoped to the orchestrator service."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Event type, expected to be an uppercase enum value.")
    ts: str = Field(default_factory=_now_iso, description="Event timestamp (ISO8601).")
    worker_id: Optional[str] = Field(default=None, description="Associated worker identifier.")

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("event type must not be empty")
        return value


class TaskEvent(BaseEvent):
    task_id: str = Field(..., description="Associated task identifier.")
    status: Optional[str] = Field(default=None, description="Task status.")
    error: Optional[str] = Field(default=None, description="Error message if any.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Additional event payload.")

    @field_validator("task_id")
    @classmethod
    def _trim_task_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("task_id must not be empty")
        return value


class WorkerEvent(BaseEvent):
    status: Optional[str] = Field(default=None, description="Worker status (IDLE/RUNNING/etc).")
    tags: Optional[list[str]] = Field(default=None, description="Worker tags.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Metrics reported in heartbeat.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Additional context.")


Event = TaskEvent | WorkerEvent


def parse_event(data: Dict[str, Any]) -> Event:
    event_type = str(data.get("type", "")).upper()
    if event_type.startswith("TASK"):
        return TaskEvent.model_validate(data)
    return WorkerEvent.model_validate(data)


def serialize_event(event: Event) -> Dict[str, Any]:
    return event.model_dump(mode="python")
