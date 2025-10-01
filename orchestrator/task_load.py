"""Helpers for classifying task load requirements based on YAML specs."""

from __future__ import annotations

from typing import Any, Dict

from utils import safe_get

# Ordered from highest to lowest demand.
TASK_LOAD_RANK: Dict[str, int] = {
    "sft": 5,
    "ppo": 5,
    "dpo": 5,
    "ppo_training": 5,
    "dpo_training": 5,
    "lora_sft": 4,
    "lora": 4,
    "rlhf": 4,
    "agent": 3,
    "rag": 3,
    "inference": 2,
}

DEFAULT_TASK_LOAD = 1
HEAVY_LOAD_THRESHOLD = TASK_LOAD_RANK["lora_sft"]
MEDIUM_LOAD_THRESHOLD = TASK_LOAD_RANK["agent"]

_KIND_FALLBACKS = {
    "lorasfttask": "lora_sft",
    "sfttask": "sft",
    "ppotrainingtask": "ppo",
    "dpotrainingtask": "dpo",
    "agenttask": "agent",
    "ragtask": "rag",
}


def _extract_task_type(spec: Dict[str, Any]) -> str:
    raw = safe_get(spec, "spec.taskType") or spec.get("taskType")
    if raw:
        return str(raw).strip().lower()

    kind = str(spec.get("kind", "")).strip().lower()
    if kind:
        for key, val in _KIND_FALLBACKS.items():
            if key in kind:
                return val

    return "inference"


def compute_task_load(task_spec: Dict[str, Any]) -> int:
    task_type = _extract_task_type(task_spec)
    return TASK_LOAD_RANK.get(task_type, DEFAULT_TASK_LOAD)

