"""
Task modules for MLOC.

This package contains all the task execution modules (wrappers) for different
types of LLM operations like SFT, PPO, RAG, and Agent tasks.
"""

from typing import Type, Optional

from mloc.common.constants import TaskType
from .base_module import BaseModule
from .sft_module import SFTModule
from .rm_module import RewardModelModule
from .ppo_module import PPOModule
from .rag_module import RAGModule
from .agent_module import AgentModule


# Module registry
MODULE_REGISTRY = {
    TaskType.SFT: SFTModule,
    TaskType.REWARD_MODEL: RewardModelModule,
    TaskType.PPO: PPOModule,
    TaskType.RAG_INFERENCE: RAGModule,
    TaskType.RAG_INDEXING: RAGModule,
    TaskType.AGENT_INFERENCE: AgentModule,
}


def get_module_class(task_type: TaskType) -> Optional[Type[BaseModule]]:
    """Get module class for a given task type"""
    return MODULE_REGISTRY.get(task_type)


def register_module(task_type: TaskType, module_class: Type[BaseModule]) -> None:
    """Register a new module class for a task type"""
    MODULE_REGISTRY[task_type] = module_class


__all__ = [
    "BaseModule",
    "SFTModule", 
    "RewardModelModule",
    "PPOModule",
    "RAGModule",
    "AgentModule",
    "get_module_class",
    "register_module",
]