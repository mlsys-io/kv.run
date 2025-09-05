"""
MLOC (Modular LLM Operations Container)

A unified containerized framework for LLM training, fine-tuning, 
inference and applications (RAG, Agent).
"""

__version__ = "0.1.0"
__author__ = "MLOC Team"

from mloc.common.constants import NodeType, TaskType, TaskStatus

__all__ = [
    "__version__",
    "__author__",
    "NodeType",
    "TaskType", 
    "TaskStatus",
]