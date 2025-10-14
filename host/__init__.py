"""
Minimal host-side orchestrator package.

Provides a simplified orchestration runtime tailored for first-come-first-serve
task scheduling while reusing the existing worker protocols and metrics stack.
"""

from .task_runtime import TaskRuntime

__all__ = ["TaskRuntime"]
