from __future__ import annotations

"""
Executor base class and a minimal example implementation.

Usage:
    from executor_base import Executor, ExecutionError, EchoExecutor

    class MyExecutor(Executor):
        name = "my-executor"
        def run(self, task: dict, out_dir: Path) -> dict:
            # ... your logic ...
            result = {"ok": True, "echo": task}
            self.save_json(out_dir / "responses.json", result)
            return result

Contract:
- Implement `run(task: dict, out_dir: Path) -> dict`.
- Optionally override `prepare()` and `teardown()` for lifecycle hooks.
- Use `save_json()` to persist structured outputs.
- Raise `ExecutionError` for user-visible failures.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ExecutionError(RuntimeError):
    """Raised when an executor fails in an expected / controlled way."""


class Executor(ABC):
    """Abstract task executor.

    Subclasses must implement `run` and may override `prepare` and `teardown`.
    """

    #: Human-readable identifier for logging/telemetry
    name: str = "executor"

    def prepare(self) -> None:
        """Optional: called once before the first `run`.
        Use for lazy initialization (e.g., loading models, warming caches).
        """
        return None

    @abstractmethod
    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        """Execute a single task.

        Args:
            task: Parsed task payload (dict).
            out_dir: Directory for any outputs. Implementations should create it if needed.

        Returns:
            A JSON-serializable dictionary summarizing the result.

        Raises:
            ExecutionError: for expected, user-facing failures.
            Exception: for unexpected errors (will be logged by the caller).
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional: called when the worker is shutting down."""
        return None

    # ---------- Convenience helpers ----------
    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_json(path: Path, data: Dict[str, Any]) -> None:
        Executor.ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# -------- Minimal example implementation --------
class EchoExecutor(Executor):
    name = "echo"

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        # Example: just echo back inputs and write a file
        result = {
            "ok": True,
            "executor": self.name,
            "task": task,
        }
        self.save_json(out_dir / "responses.json", result)
        return result
