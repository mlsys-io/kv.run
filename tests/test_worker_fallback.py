from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

redis_stub = types.ModuleType("redis")


class _FakeRedis:
    def ping(self):
        return True


redis_stub.from_url = lambda *args, **kwargs: _FakeRedis()

sys.modules.setdefault("redis", redis_stub)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.main import initialize_executors  # type: ignore  # noqa: E402


class _Dummy:
    def __init__(self, name: str):
        self.name = name


def test_gpu_to_cpu_fallback(monkeypatch):
    registry = {
        "default": lambda: _Dummy("hf"),
        "vllm": lambda: _Dummy("vllm"),
    }

    executors, default_executor = initialize_executors(
        logging.getLogger("test"),
        registry=registry,
        import_errors={},
        cuda_available=False,
    )

    assert default_executor.name == "hf"
    assert "default" in executors
    assert "vllm" not in executors, "GPU 不可用时 vllm 应被跳过"


def test_initialize_keeps_optional(monkeypatch):
    registry = {
        "default": lambda: _Dummy("hf"),
        "echo": lambda: _Dummy("echo"),
    }

    executors, default_executor = initialize_executors(
        logging.getLogger("test"),
        registry=registry,
        import_errors={},
        cuda_available=True,
    )

    assert default_executor.name == "hf"
    assert executors["echo"].name == "echo"
