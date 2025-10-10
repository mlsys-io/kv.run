from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from .base_executor import Executor

class EchoExecutor(Executor):
    name = "echo"

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        payload = {
            "ok": True,
            "echo": spec,
        }
        self.save_json(out_dir / "responses.json", payload)
        return payload
