#!/usr/bin/env python3
"""Distributed worker used to execute SFT tasks launched via torchrun or DeepSpeed.

The ``SFTExecutor`` persists task state to disk, and this module rehydrates the
task inside each distributed rank. DeepSpeed injects ``--local_rank`` flags when
spawning processes, so this entrypoint accepts and ignores that flag while
forwarding the remaining positional arguments to ``SFTExecutor``.
"""

import json
import sys
from pathlib import Path
import argparse

from .sft_executor import SFTExecutor


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Distributed SFT worker entrypoint")
    parser.add_argument("task_json", type=Path, help="Path to serialized task specification")
    parser.add_argument("out_dir", type=Path, help="Output directory for training artifacts")
    parser.add_argument("--local_rank", type=int, default=None, help="Rank injected by torchrun/DeepSpeed")
    args = parser.parse_args(argv[1:])

    task_path = args.task_json
    out_dir = args.out_dir
    with task_path.open("r") as fh:
        task = json.load(fh)
    ex = SFTExecutor()
    try:
        result = ex.run(task, out_dir)
        # Ensure a responses.json is present for the parent to consume
        try:
            (out_dir / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception:
            pass
    finally:
        try:
            ex.cleanup_after_run()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
