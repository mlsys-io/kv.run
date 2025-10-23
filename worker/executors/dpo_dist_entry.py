#!/usr/bin/env python3
"""Distributed entrypoint for DPOExecutor launched via torchrun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dpo_executor import DPOExecutor


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Distributed DPO worker entrypoint")
    parser.add_argument("task_json", type=Path, help="Path to serialized task specification")
    parser.add_argument("out_dir", type=Path, help="Output directory for training artifacts")
    parser.add_argument("--local_rank", type=int, default=None, help="Rank injected by torchrun")
    args = parser.parse_args(argv[1:])

    with args.task_json.open("r", encoding="utf-8") as fh:
        task = json.load(fh)
    executor = DPOExecutor()
    try:
        result = executor.run(task, args.out_dir)
        if args.local_rank in (None, 0):
            try:
                (args.out_dir / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
    finally:
        try:
            executor.cleanup_after_run()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
