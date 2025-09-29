#!/usr/bin/env python3
"""
Distributed SFT entrypoint launched via torchrun by SFTExecutor.
Usage:
  torchrun --nproc_per_node N -m executors.sft_dist_entry <task_json> <out_dir>

It loads the serialized task spec and delegates to SFTExecutor.run.
"""

import json
import sys
from pathlib import Path

from .sft_executor import SFTExecutor


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: -m executors.sft_dist_entry <task_json> <out_dir>")
        return 2
    task_path = Path(argv[1])
    out_dir = Path(argv[2])
    with task_path.open("r") as fh:
        task = json.load(fh)
    ex = SFTExecutor()
    ex.run(task, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

