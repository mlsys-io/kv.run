#!/usr/bin/env python3
"""Replay a recorded task from the orchestrator state snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_STATE_DIR = Path.cwd() / "data" / "orchestrator" / "state"


def load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def submit_task(orchestrator_url: str, token: str | None, raw_yaml: str) -> dict:
    headers = {"Content-Type": "text/yaml"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(f"{orchestrator_url.rstrip('/')}/api/v1/tasks", headers=headers, data=raw_yaml.encode("utf-8"), timeout=60)
    resp.raise_for_status()
    return resp.json()


def resolve_state_path(state_dir: Path | None, state_file: Path | None) -> Path:
    if state_file:
        return state_file
    if state_dir:
        return state_dir / "task_state.json"
    return Path("./state/task_state.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a task from orchestrator snapshots")
    parser.add_argument("task_id", help="The task ID to replay")
    parser.add_argument("--orchestrator-url", default="http://127.0.0.1:8000", help="Orchestrator base URL")
    parser.add_argument("--token", default=None, help="Optional bearer token")
    parser.add_argument("--state-dir", type=Path, default=None, help="Directory containing task_state.json")
    parser.add_argument("--state-file", type=Path, default=None, help="Explicit snapshot file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state_path = resolve_state_path(args.state_dir, args.state_file)
    snapshot = load_state(state_path)
    tasks = snapshot.get("tasks", {})
    record = tasks.get(args.task_id)
    if not record:
        parser.error(f"Task {args.task_id} not found in {state_path}")

    raw_yaml = record.get("raw_yaml")
    if not raw_yaml:
        parser.error("Snapshot does not contain raw_yaml for this task")

    submission = submit_task(args.orchestrator_url, args.token, raw_yaml)
    new_task_id = submission["tasks"][0]["task_id"]
    print(f"Replayed task {args.task_id} -> new task_id {new_task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
