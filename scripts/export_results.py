#!/usr/bin/env python3
"""Aggregate responses.json artifacts into a single CSV for analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def load_state_map(state_path: Path | None) -> dict[str, dict[str, Any]]:
    if not state_path:
        return {}
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tasks = state.get("tasks", {})
    return tasks


def extract_rows(results_dir: Path, state_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in results_dir.glob("**/manifest.json"):
        task_dir = manifest.parent
        task_id = task_dir.name
        responses = task_dir / "responses.json"
        data: Dict[str, Any] = {}
        if responses.exists():
            data = json.loads(responses.read_text(encoding="utf-8"))

        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        state_entry = state_map.get(task_id, {})
        parsed = state_entry.get("parsed", {})
        spec = parsed.get("spec", {})

        row = {
            "task_id": task_id,
            "task_type": spec.get("taskType"),
            "status": state_entry.get("status", "UNKNOWN"),
            "submitted_ts": state_entry.get("submitted_ts"),
            "manifest_generated_at": manifest_payload.get("generated_at"),
            "executor": data.get("executor") or data.get("result", {}).get("executor"),
            "result_path": str(responses) if responses.exists() else "",
            "manifest_path": str(manifest),
            "result_json": json.dumps(data, ensure_ascii=False),
        }
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export results manifests into a CSV")
    parser.add_argument("--results-dir", type=Path, default=Path("./results_host"), help="Directory storing task results")
    parser.add_argument("--state-file", type=Path, default=None, help="Optional task_state.json to enrich metadata")
    parser.add_argument("--output", type=Path, default=Path("./results_export.csv"), help="Output CSV path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.results_dir.exists():
        parser.error(f"Results directory not found: {args.results_dir}")

    state_map = load_state_map(args.state_file)
    rows = extract_rows(args.results_dir, state_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task_id",
                "task_type",
                "status",
                "submitted_ts",
                "manifest_generated_at",
                "executor",
                "result_path",
                "manifest_path",
                "result_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
