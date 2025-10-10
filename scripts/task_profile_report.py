#!/usr/bin/env python3
"""Generate a Markdown report summarising task outcomes."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def build_report(rows: list[dict[str, str]]) -> str:
    total = len(rows)
    status_counter = Counter(row.get("status", "UNKNOWN") for row in rows)
    type_counter = Counter(row.get("task_type") or "unknown" for row in rows)

    durations: list[float] = []
    for row in rows:
        submitted = parse_timestamp(row.get("submitted_ts"))
        generated = parse_timestamp(row.get("manifest_generated_at"))
        if submitted and generated:
            durations.append((generated - submitted).total_seconds())

    duration_summary = "N/A"
    if durations:
        duration_summary = f"平均 {statistics.mean(durations):.2f}s，中位 {statistics.median(durations):.2f}s"

    success_by_executor = defaultdict(lambda: Counter())
    for row in rows:
        executor = row.get("executor") or row.get("task_type") or "unknown"
        success_by_executor[executor][row.get("status", "UNKNOWN")] += 1

    lines = ["# Task Profile Report", ""]
    lines.append(f"- 总任务数：{total}")
    lines.append("- 按状态统计：" + ", ".join(f"{k}={v}" for k, v in status_counter.items()))
    lines.append("- 平均任务时长：" + duration_summary)
    lines.append("- 按 taskType 分布：" + ", ".join(f"{k}={v}" for k, v in type_counter.items()))
    lines.append("")
    lines.append("## 按执行器成功率")
    lines.append("")
    lines.append("| Executor | SUCCESS | FAILED | UNKNOWN |")
    lines.append("| --- | ---: | ---: | ---: |")
    for executor, counter in sorted(success_by_executor.items()):
        lines.append(
            f"| {executor} | {counter.get('DONE', 0)} | {counter.get('FAILED', 0)} | {counter.get('UNKNOWN', 0)} |"
        )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Markdown report from exported results")
    parser.add_argument("--input", type=Path, default=Path("./results_export.csv"), help="CSV produced by export_results.py")
    parser.add_argument("--output", type=Path, default=Path("./task_profile_report.md"), help="Markdown output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = load_rows(args.input)
    report = build_report(rows)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
