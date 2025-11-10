#!/usr/bin/env python3
"""
Generate task submission rate vs. active worker plots from recorded metrics.
"""

from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def configure_fonts(base=13, family="Times New Roman"):
    """Configure matplotlib rcParams for paper-ready figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [family, "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot task submission rate and active worker trends.")
    parser.add_argument("--data", type=Path, default=Path("metrics/task_worker_density.json"))
    parser.add_argument("--output", type=Path, default=Path("pt/task_worker_density.pdf"))
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.0, 3.2))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--worker-smoothing-window", type=int, default=0)
    return parser.parse_args()


def load_density_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Density JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {path}: {type(payload)}")
    return payload


def _parse_iso_datetime(raw: str) -> datetime:
    value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    return datetime.fromisoformat(value)


def _extract_series(entries: List[Dict[str, Any]], time_key: str, value_key: str, *, cast=None) -> List[tuple[datetime, float]]:
    series: List[tuple[datetime, float]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_time = entry.get(time_key)
        if not isinstance(raw_time, str):
            continue
        try:
            ts = _parse_iso_datetime(raw_time)
        except ValueError:
            continue
        value = entry.get(value_key)
        if value is None:
            continue
        try:
            numeric = cast(value) if cast else float(value)
        except (TypeError, ValueError):
            continue
        series.append((ts, float(numeric)))
    series.sort(key=lambda item: item[0])
    return series


def _smooth_worker_series(series: List[tuple[datetime, float]], window: int) -> List[tuple[datetime, float]]:
    if window <= 1 or len(series) < 2:
        return series
    window = max(2, window)
    timestamps = [item[0] for item in series]
    values = [item[1] for item in series]
    smoothed: List[tuple[datetime, float]] = []
    for idx in range(len(values)):
        start = max(0, idx - window // 2)
        end = start + window
        if end > len(values):
            end = len(values)
            start = max(0, end - window)
        avg = sum(values[start:end]) / (end - start)
        smoothed.append((timestamps[idx], avg))
    return smoothed


def build_plot(payload: Dict[str, Any], output: Path, figsize: tuple[float, float], dpi: int, *, worker_smoothing_window: int = 0) -> None:
    raw_buckets = payload.get("task_density_buckets") or []
    raw_workers = payload.get("worker_count_series") or []
    bucket_seconds_raw = payload.get("bucket_seconds")
    try:
        bucket_seconds = float(bucket_seconds_raw)
    except (TypeError, ValueError):
        bucket_seconds = 60.0
    if bucket_seconds <= 0:
        bucket_seconds = 60.0

    task_series = _extract_series(raw_buckets, "bucket_start", "tasks_per_minute", cast=float)
    worker_series = _extract_series(raw_workers, "ts", "active_workers", cast=int)
    if task_series:
        offset = timedelta(seconds=bucket_seconds / 2.0)
        task_series = [(ts + offset, value) for ts, value in task_series]
    if worker_smoothing_window and worker_series:
        worker_series = _smooth_worker_series(worker_series, worker_smoothing_window)

    if not task_series and not worker_series:
        raise ValueError("Density payload contains no data to plot.")

    if task_series and worker_series:
        final_ts = max(task_series[-1][0], worker_series[-1][0])
        if task_series[-1][0] < final_ts:
            task_series.append((final_ts, task_series[-1][1]))
        if worker_series[-1][0] < final_ts:
            worker_series.append((final_ts, worker_series[-1][1]))

    origins = [s[0] for s in (task_series + worker_series)] if (task_series or worker_series) else []
    origin = min(origins)

    task_elapsed = [(ts - origin).total_seconds() for ts, _ in task_series]
    task_values = [v for _, v in task_series]

    worker_elapsed = [(ts - origin).total_seconds() for ts, _ in worker_series]
    worker_values = [v for _, v in worker_series]

    fig, ax_left = plt.subplots(figsize=figsize)
    lines = []

    # 左轴：任务提交速率
    line_tasks, = ax_left.plot(
        task_elapsed,
        task_values,
        color="tab:blue",
        linewidth=2.6,
        marker="o",
        label="task_submission_rate",
        zorder=5,
    )
    lines.append(line_tasks)
    ax_left.set_ylabel("Task submission rate\n(tasks/min)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")

    ax_left.set_xlabel("Elapsed time (s)")
    ax_left.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax_left.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
    ax_left.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both", nbins=6))

    # 右轴：活跃 worker 数
    ax_right = ax_left.twinx()
    if worker_smoothing_window and worker_smoothing_window > 1:
        (worker_line,) = ax_right.plot(
            worker_elapsed,
            worker_values,
            color="tab:orange",
            linewidth=2.6,
            label="active_worker_count",
            zorder=3,
        )
    else:
        (worker_line,) = ax_right.step(
            worker_elapsed,
            worker_values,
            where="post",
            color="tab:orange",
            linewidth=2.6,
            label="active_worker_count",
            zorder=3,
        )
    lines.append(worker_line)
    ax_right.set_ylabel("# Active workers", color="tab:orange")
    ax_right.tick_params(axis="y", labelcolor="tab:orange")
    ax_right.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    if lines:
        legend = ax_left.legend(
            [line_tasks, worker_line],
            ["Task submission rate", "Active worker count"],
            loc="upper right",
            borderaxespad=0.6,
        )
        legend.set_zorder(10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    configure_fonts(base=24, family="Times New Roman")
    args = parse_args()
    payload = load_density_payload(args.data)
    build_plot(
        payload,
        args.output,
        tuple(args.figsize),
        args.dpi,
        worker_smoothing_window=max(0, args.worker_smoothing_window),
    )
    print(f"Saved density figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
