#!/usr/bin/env python3
"""
FlowMesh Evaluation Plot Generator — Scale A/B (Left-Right Overlay Offset)

- 同一类目画两根“独立柱子”，在 x 方向左右轻微错开（overlay），而非上下错开/覆盖
- Group A: 左 y 轴；Group B: 右 y 轴
- 图例：Avg Queueing Time / Avg Total Latency（右上角内嵌）
- 输出：scale_a.pdf / scale_b.pdf / scale_latency_queueing_overlay.csv
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import csv
import matplotlib.patches as mpatches

def configure_fonts(base=13, family="Times New Roman"):
    plt.rcParams.update({
        "font.family": family,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base - 1,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })

# Okabe–Ito
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2"]
COLOR_TOTAL = OKABE_ITO[0]   # Avg Total Latency
COLOR_QUEUE = OKABE_ITO[1]   # Avg Queueing Time
HATCH_TOTAL = ""
HATCH_QUEUE = "///"

def add_legend_top_right(ax):
    patches = [
        mpatches.Patch(facecolor=COLOR_QUEUE, hatch=HATCH_QUEUE, edgecolor="black",
                       linewidth=0.6, label="Avg Queueing Time"),
        mpatches.Patch(facecolor=COLOR_TOTAL, hatch=HATCH_TOTAL, edgecolor="black",
                       linewidth=0.6, label="Avg Total Latency"),
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.4,
    )

def plot_overlay_lr_offset(
    x_vals, queue_vals, total_vals,
    title, x_label, y_label,
    width, height, dpi, save_path,
    y_on_right=False, bar_width=0.53, x_offset_frac=0.35, alpha_overlay=1.0
):
    """
    左右错开的 overlay：两根柱共享同一类目，但在 x 方向以 ±offset 轻微分离。
    - offset = x_offset_frac * bar_width (默认 0.35 倍柱宽)
    - 可选 alpha_overlay 控制透明度（默认不透明）。
    """
    fig, ax = plt.subplots(figsize=(width, height))
    if y_on_right:
        axR = ax.twinx()
        target_ax = axR
        ax.yaxis.set_visible(False)
    else:
        target_ax = ax

    x_positions = np.arange(len(x_vals))
    offset = x_offset_frac * bar_width

    # 左右偏移位置（保证视觉重叠但不完全遮挡）
    x_total = x_positions - offset/2.0
    x_queue = x_positions + offset/2.0

    # 先画 Total（略低 zorder），再画 Queue
    target_ax.bar(
        x_total, total_vals,
        width=bar_width,
        color=COLOR_TOTAL,
        hatch=HATCH_TOTAL,
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
        alpha=alpha_overlay,
        label="Avg Total Latency",
    )

    target_ax.bar(
        x_queue, queue_vals,
        width=bar_width,
        color=COLOR_QUEUE,
        hatch=HATCH_QUEUE,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        alpha=alpha_overlay,
        label="Avg Queueing Time",
    )

    # x 轴刻度依然对齐到类目中心
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_vals)

    target_ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    target_ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    add_legend_top_right(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

def main(save_dir="figures", width=6.2, height=4.2, dpi=400, font_size=18, font_family="Times New Roman"):
    configure_fonts(base=font_size, family=font_family)

    worker_nums = np.array([8, 16, 32, 48], dtype=int)

    # 数据
    total_a = np.array([451.08027727066815, 348.58129857322587, 336.8485259575148, 224.00204573321986])
    queue_a = np.array([390.1907398595915, 293.9809974964863, 279.21600220600766, 152.8827445635924])

    total_b = np.array([574.2444612314534, 489.7815601974726, 399.45521476605666, 387.9609693012587])
    queue_b = np.array([505.11330456927766, 406.5804646001621, 330.2485410784898, 277.7997229666245])

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scale A（左轴）
    plot_overlay_lr_offset(
        worker_nums,
        queue_vals=queue_a,
        total_vals=total_a,
        title="Task Group A",
        x_label="# Workers",
        y_label="Time (s)",
        width=width,
        height=height,
        dpi=dpi,
        save_path=out_dir / "scale_a.pdf",
        y_on_right=False,
        bar_width=0.45,
        x_offset_frac=0.4,    # 调整左右错开强度（0.25 ~ 0.45 常用）
        alpha_overlay=0.9,
    )

    # Scale B（右轴）
    plot_overlay_lr_offset(
        worker_nums,
        queue_vals=queue_b,
        total_vals=total_b,
        title="Task Group B",
        x_label="# Workers",
        y_label="Time (s)",
        width=width,
        height=height,
        dpi=dpi,
        save_path=out_dir / "scale_b.pdf",
        y_on_right=True,
        bar_width=0.45,
        x_offset_frac=0.4,
        alpha_overlay=0.9,
    )

    # CSV
    with open(out_dir / "scale_latency_queueing_overlay.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["AvailableWorkerNum", "AvgQueueing_A", "AvgTotal_A", "AvgQueueing_B", "AvgTotal_B"])
        for i in range(len(worker_nums)):
            w.writerow([int(worker_nums[i]), queue_a[i], total_a[i], queue_b[i], total_b[i]])

    print(f"Saved: { (out_dir / 'scale_a.pdf').resolve() }")
    print(f"Saved: { (out_dir / 'scale_b.pdf').resolve() }")
    print(f"CSV:   { (out_dir / 'scale_latency_queueing_overlay.csv').resolve() }")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowMesh left-right overlay offset (independent bars with slight horizontal shift)")
    parser.add_argument("--save-dir", default="figures", type=str)
    parser.add_argument("--width", type=float, default=6.2)
    parser.add_argument("--height", type=float, default=4.2)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--font-size", type=int, default=26)
    parser.add_argument("--font-family", type=str, default="Times New Roman")
    args = parser.parse_args()
    main(
        save_dir=args.save_dir,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        font_size=args.font_size,
        font_family=args.font_family,
    )
