#!/usr/bin/env python3
"""
FlowMesh Latency Line Plots (legend top-right; Set #2 has right-side Y-axis)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def configure_fonts(base=13, family="Times New Roman"):
    plt.rcParams.update({
        "font.family": family,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base - 1,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2"]


def add_infigure_legend_top_right(ax, labels, colors, lw=3.4, ms=9.5):
    handles = []
    for i, name in enumerate(labels):
        (ln,) = ax.plot([], [], color=colors[i], linewidth=lw, marker="o",
                        markersize=ms, label=name)
        handles.append(ln)
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.4,
    )


def plot_latency_lines(ax, x_vals, series_dict, labels, colors,
                       title, x_label, y_label,
                       lw=3.4, ms=9.5):
    for i, name in enumerate(labels):
        y = np.asarray(series_dict[name], dtype=float)
        ax.plot(x_vals, y, color=colors[i], linewidth=lw, marker="o",
                markersize=ms, label=name, zorder=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_title(title)


def main(save_dir="figures", width=6.2, height=4.2,
         dpi=400, font_size=22, font_family="Times New Roman"):
    configure_fonts(base=font_size, family=font_family)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_vals = np.array([3, 6, 12])
    schedulers = ["FlowMesh", "MF", "DR", "DS"]

    latency_set1 = {
        "FlowMesh": [176.55, 146.24, 103.52],
        "MF": [371.58, 236.78, 116.53],       # 原 Static
        "DR": [377.30, 217.83, 114.70],       # 原 Round-Robin
        "DS": [956.10, 462.46, 197.08],       # 原 Fix
    }

    latency_set2 = {
        "FlowMesh": [482.98, 285.87, 281.31],
        "MF": [523.58, 305.57, 284.08],       # 原 Static
        "DR": [616.02, 405.56, 359.79],       # 原 Round-Robin
        "DS": [594.78, 348.74, 292.85],       # 原 Fix
    }
    colors = OKABE_ITO

    # --- Figure (a): Left-axis ---
    fig_a, ax_a = plt.subplots(figsize=(width, height))
    plot_latency_lines(
        ax_a, x_vals, latency_set1, schedulers, colors,
        title="(a) Avg. Task Latency (Group A)",
        x_label="Available worker count",
        y_label="Avg. Task Latency (s)",
        lw=3.6, ms=10.0,
    )
    add_infigure_legend_top_right(ax_a, schedulers, colors, lw=3.6, ms=10.0)
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "latency_a.pdf", dpi=dpi)
    plt.close(fig_a)

    # --- Figure (b): Right-axis ---
    fig_b, ax_b_left = plt.subplots(figsize=(width, height))
    ax_b_right = ax_b_left.twinx()  # right y-axis

    plot_latency_lines(
        ax_b_right, x_vals, latency_set2, schedulers, colors,
        title="(b) Avg. Task Latency (Group B)",
        x_label=None,  # prevent setting label on the right axis
        y_label="Avg. Task Latency (s)",
        lw=3.6, ms=10.0,
    )

    # Move x-label and ticks to main (left) axis
    ax_b_left.set_xlabel("Available worker count")
    ax_b_left.set_xticks(x_vals)
    ax_b_left.set_xticklabels([str(x) for x in x_vals])

    # Hide left y-axis
    ax_b_left.set_yticks([])
    ax_b_left.set_ylabel("")

    add_infigure_legend_top_right(ax_b_right, schedulers, colors, lw=3.6, ms=10.0)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "latency_b.pdf", dpi=dpi)
    plt.close(fig_b)

    # --- CSV Export ---
    import csv
    with open(out_dir / "latency_set1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Workers", *x_vals])
        for s in schedulers:
            w.writerow([s, *latency_set1[s]])
    with open(out_dir / "latency_set2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Workers", *x_vals])
        for s in schedulers:
            w.writerow([s, *latency_set2[s]])

    print(f"Saved latency figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Latency line plots (legend top-right; Set #2 right-side y-axis)")
    p.add_argument("--save-dir", default="figures")
    p.add_argument("--width", type=float, default=6.2)
    p.add_argument("--height", type=float, default=4.2)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--font-size", type=int, default=26)
    p.add_argument("--font-family", type=str, default="Times New Roman")
    args = p.parse_args()
    main(save_dir=args.save_dir,
         width=args.width,
         height=args.height,
         dpi=args.dpi,
         font_size=args.font_size,
         font_family=args.font_family)
