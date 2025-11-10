#!/usr/bin/env python3
"""
FlowMesh Throughput Line Plots
Left: Group A (Y-axis left)
Right: Group B (Y-axis right)
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
        "axes.linewidth": 1.1,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2"]


def plot_single_series(ax, x_vals, y_vals, color,
                       title, x_label, y_label, lw=3.6, ms=8.5,
                       yaxis_right=False):
    """Draw a single line with optional right-side Y axis (no legend)."""
    y = np.asarray(y_vals, dtype=float)
    ax.plot(
        x_vals, y,
        color=color,
        linewidth=lw,
        marker="o",
        markersize=ms,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_title(title)

    if yaxis_right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()


def main(save_dir="figures", width=6.2, height=4.2,
         dpi=400, font_size=26, font_family="Times New Roman"):
    configure_fonts(base=font_size, family=font_family)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_vals = np.array([8, 16, 32, 48], dtype=int)
    y_a = [16.7273, 27.4286, 29.3077, 32.0]
    y_b = [12.5714, 19.0526, 22.0, 26.5882]

    # --- Figure 1: Task Group A (left Y-axis) ---
    fig_a, ax_a = plt.subplots(figsize=(width, height))
    plot_single_series(
        ax_a, x_vals, y_a,
        color=OKABE_ITO[0],
        title="Task Group A",
        x_label="# Workers",
        y_label="Complete Tasks / min",
        lw=3.6, ms=8.5,
    )
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "throughput_groupA.pdf", dpi=dpi)
    plt.close(fig_a)

    # --- Figure 2: Task Group B (right Y-axis) ---
    fig_b, ax_b = plt.subplots(figsize=(width, height))
    plot_single_series(
        ax_b, x_vals, y_b,
        color=OKABE_ITO[1],
        title="Task Group B",
        x_label="# Workers",
        y_label="Complete Tasks / min",
        lw=3.6, ms=8.5,
        yaxis_right=True,
    )
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "throughput_groupB.pdf", dpi=dpi)
    plt.close(fig_b)

    # --- CSV Export ---
    import csv
    with open(out_dir / "throughput_groupA.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# Workers", *x_vals])
        w.writerow(["Complete Workflows / min", *y_a])

    with open(out_dir / "throughput_groupB.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# Workers", *x_vals])
        w.writerow(["Complete Workflows / min", *y_b])

    print(f"Saved throughput figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Throughput line plots (right Y-axis for Group B)")
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
