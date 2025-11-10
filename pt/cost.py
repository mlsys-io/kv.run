#!/usr/bin/env python3
"""
FlowMesh Evaluation Plot Generator (CDP/EDP with in-figure legend, top-left)

- Compact horizontal layout (configurable)
- Okabe–Ito scientific color palette + distinct hatch patterns
- Dual y-axes with separate scales for both figures
- Legend drawn INSIDE each figure (top-left corner)
- Paper-ready font sizes (configurable)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import csv
import matplotlib.patches as mpatches


# =========================
# Global Matplotlib settings
# =========================
def configure_fonts(base=13, family="Times New Roman"):
    """Configure matplotlib rcParams for paper-ready figures."""
    plt.rcParams.update({
        "font.family": family,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base - 1,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2"]
HATCHES = ["", "///", "xx", ".."]


def plot_dual_grouped_bars(ax_left, ax_right, x_positions, left_values, right_values,
                           schedulers, colors, hatches, bar_width=0.16,
                           title="", left_label="", right_label=""):
    """Draw 2×4 grouped bars with dual y-axes."""
    x0, x1 = x_positions
    n_sched = len(schedulers)
    offsets = (np.arange(n_sched) - (n_sched - 1) / 2.0) * bar_width

    # Left group
    for i in range(n_sched):
        ax_left.bar(x0 + offsets[i], left_values[i],
                    width=bar_width, color=colors[i], hatch=hatches[i % len(hatches)],
                    edgecolor="black", linewidth=0.7, zorder=3)

    # Right group
    for i in range(n_sched):
        ax_right.bar(x1 + offsets[i], right_values[i],
                     width=bar_width, color=colors[i], hatch=hatches[i % len(hatches)],
                     edgecolor="black", linewidth=0.7, zorder=3)

    ax_left.set_xlim(x0 - 0.6, x1 + 0.6)
    ax_left.set_xticks([x0, x1])
    ax_left.set_ylabel(left_label)
    ax_right.set_ylabel(right_label)
    ax_left.set_title(title)
    ax_left.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)


def add_infigure_legend_top_left(ax, schedulers, colors, hatches):
    """Add legend inside the plot (top-left corner)."""
    patches = [
        mpatches.Patch(facecolor=colors[i], hatch=hatches[i % len(hatches)],
                       edgecolor="black", label=schedulers[i], linewidth=0.6)
        for i in range(len(schedulers))
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),  # slight inset from edges
        frameon=False,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.4,
    )


def main(save_dir="figures", width=6.4, height=4.3, dpi=400, font_size=13, font_family="Times New Roman"):
    configure_fonts(base=font_size, family=font_family)

    # ---- Data ----
    schedulers = ["FlowMesh", "MF", "DR", "DS"]
    cost = np.array([0.75, 1.32, 2.22, 2.84])
    energy = np.array([0.068, 0.086, 0.124, 0.138])
    avg_total_time = np.array([69.00, 65.15, 79.74, 382.06])
    submitted = np.array([84, 84, 84, 84])

    cost_per_sub = cost / submitted
    energy_per_sub = energy / submitted
    cdp = cost_per_sub * avg_total_time
    edp = energy_per_sub * avg_total_time

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = OKABE_ITO
    hatches = HATCHES
    x_positions = [0.0, 1.0]

    # ---------- Figure (a): Cost & Energy ----------
    fig_a, axL = plt.subplots(figsize=(width, height))
    axR = axL.twinx()
    plot_dual_grouped_bars(
        axL, axR, x_positions,
        cost, energy, schedulers,
        colors, hatches,
        title="Cost and Energy across Baselines",
        left_label="Cost ($)",
        right_label="Energy (kWh)",
    )
    axL.set_xticklabels(["Cost", "Energy"])
    add_infigure_legend_top_left(axL, schedulers, colors, hatches)
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "cost_a.pdf", dpi=dpi)
    plt.close(fig_a)

    # ---------- Figure (b): CDP & EDP ----------
    fig_b, bxL = plt.subplots(figsize=(width, height))
    bxR = bxL.twinx()
    plot_dual_grouped_bars(
        bxL, bxR, x_positions,
        cdp, edp, schedulers,
        colors, hatches,
        title="CDP and EDP across Baselines",
        left_label="Cost Delay Product",
        right_label="Energy Delay Product",
    )
    bxL.set_xticklabels(["CDP", "EDP"])
    add_infigure_legend_top_left(bxL, schedulers, colors, hatches)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "cost_b.pdf", dpi=dpi)
    plt.close(fig_b)

    # ---------- CSV export ----------
    with open(out_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Scheduler", "Cost", "Energy", "AvgTotalTime", "Submitted", "CDP", "EDP"])
        for i, s in enumerate(schedulers):
            writer.writerow([s, cost[i], energy[i], avg_total_time[i], int(submitted[i]), cdp[i], edp[i]])

    print(f"Saved inline-legend (top-left) figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowMesh CDP/EDP plot generator (in-figure legend, top-left)")
    parser.add_argument("--save-dir", default="figures")
    parser.add_argument("--width", type=float, default=6.2)
    parser.add_argument("--height", type=float, default=4.2)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--font-size", type=int, default=26)    
    parser.add_argument("--font-family", type=str, default="Times New Roman")
    args = parser.parse_args()
    main(save_dir=args.save_dir, width=args.width, height=args.height, dpi=args.dpi,
         font_size=args.font_size, font_family=args.font_family)
