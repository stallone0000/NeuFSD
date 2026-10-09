#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ALGORITHMS = ["mrac", "elastic", "sample", "davinci", "neufsd"]
LABELS = {
    "mrac": "MRAC",
    "elastic": "Elastic",
    "sample": "Sample",
    "davinci": "DaVinci",
    "neufsd": "NeuFSD",
}
COLORS = {
    "mrac": "C1",
    "elastic": "C0",
    "sample": "C2",
    "davinci": "C5",
    "neufsd": "C3",
}
MEM_KB = [16, 32, 64, 128, 256]

FIGSIZE = (12.0, 4.0)
TICK_SIZE = 24
LABEL_SIZE = 30
LINE_WIDTH = 4.8
BAR_WIDTH = 0.15
BAR_EDGE_WIDTH = 2.2
ERROR_LINE_WIDTH = 2.7


def style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def bold_ticks(ax) -> None:
    ax.tick_params(labelsize=TICK_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def grouped_barplot(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    *,
    right_axis: bool = False,
    gbps_factor: float = 0.672,
    y_scale_note: str | None = None,
    y_unit_note: str | None = None,
    right_unit_note: str | None = None,
    y_max: float | None = None,
    right_y_max: float | None = None,
) -> None:
    centers = np.arange(len(MEM_KB), dtype=float)
    offsets = np.linspace(-0.34, 0.34, len(ALGORITHMS))
    all_values: list[float] = []

    for alg, offset in zip(ALGORITHMS, offsets):
        medians = []
        lower_err = []
        upper_err = []
        positions = []
        for i, kb in enumerate(MEM_KB):
            vals = pd.to_numeric(
                df[(df["algorithm"] == alg) & (df["memory_kb"] == kb) & (df["status"] == "ok")][value_col],
                errors="coerce",
            ).dropna()
            arr = vals.to_numpy(dtype=float)
            positions.append(centers[i] + offset)
            all_values.extend(arr.tolist())
            if len(arr):
                median = float(np.median(arr))
                medians.append(median)
                lower_err.append(median - float(np.min(arr)))
                upper_err.append(float(np.max(arr)) - median)
            else:
                medians.append(np.nan)
                lower_err.append(0.0)
                upper_err.append(0.0)

        ax.bar(
            positions,
            medians,
            width=BAR_WIDTH,
            color=COLORS[alg],
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            alpha=0.72,
            yerr=np.vstack([lower_err, upper_err]),
            error_kw={
                "elinewidth": ERROR_LINE_WIDTH,
                "ecolor": "black",
                "capsize": 5,
                "capthick": ERROR_LINE_WIDTH,
            },
            zorder=5,
        )

    ax.set_xlim(-0.62, len(MEM_KB) - 0.38)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(k) for k in MEM_KB])
    ax.set_xlabel("Memory (KB)", fontweight="bold", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle="--", axis="both", alpha=0.55)
    bold_ticks(ax)
    if y_scale_note:
        ax.text(
            0.02,
            0.98,
            y_scale_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=TICK_SIZE,
        )
    if y_unit_note:
        ax.text(
            0.02,
            0.98,
            y_unit_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=TICK_SIZE,
        )

    vals = np.asarray([v for v in all_values if np.isfinite(v)], dtype=float)
    if len(vals):
        hi = float(vals.max())
        ax.set_ylim(0.0, hi * 1.16)
    if y_max is not None:
        ax.set_ylim(0.0, y_max)

    if right_axis:
        ax2 = ax.twinx()
        if right_y_max is not None:
            _, ymax = ax.get_ylim()
            ax.set_ylim(ax.get_ylim()[0], right_y_max / gbps_factor)
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * gbps_factor, ymax * gbps_factor)
        ax2.set_ylabel("Throughput", fontweight="bold", fontsize=LABEL_SIZE)
        ax2.tick_params(labelsize=TICK_SIZE, direction="in")
        for tick in ax2.get_yticklabels():
            tick.set_fontweight("bold")
        if right_unit_note:
            ax2.text(
                0.98,
                0.98,
                right_unit_note,
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
                fontsize=TICK_SIZE,
            )


def draw_legend(out_dir: Path) -> None:
    handles = [
        Patch(facecolor=COLORS[alg], edgecolor="black", linewidth=1.5, alpha=0.72, label=LABELS[alg])
        for alg in ALGORITHMS
    ]
    fig, ax = plt.subplots(figsize=(8.0, 0.72))
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", ncol=5, frameon=False, handlelength=2.8, columnspacing=1.4)
    plt.setp(leg.get_texts(), fontweight="bold", fontsize=16)
    fig.savefig(out_dir / "legend_redis_bess.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    redis = pd.read_csv(args.input_dir / "redis_insert_throughput_repeats.csv")
    bess = pd.read_csv(args.input_dir / "bess_packet_rate_repeats.csv")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    grouped_barplot(ax, redis, "throughput_mops", "Events/s", y_scale_note=r"$\times 10^6$", y_max=1.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "redis_events_memory_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)

    packet_size = float(bess["packet_size_bytes"].dropna().iloc[0])
    overhead = float(bess["l1_overhead_bytes"].dropna().iloc[0])
    gbps_factor = (packet_size + overhead) * 8.0 / 1000.0
    fig, ax = plt.subplots(figsize=FIGSIZE)
    grouped_barplot(
        ax,
        bess,
        "throughput_mpps",
        "Packet Rate",
        right_axis=True,
        gbps_factor=gbps_factor,
        y_unit_note="Mpps",
        right_unit_note="Gbps",
        right_y_max=2.1,
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "bess_packet_rate_memory_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)

    draw_legend(args.out_dir)

    summary = []
    for platform, df, col in [("redis", redis, "throughput_mops"), ("bess", bess, "throughput_mpps")]:
        for alg in ALGORITHMS:
            for kb in MEM_KB:
                vals = pd.to_numeric(
                    df[(df["algorithm"] == alg) & (df["memory_kb"] == kb) & (df["status"] == "ok")][col],
                    errors="coerce",
                ).dropna()
                if len(vals):
                    summary.append(
                        {
                            "platform": platform,
                            "algorithm": alg,
                            "memory_kb": kb,
                            "min": float(vals.min()),
                            "median": float(vals.median()),
                            "max": float(vals.max()),
                        }
                    )
    pd.DataFrame(summary).to_csv(args.out_dir / "redis_bess_plot_summary.csv", index=False)


if __name__ == "__main__":
    main()
