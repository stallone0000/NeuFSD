#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = [
    ("scratch_independent", "scratch independent"),
    ("scratch_continuous", "scratch continuous"),
    ("pretrain_independent", "pretrain independent"),
    ("pretrain_continuous", "pretrain continuous"),
]

RESOLUTIONS = ["128_128", "64_64"]
TRACES = ["caida_2016", "caida_2018", "caida_2018_new"]

MODE_STYLES = {
    "scratch_independent": "-",
    "scratch_continuous": "--",
    "pretrain_independent": "-.",
    "pretrain_continuous": ":",
}

MRD_COLORS = {
    "scratch_independent": "#0B5FA5",
    "scratch_continuous": "#1C8DAB",
    "pretrain_independent": "#2AA876",
    "pretrain_continuous": "#006D77",
}

WMRD_COLORS = {
    "scratch_independent": "#C7362F",
    "scratch_continuous": "#E97822",
    "pretrain_independent": "#B15D12",
    "pretrain_continuous": "#8F2D56",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MRD and WMRD minute curves for four MRAC modes."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_metric_rows(path: Path) -> tuple[list[int], list[float], list[float]]:
    minutes: list[int] = []
    mrds: list[float] = []
    wmrds: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            minutes.append(int(row["minute"]))
            mrds.append(float(row["mrd_avg"]))
            wmrds.append(float(row["wmrd_avg"]))
    if not minutes:
        raise RuntimeError(f"empty minute metrics: {path}")
    return minutes, mrds, wmrds


def plot_one(run_root: Path, out_dir: Path, res: str, trace: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)

    plotted = 0
    for mode, label in MODES:
        metrics_path = (
            run_root
            / "online"
            / mode
            / f"{res}_{trace}_final"
            / "plots"
            / "pipeline_eval"
            / "minute_metrics.csv"
        )
        minutes, mrds, wmrds = read_metric_rows(metrics_path)
        style = MODE_STYLES[mode]
        ax.plot(
            minutes,
            mrds,
            linestyle=style,
            linewidth=1.9,
            color=MRD_COLORS[mode],
            label=f"MRD - {label}",
        )
        ax.plot(
            minutes,
            wmrds,
            linestyle=style,
            linewidth=1.9,
            color=WMRD_COLORS[mode],
            label=f"WMRD - {label}",
        )
        plotted += 2

    ax.set_title(f"{res} {trace}: MRD and WMRD over time")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Error")
    ax.grid(True, axis="both", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=8.7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.margins(x=0.01)

    if plotted != 8:
        raise RuntimeError(f"expected 8 lines for {res} {trace}, got {plotted}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"error_8lines_{res}_{trace}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "plots" / "error_8line_curves")
    outputs = []
    for res in RESOLUTIONS:
        for trace in TRACES:
            outputs.append(plot_one(args.run_root, out_dir, res, trace, args.dpi))
    print(f"wrote {len(outputs)} figures to {out_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
