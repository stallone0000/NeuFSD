#!/usr/bin/env python3
"""Plot per-minute MRD/WMRD from pipeline summary_metrics.csv files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


DATASET_RE = re.compile(r"dataset_(\d+)$")


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset_id"] != "OVERALL_AVG"].copy()
    minutes = []
    for dataset_id in df["dataset_id"]:
        match = DATASET_RE.match(str(dataset_id))
        if not match:
            raise ValueError(f"Unexpected dataset_id format in {path}: {dataset_id}")
        minutes.append(int(match.group(1)))
    df.insert(0, "minute", minutes)
    return df.sort_values("minute").reset_index(drop=True)


def default_summary(root: Path, res: str, trace: str) -> Path:
    run_summary = root / "run_full_matrix" / f"{res}_{trace}_final" / "plots" / "pipeline_eval" / "summary_metrics.csv"
    if run_summary.exists():
        return run_summary
    result_summary = root / "results" / f"{res}_{trace}" / "summary_metrics.csv"
    if result_summary.exists():
        return result_summary
    raise FileNotFoundError(f"No summary_metrics.csv found for {res}_{trace}")


def plot_trace(trace: str, series: list[tuple[str, pd.DataFrame]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows = []
    for label, df in series:
        tmp = df.copy()
        tmp.insert(0, "series", label)
        merged_rows.append(tmp)
    pd.concat(merged_rows, ignore_index=True).to_csv(out_dir / f"{trace}_minute_error.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    colors = {"128_128": "#1f77b4", "64_64": "#d62728"}
    markers = {"128_128": "o", "64_64": "s"}

    for label, df in series:
        color = colors.get(label)
        marker = markers.get(label, "o")
        axes[0].plot(
            df["minute"],
            df["mrd_avg"],
            label=label,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.7,
            markerfacecolor="white",
        )
        axes[1].plot(
            df["minute"],
            df["wmrd_avg"],
            label=label,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.7,
            markerfacecolor="white",
        )

    axes[0].set_ylabel("MRD", fontweight="bold")
    axes[1].set_ylabel("WMRD", fontweight="bold")
    axes[1].set_xlabel("Minute", fontweight="bold")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False, ncol=len(series), loc="upper right")
    fig.suptitle(f"{trace}: per-minute error", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"{trace}_minute_error.png", dpi=220)
    fig.savefig(out_dir / f"{trace}_minute_error.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("zipf_fsd_release"))
    parser.add_argument("--out-dir", type=Path, default=Path("zipf_fsd_release/reproduced_figures/minute_error"))
    parser.add_argument("--traces", nargs="+", default=["caida_2016", "caida_2018"])
    parser.add_argument("--res", nargs="+", default=["128_128", "64_64"])
    args = parser.parse_args()

    for trace in args.traces:
        series = []
        for res in args.res:
            summary_path = default_summary(args.root, res, trace)
            series.append((res, load_summary(summary_path)))
            print(f"{trace}/{res}: {summary_path}")
        plot_trace(trace, series, args.out_dir)
        print(f"wrote {args.out_dir / f'{trace}_minute_error.png'}")


if __name__ == "__main__":
    main()
