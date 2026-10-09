#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACE_DIRS = {
    "caida_2016": "caida_2016_fixed_front5_e20",
    "caida_2018": "caida_2018_fixed_front5_e20",
    "imc": "imc_fixed_front5_e20",
    "mawi": "mawi_fixed_front5_e20",
}
TRACE_LABELS = {
    "caida_2016": "CAIDA2016",
    "caida_2018": "CAIDA2018",
    "imc": "IMC",
    "mawi": "MAWI",
}
METRICS = ("wmrd", "mrd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed-front5 raw/no-gate drift curves.")
    parser.add_argument("--root", type=Path, required=True, help="fixed_front5_fourdataset run root")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def apply_paper_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_stats(trace: str, metric: str, minutes: np.ndarray, values: np.ndarray, df: pd.DataFrame) -> dict[str, object]:
    n = len(values)
    edge_n = max(5, int(round(n * 0.10)))
    x = minutes.astype(float)
    y = values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 1 and np.std(y) > 0 else 0.0
    first = float(np.mean(y[:edge_n]))
    last = float(np.mean(y[-edge_n:]))
    return {
        "trace": trace,
        "trace_label": TRACE_LABELS[trace],
        "metric": metric.upper(),
        "n_windows": n,
        "minute_start": int(minutes[0]),
        "minute_end": int(minutes[-1]),
        "mean": float(np.mean(y)),
        "p50": float(np.percentile(y, 50)),
        "p95": float(np.percentile(y, 95)),
        "max": float(np.max(y)),
        "first10pct_mean": first,
        "last10pct_mean": last,
        "last_minus_first": last - first,
        "relative_change": (last / first - 1.0) if first > 0 else np.nan,
        "linear_slope_per_minute": float(slope),
        "linear_slope_per_100min": float(slope * 100.0),
        "pearson_time_corr": corr,
        "bootstrap_sft_sec": float(pd.to_numeric(df["bootstrap_sft_sec"], errors="coerce").max())
        if "bootstrap_sft_sec" in df
        else np.nan,
        "infer_sec_p95": float(np.percentile(pd.to_numeric(df["infer_sec"], errors="coerce"), 95))
        if "infer_sec" in df
        else np.nan,
    }


def plot_one(out_dir: Path, trace: str, metric: str, minutes: np.ndarray, values: np.ndarray) -> None:
    tick_size = 19
    label_size = 24
    line_width = 2.5
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 6.0))
    ax.plot(minutes, values, linestyle="-", linewidth=line_width, color="C0")
    xmin = float(np.min(minutes))
    xmax = float(np.max(minutes))
    pad = 0.05 * max(1.0, xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="y")
    ax.grid(True, linestyle="--", axis="x")
    fig.tight_layout()
    stem = f"{trace}_fixed_front5_raw_{metric}_vs_time"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for trace, dirname in TRACE_DIRS.items():
        path = args.root / dirname / "window_metrics.csv"
        df = pd.read_csv(path).sort_values("minute").reset_index(drop=True)
        minutes = df["minute"].astype(int).to_numpy()
        curve_df = pd.DataFrame({"minute": minutes})
        for metric in METRICS:
            values = df[metric].astype(float).to_numpy()
            curve_df[metric] = values
            plot_one(args.out_dir, trace, metric, minutes, values)
            summary_rows.append(metric_stats(trace, metric, minutes, values, df))
        curve_df.to_csv(curves_dir / f"{trace}_fixed_front5_raw_curves.csv", index=False)
    write_csv(args.out_dir / "summary_fixed_front5_raw_drift.csv", summary_rows)
    print(args.out_dir / "summary_fixed_front5_raw_drift.csv")


if __name__ == "__main__":
    main()
