#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TRACE_LABELS = {
    "caida_2016": "CAIDA2016",
    "caida_2018": "CAIDA2018",
    "imc": "IMC",
    "mawi": "MAWI",
}
PRETRAIN_RUNS = {
    "caida_2016": "caida2016_avg5_e5_s2000_newpre",
    "caida_2018": "caida2018_avg5_e5_s2000_newpre",
    "imc": "imc_avg5_e5_s2000_newpre_imcdefault",
    "mawi": "mawi_avg5_e5_s2000_newpre",
}
METRICS = ("wmrd", "mrd")
LINE_STYLES = [
    ("fixed_front5", "One-time SFT", "C0"),
    ("scratch_avg5", "+ Continuous SFT", "C1"),
    ("pretrain_avg5", "+ Pre-train", "C2"),
]
Y_LIMITS = {
    ("caida_2018", "wmrd"): (0.0, 0.50),
    ("mawi", "wmrd"): (0.0, 0.85),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation curves: fixed front5, continuous avg5 scratch, and pretrain+avg5 SFT."
    )
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--pretrain-root", type=Path, required=True)
    parser.add_argument(
        "--pretrain-gate-curves-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing dataset-specific best-gate curves. "
            "When set, the pretrain+SFT curve uses curves/<trace>_avg5_curves.csv gate_<metric>."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=15)
    return parser.parse_args()


def apply_paper_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def fixed_path(root: Path, trace: str) -> Path:
    return root / f"{trace}_fixed_front5_e20" / "window_metrics.csv"


def scratch_path(root: Path, trace: str) -> Path:
    return root / f"{trace}_scratch_avg5_e5" / "window_metrics.csv"


def pretrain_path(root: Path, trace: str) -> Path:
    return root / PRETRAIN_RUNS[trace] / "window_metrics.csv"


def load_curve(path: Path, metric: str, smooth_window: int, column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("minute").reset_index(drop=True)
    value_column = column or metric
    out = pd.DataFrame({"minute": df["minute"].astype(int), "raw": df[value_column].astype(float)})
    out["smooth"] = out["raw"].rolling(window=smooth_window, min_periods=1).mean()
    return out


def recompute_smooth(curve: pd.DataFrame, smooth_window: int) -> None:
    curve["smooth"] = curve["raw"].rolling(window=smooth_window, min_periods=1).mean()


def align_continuous_start(fixed: pd.DataFrame, continuous: pd.DataFrame, smooth_window: int) -> None:
    if fixed.empty or continuous.empty:
        return
    first_minute = max(int(fixed["minute"].iloc[0]), int(continuous["minute"].iloc[0]))
    fixed_match = fixed.index[fixed["minute"] == first_minute]
    continuous_match = continuous.index[continuous["minute"] == first_minute]
    if len(fixed_match) == 0 or len(continuous_match) == 0:
        return
    continuous.loc[continuous_match[0], "raw"] = float(fixed.loc[fixed_match[0], "raw"])
    recompute_smooth(continuous, smooth_window)


def pretrain_gate_curve_path(root: Path, trace: str) -> Path:
    return root / f"{trace}_avg5_curves.csv"


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


def add_summary(rows: list[dict[str, object]], trace: str, metric: str, method: str, curve: pd.DataFrame) -> None:
    values = curve["raw"].to_numpy(dtype=float)
    smooth = curve["smooth"].to_numpy(dtype=float)
    n = len(values)
    edge_n = max(5, int(round(n * 0.10)))
    rows.append(
        {
            "trace": trace,
            "trace_label": TRACE_LABELS[trace],
            "metric": metric.upper(),
            "method": method,
            "smooth_window": int(curve.attrs.get("smooth_window", 0)),
            "n_windows": n,
            "minute_start": int(curve["minute"].iloc[0]),
            "minute_end": int(curve["minute"].iloc[-1]),
            "raw_mean": float(np.mean(values)),
            "raw_p95": float(np.percentile(values, 95)),
            "raw_max": float(np.max(values)),
            "smooth_mean": float(np.mean(smooth)),
            "smooth_p95": float(np.percentile(smooth, 95)),
            "smooth_max": float(np.max(smooth)),
            "first10pct_smooth_mean": float(np.mean(smooth[:edge_n])),
            "last10pct_smooth_mean": float(np.mean(smooth[-edge_n:])),
            "last_minus_first_smooth": float(np.mean(smooth[-edge_n:]) - np.mean(smooth[:edge_n])),
        }
    )


def save_curve_csv(path: Path, curves: dict[str, pd.DataFrame]) -> None:
    base = None
    for name, curve in curves.items():
        sub = curve[["minute", "smooth"]].rename(columns={"smooth": name})
        base = sub if base is None else base.merge(sub, on="minute", how="outer")
    assert base is not None
    base.sort_values("minute").to_csv(path, index=False)


def plot_one(out_dir: Path, trace: str, metric: str, curves: dict[str, pd.DataFrame], smooth_window: int) -> None:
    tick_size = 24
    label_size = 31
    line_width = 5.2
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
    all_minutes = []
    for key, label, color in LINE_STYLES:
        curve = curves[key]
        minutes = curve["minute"].to_numpy(dtype=int)
        all_minutes.append(minutes)
        ax.plot(
            minutes,
            curve["smooth"].to_numpy(dtype=float),
            label=label,
            linestyle="-",
            linewidth=line_width,
            color=color,
        )
    merged_minutes = np.concatenate(all_minutes)
    xmin = float(np.min(merged_minutes))
    xmax = float(np.max(merged_minutes))
    pad = 0.05 * max(1.0, xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    if (trace, metric) in Y_LIMITS:
        ax.set_ylim(*Y_LIMITS[(trace, metric)])
    ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="y")
    ax.grid(True, linestyle="--", axis="x")
    fig.tight_layout()
    stem = f"{trace}_ablation_rolling{smooth_window}_{metric}_vs_time"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_standalone_legend(out_dir: Path) -> None:
    line_width = 5.2
    handles = [
        Line2D([0], [0], color=color, linewidth=line_width, linestyle="-", label=label)
        for _, label, color in LINE_STYLES
    ]
    fig, ax = plt.subplots(figsize=(8.0, 0.9))
    ax.axis("off")
    legend = ax.legend(handles=handles, loc="center", ncol=3, frameon=True, handlelength=3.0)
    plt.setp(legend.get_texts(), fontweight="bold", fontsize=18)
    fig.savefig(out_dir / "legend_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def set_curve_attrs(curve: pd.DataFrame, smooth_window: int, source_csv: Path, source_column: str) -> pd.DataFrame:
    curve.attrs["smooth_window"] = smooth_window
    curve.attrs["source_csv"] = str(source_csv)
    curve.attrs["source_column"] = source_column
    return curve


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for trace in TRACE_LABELS:
        for metric in METRICS:
            fixed_csv = fixed_path(args.fixed_root, trace)
            scratch_csv = scratch_path(args.scratch_root, trace)
            if args.pretrain_gate_curves_dir is not None:
                pretrain_csv = pretrain_gate_curve_path(args.pretrain_gate_curves_dir, trace)
                pretrain_column = f"gate_{metric}"
            else:
                pretrain_csv = pretrain_path(args.pretrain_root, trace)
                pretrain_column = metric
            curves = {
                "fixed_front5": set_curve_attrs(
                    load_curve(fixed_csv, metric, args.smooth_window),
                    args.smooth_window,
                    fixed_csv,
                    metric,
                ),
                "scratch_avg5": set_curve_attrs(
                    load_curve(scratch_csv, metric, args.smooth_window),
                    args.smooth_window,
                    scratch_csv,
                    metric,
                ),
                "pretrain_avg5": set_curve_attrs(
                    load_curve(pretrain_csv, metric, args.smooth_window, column=pretrain_column),
                    args.smooth_window,
                    pretrain_csv,
                    pretrain_column,
                ),
            }
            align_continuous_start(curves["fixed_front5"], curves["scratch_avg5"], args.smooth_window)
            for method, curve in curves.items():
                add_summary(summary_rows, trace, metric, method, curve)
            save_curve_csv(curves_dir / f"{trace}_rolling{args.smooth_window}_{metric}_curves.csv", curves)
            plot_one(args.out_dir, trace, metric, curves, args.smooth_window)
    save_standalone_legend(args.out_dir)
    write_csv(args.out_dir / "summary_ablation_rolling.csv", summary_rows)
    print(args.out_dir / "summary_ablation_rolling.csv")


if __name__ == "__main__":
    main()
