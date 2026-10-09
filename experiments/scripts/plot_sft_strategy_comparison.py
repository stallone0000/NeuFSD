#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.cbook import boxplot_stats
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

RUN_DIRS = {
    ("caida_2016", "last1"): "caida2016_last_e5_s2000_newpre",
    ("caida_2016", "window5"): "caida2016_window5_e5_s2000_newpre",
    ("caida_2016", "avg5"): "caida2016_avg5_e5_s2000_newpre",
    ("caida_2018", "last1"): "caida2018_last_e5_s2000_newpre",
    ("caida_2018", "window5"): "caida2018_window5_e5_s2000_newpre",
    ("caida_2018", "avg5"): "caida2018_avg5_e5_s2000_newpre",
    ("imc", "last1"): "imc_last_e5_s2000_newpre_imcdefault",
    ("imc", "window5"): "imc_window5_e5_s2000_newpre_imcdefault",
    ("imc", "avg5"): "imc_avg5_e5_s2000_newpre_imcdefault",
    ("mawi", "last1"): "mawi_last_e5_s2000_newpre",
    ("mawi", "window5"): "mawi_window5_e5_s2000_newpre",
    ("mawi", "avg5"): "mawi_avg5_e5_s2000_newpre",
}

STRATEGIES = [
    ("last1", "Last-1", "C0"),
    ("window5", "Window-5", "C1"),
    ("avg5", "Avg-5", "C2"),
]
METRICS = ("wmrd", "mrd")
Y_LIMITS = {
    ("caida_2016", "wmrd"): (0.0, 0.16),
    ("caida_2018", "wmrd"): (0.0, 0.16),
    ("imc", "wmrd"): (0.0, 0.60),
    ("mawi", "wmrd"): (0.0, 0.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Last-1, Window-5, and Avg-5 continuous SFT comparison.")
    parser.add_argument("--run-root", type=Path, required=True)
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


def load_curve(path: Path, metric: str, smooth_window: int) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("minute").reset_index(drop=True)
    out = pd.DataFrame({"minute": df["minute"].astype(int), "raw": df[metric].astype(float)})
    out["smooth"] = out["raw"].rolling(window=smooth_window, min_periods=1).mean()
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_curves_csv(path: Path, curves: dict[str, pd.DataFrame]) -> None:
    base = None
    for strategy, curve in curves.items():
        sub = curve[["minute", "smooth"]].rename(columns={"smooth": strategy})
        base = sub if base is None else base.merge(sub, on="minute", how="outer")
    assert base is not None
    base.sort_values("minute").to_csv(path, index=False)


def plot_one(
    out_dir: Path,
    trace: str,
    metric: str,
    curves: dict[str, pd.DataFrame],
    smooth_window: int,
) -> None:
    tick_size = 24
    label_size = 31
    inset_tick_size = tick_size
    line_width = 5.2
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
    all_minutes = []
    all_smooth = []
    all_raw = []
    for key, label, color in STRATEGIES:
        curve = curves[key]
        minutes = curve["minute"].to_numpy(dtype=int)
        smooth = curve["smooth"].to_numpy(dtype=float)
        raw = curve["raw"].to_numpy(dtype=float)
        all_minutes.append(minutes)
        all_smooth.append(smooth)
        all_raw.append(raw)
        ax.plot(
            minutes,
            smooth,
            linestyle="-",
            linewidth=line_width,
            color=color,
        )
    merged_minutes = np.concatenate(all_minutes)
    xmin = float(np.min(merged_minutes))
    xmax = float(np.max(merged_minutes))
    pad = 0.05 * max(1.0, xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ymax = float(np.max(np.concatenate(all_smooth)))
    ytop = ymax * 2.35 if ymax > 0 else 1.0
    if (trace, metric) in Y_LIMITS:
        ymin, fixed_ymax = Y_LIMITS[(trace, metric)]
        ax.set_ylim(ymin, max(fixed_ymax, ytop))
    else:
        ax.set_ylim(0.0, ytop)
    ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="y")
    ax.grid(True, linestyle="--", axis="x")

    inset = ax.inset_axes([0.60, 0.54, 0.35, 0.40])
    data = [curves[key]["raw"].to_numpy(dtype=float) for key, _, _ in STRATEGIES]
    colors = [color for _, _, color in STRATEGIES]
    box = inset.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        widths=0.52,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 3.8,
        },
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    inset.set_xticks([1, 2, 3])
    inset.set_xticklabels(["L", "W", "A"])
    inset.tick_params(axis="both", labelsize=inset_tick_size, direction="in", pad=1.5)
    for tick in inset.get_xticklabels() + inset.get_yticklabels():
        tick.set_fontweight("bold")
    inset.grid(True, linestyle="--", axis="y", linewidth=0.6, alpha=0.8)
    stats = boxplot_stats(data, whis=1.5)
    raw_low = min(float(stat["whislo"]) for stat in stats)
    raw_high = max(float(stat["whishi"]) for stat in stats)
    raw_pad = 0.08 * max(1e-9, raw_high - raw_low)
    inset.set_ylim(max(0.0, raw_low - raw_pad), raw_high + raw_pad)
    fig.tight_layout()
    stem = f"{trace}_sft_strategy_rolling{smooth_window}_{metric}_vs_time"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_standalone_legend(out_dir: Path) -> None:
    line_width = 5.2
    handles = [
        Line2D([0], [0], color=color, linewidth=line_width, linestyle="-", label=label)
        for _, label, color in STRATEGIES
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.9))
    ax.axis("off")
    legend = ax.legend(handles=handles, loc="center", ncol=3, frameon=True, handlelength=3.0)
    plt.setp(legend.get_texts(), fontweight="bold", fontsize=18)
    fig.savefig(out_dir / "legend_sft_strategy.pdf", bbox_inches="tight")
    plt.close(fig)


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
            curves: dict[str, pd.DataFrame] = {}
            means: dict[str, float] = {}
            for strategy, label, _ in STRATEGIES:
                csv_path = args.run_root / RUN_DIRS[(trace, strategy)] / "window_metrics.csv"
                curve = load_curve(csv_path, metric, args.smooth_window)
                curves[strategy] = curve
                means[strategy] = float(curve["raw"].mean())
                values = curve["raw"].to_numpy(dtype=float)
                smooth = curve["smooth"].to_numpy(dtype=float)
                summary_rows.append(
                    {
                        "trace": trace,
                        "trace_label": TRACE_LABELS[trace],
                        "metric": metric.upper(),
                        "strategy": strategy,
                        "strategy_label": label,
                        "smooth_window": args.smooth_window,
                        "n_windows": len(curve),
                        "minute_start": int(curve["minute"].iloc[0]),
                        "minute_end": int(curve["minute"].iloc[-1]),
                        "raw_mean": float(np.mean(values)),
                        "raw_p95": float(np.percentile(values, 95)),
                        "raw_max": float(np.max(values)),
                        "smooth_mean": float(np.mean(smooth)),
                        "smooth_p95": float(np.percentile(smooth, 95)),
                        "smooth_max": float(np.max(smooth)),
                        "source_csv": str(csv_path),
                    }
                )
            save_curves_csv(curves_dir / f"{trace}_rolling{args.smooth_window}_{metric}_strategy_curves.csv", curves)
            plot_one(args.out_dir, trace, metric, curves, args.smooth_window)
    save_standalone_legend(args.out_dir)
    write_csv(args.out_dir / "summary_sft_strategy.csv", summary_rows)
    print(args.out_dir / "summary_sft_strategy.csv")


if __name__ == "__main__":
    main()
