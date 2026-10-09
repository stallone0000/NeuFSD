#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import run_comprehensive_compare as comp


def style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


COLORS = {
    "NeuFSD": "C3",
    "Elastic": "C0",
    "MRAC": "C1",
    "Array Sample": "C2",
    "Hash Sample": "C4",
    "DaVinci": "C5",
    "NeuFSD-GPU": "C3",
    "NeuFSD-CPU": "C6",
}
MARKERS = {
    "NeuFSD": "*",
    "NeuFSD-CPU": "P",
    "Elastic": "o",
    "MRAC": "s",
    "Array Sample": "^",
    "Hash Sample": "D",
    "DaVinci": "v",
}

FIGSIZE = (8.0, 4.0)
TICK_SIZE = 24
LABEL_SIZE = 31
LINE_WIDTH = 4.2
MARKER_SIZE = 15
MARKER_EDGE_WIDTH = 2.8
BASELINE_ZORDER = 3
NEUFSD_ZORDER = 10
BAND_ZORDER = 1


def bold_ticks(ax) -> None:
    ax.tick_params(labelsize=TICK_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def set_memory_xaxis(ax) -> None:
    ax.set_xscale("log", base=2)
    log_min = math.log2(comp.MEM_KB[0])
    log_max = math.log2(comp.MEM_KB[-1])
    pad = 0.05 * (log_max - log_min)
    ax.set_xlim(2 ** (log_min - pad), 2 ** (log_max + pad))
    ax.set_xticks(comp.MEM_KB)
    ax.set_xticklabels([str(k) for k in comp.MEM_KB])


def set_padded_log_ylim(ax, values: list[float], *, pad_frac: float = 0.12) -> None:
    vals = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=float)
    if len(vals) == 0:
        return
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        ax.set_ylim(lo / 2.0, hi * 2.0)
        return
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    pad = pad_frac * (log_hi - log_lo)
    ax.set_ylim(10 ** (log_lo - pad), 10 ** (log_hi + pad))


def set_padded_linear_ylim(ax, values: list[float], *, bottom_zero: bool = False) -> None:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(1e-12, hi - lo)
    bottom = 0.0 if bottom_zero else lo - 0.12 * span
    ax.set_ylim(bottom, hi + 0.14 * span)


def dataset_max_window(trace: str) -> int:
    return comp.dataset_max_window(trace)


def windows(start: int, count: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for trace in comp.DATASETS:
        end = min(dataset_max_window(trace), start + count - 1)
        out[trace] = list(range(start, end + 1))
    return out


def summarize(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan, math.nan
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def load_ours_latest(run_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, Path]]]:
    latest_imc = (
        comp.ROOT
        / "mainonly_runs_20260623"
        / "imc_current_oracle_20260630"
        / "p50_s2000_e20_full"
        / "window_metrics.csv"
    )
    if latest_imc.exists():
        comp.OURS_CANDIDATES["imc"] = [latest_imc] + [
            p for p in comp.OURS_CANDIDATES["imc"] if p != latest_imc
        ]
    ours_cache: dict[str, pd.DataFrame] = {}
    chosen: dict[str, tuple[str, Path]] = {}
    for trace in comp.DATASETS:
        ours, opt_name, src = comp.load_ours(trace)
        ours_cache[trace] = ours
        chosen[trace] = (opt_name, src)
    return ours_cache, chosen


def load_timing(run_root: Path) -> dict[str, float]:
    path = run_root / "forward_timing.json"
    if path.exists():
        return json.loads(path.read_text())
    return comp.measure_forward_time(run_root, force=False)


def draw_legend(plot_dir: Path) -> None:
    handles = [
        Line2D([0], [0], color=COLORS["NeuFSD"], lw=3.8, label="NeuFSD"),
        Line2D([0], [0], color=COLORS["Elastic"], marker="o", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="Elastic"),
        Line2D([0], [0], color=COLORS["MRAC"], marker="s", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="MRAC"),
        Line2D([0], [0], color=COLORS["Array Sample"], marker="^", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="Array Sample"),
        Line2D([0], [0], color=COLORS["Hash Sample"], marker="D", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="Hash Sample"),
        Line2D([0], [0], color=COLORS["DaVinci"], marker="v", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="DaVinci"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 0.75))
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", ncol=6, frameon=False, handlelength=3, columnspacing=1.5)
    plt.setp(leg.get_texts(), fontweight="bold", fontsize=16)
    fig.savefig(plot_dir / "legend_algorithms.pdf", bbox_inches="tight")
    plt.close(fig)

    handles = [
        Line2D([0], [0], color=COLORS["NeuFSD"], lw=3.8, label="NeuFSD"),
        Line2D([0], [0], color=COLORS["Elastic"], lw=3.2, label="Elastic"),
        Line2D([0], [0], color=COLORS["MRAC"], lw=3.2, label="MRAC"),
        Line2D([0], [0], color=COLORS["Array Sample"], lw=3.2, label="Array Sample"),
        Line2D([0], [0], color=COLORS["Hash Sample"], lw=3.2, label="Hash Sample"),
        Line2D([0], [0], color=COLORS["DaVinci"], lw=3.2, label="DaVinci"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 0.75))
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", ncol=6, frameon=False, handlelength=3, columnspacing=1.5)
    plt.setp(leg.get_texts(), fontweight="bold", fontsize=16)
    fig.savefig(plot_dir / "legend_algorithms_lines.pdf", bbox_inches="tight")
    plt.close(fig)

    handles = [
        Line2D([0], [0], color=COLORS["NeuFSD-GPU"], lw=3.8, label="NeuFSD-GPU"),
        Line2D([0], [0], color=COLORS["NeuFSD-CPU"], lw=3.8, ls="--", label="NeuFSD-CPU"),
        Line2D([0], [0], color=COLORS["Elastic"], marker="o", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="Elastic"),
        Line2D([0], [0], color=COLORS["MRAC"], marker="s", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="MRAC"),
        Line2D([0], [0], color=COLORS["DaVinci"], marker="v", markerfacecolor="none", markeredgewidth=2.6, markersize=14, lw=3.2, label="DaVinci"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 0.75))
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", ncol=5, frameon=False, handlelength=3, columnspacing=1.5)
    plt.setp(leg.get_texts(), fontweight="bold", fontsize=16)
    fig.savefig(plot_dir / "legend_decode.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_memory_sweep(
    run_root: Path,
    df: pd.DataFrame,
    plot_dir: Path,
    sweep_windows: dict[str, list[int]],
    ours_cache: dict[str, pd.DataFrame],
    timing: dict[str, float],
) -> None:
    specs = [
        ("wmrd", "WMRD", True),
        ("mrd", "MRD", True),
        ("decode_ms", "Time (s)", True),
        ("insert_speed", "Speed (Mps)", False),
    ]
    for metric, ylabel, logy in specs:
        for trace, minutes in sweep_windows.items():
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
            set_memory_xaxis(ax)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", axis="both", alpha=0.55)
            all_y_values: list[float] = []
            error_ylim_values: list[float] = []

            for alg in comp.BASELINE_ALGS:
                if metric == "decode_ms" and alg in {"array", "hash"}:
                    continue
                label = comp.ALG_LABEL[alg]
                means, lows, highs = [], [], []
                for kb in comp.MEM_KB:
                    sub = df[
                        (df.trace == trace)
                        & (df.algorithm == alg)
                        & (df.memory_kb == kb)
                        & (df.minute.isin(minutes))
                        & (df.status == "ok")
                    ]
                    if metric == "insert_speed":
                        vals = comp.insert_speed_mpps(sub)
                    elif metric == "decode_ms":
                        vals = pd.to_numeric(sub["decode_ms"], errors="coerce") / 1000.0
                        vals = vals.mask(vals <= 0, 1e-6)
                    else:
                        vals = pd.to_numeric(sub[metric], errors="coerce")
                    mean, lo, hi = summarize(vals)
                    means.append(mean)
                    lows.append(lo)
                    highs.append(hi)
                y = np.array(means, dtype=float)
                low = np.array(lows, dtype=float)
                high = np.array(highs, dtype=float)
                finite = y[np.isfinite(y)]
                all_y_values.extend([float(v) for v in finite])
                if metric in {"wmrd", "mrd"} and label != "DaVinci":
                    error_ylim_values.extend([float(v) for v in finite if v > 0])
                ax.errorbar(
                    comp.MEM_KB,
                    y,
                    yerr=np.vstack([np.maximum(y - low, 0), np.maximum(high - y, 0)]),
                    color=COLORS[label],
                    marker=MARKERS[label],
                    markersize=MARKER_SIZE,
                    markerfacecolor="none",
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    linewidth=LINE_WIDTH,
                    capsize=4,
                    elinewidth=2.0,
                    zorder=BASELINE_ZORDER,
                )

            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(minutes)]
            if metric in {"wmrd", "mrd"}:
                vals = pd.to_numeric(ours[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals):
                    mean_val = float(np.mean(vals))
                    ax.plot(
                        comp.MEM_KB,
                        [mean_val] * len(comp.MEM_KB),
                        color=COLORS["NeuFSD"],
                        linewidth=LINE_WIDTH + 0.3,
                        zorder=NEUFSD_ZORDER,
                    )
                    ax.fill_between(
                        comp.MEM_KB,
                        np.min(vals),
                        np.max(vals),
                        color=COLORS["NeuFSD"],
                        alpha=0.13,
                        zorder=BAND_ZORDER,
                    )
                    all_y_values.append(mean_val)
                    error_ylim_values.extend([float(v) for v in vals if np.isfinite(v) and v > 0])
            elif metric == "decode_ms":
                if "h800_ms" in timing:
                    h800_val = timing["h800_ms"] / 1000.0
                    ax.plot(
                        comp.MEM_KB,
                        [h800_val] * len(comp.MEM_KB),
                        color=COLORS["NeuFSD-GPU"],
                        linewidth=LINE_WIDTH + 0.3,
                        zorder=NEUFSD_ZORDER,
                    )
                    all_y_values.append(float(h800_val))
                cpu_val = timing["cpu_ms"] / 1000.0
                ax.plot(
                    comp.MEM_KB,
                    [cpu_val] * len(comp.MEM_KB),
                    color=COLORS["NeuFSD-CPU"],
                    linestyle="--",
                    linewidth=LINE_WIDTH + 0.3,
                    zorder=NEUFSD_ZORDER,
                )
                all_y_values.append(float(cpu_val))
            else:
                vals = comp.ours_insert_speed_from_elastic(df, trace, minutes)
                if len(vals):
                    mean_val = float(np.mean(vals))
                    ax.plot(
                        comp.MEM_KB,
                        [mean_val] * len(comp.MEM_KB),
                        color=COLORS["NeuFSD"],
                        linewidth=LINE_WIDTH + 0.3,
                        zorder=NEUFSD_ZORDER,
                    )
                    ax.fill_between(
                        comp.MEM_KB,
                        np.min(vals),
                        np.max(vals),
                        color=COLORS["NeuFSD"],
                        alpha=0.13,
                        zorder=BAND_ZORDER,
                    )
                    all_y_values.extend([float(v) for v in vals if np.isfinite(v)])

            if metric in {"wmrd", "mrd"}:
                set_padded_log_ylim(ax, error_ylim_values)
            elif metric == "decode_ms":
                set_padded_log_ylim(ax, all_y_values, pad_frac=0.18)
            elif metric == "insert_speed":
                set_padded_linear_ylim(ax, all_y_values, bottom_zero=True)

            ax.set_xlabel("Memory (KB)", fontweight="bold", fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=LABEL_SIZE)
            bold_ticks(ax)
            fig.tight_layout()
            fig.savefig(plot_dir / f"memory_sweep_{metric}_{trace}.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_time_series(
    df: pd.DataFrame,
    plot_dir: Path,
    time_windows: dict[str, list[int]],
    ours_cache: dict[str, pd.DataFrame],
) -> None:
    all_minutes = [m for minutes in time_windows.values() for m in minutes]
    global_xmin = float(np.min(all_minutes))
    global_xmax = float(np.max(all_minutes))
    global_pad = 0.05 * max(1.0, global_xmax - global_xmin)
    for trace, minutes in time_windows.items():
        for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(minutes)].sort_values("minute")
            ax.plot(
                ours["minute"],
                ours[metric],
                color=COLORS["NeuFSD"],
                linewidth=LINE_WIDTH,
                zorder=NEUFSD_ZORDER,
            )
            ylim_values = [
                float(v)
                for v in pd.to_numeric(ours[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if np.isfinite(v)
            ]
            for alg in comp.BASELINE_ALGS:
                sub = df[
                    (df.trace == trace)
                    & (df.algorithm == alg)
                    & (df.memory_kb == comp.TIME_BASELINE_KB)
                    & (df.minute.isin(minutes))
                    & (df.status == "ok")
                ].sort_values("minute")
                if sub.empty:
                    continue
                label = comp.ALG_LABEL[alg]
                values = pd.to_numeric(sub[metric], errors="coerce")
                ax.plot(
                    sub["minute"],
                    values,
                    linewidth=LINE_WIDTH,
                    color=COLORS[label],
                    zorder=BASELINE_ZORDER,
                )
                if label != "DaVinci":
                    ylim_values.extend([float(v) for v in values.dropna().to_numpy(dtype=float) if np.isfinite(v)])
            ax.set_xlim(global_xmin - global_pad, global_xmax + global_pad)
            set_padded_linear_ylim(ax, ylim_values, bottom_zero=True)
            ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=LABEL_SIZE)
            bold_ticks(ax)
            ax.grid(True, linestyle="--", axis="both", alpha=0.55)
            fig.tight_layout()
            fig.savefig(plot_dir / f"timeseries_{metric}_{trace}.pdf", bbox_inches="tight")
            plt.close(fig)


def write_latest_summary(
    run_root: Path,
    df: pd.DataFrame,
    plot_dir: Path,
    sweep_windows: dict[str, list[int]],
    time_windows: dict[str, list[int]],
    ours_cache: dict[str, pd.DataFrame],
    chosen: dict[str, tuple[str, Path]],
    timing: dict[str, float],
) -> None:
    rows = []
    for trace in comp.DATASETS:
        for phase, win in [("memory10", sweep_windows[trace]), ("timeseries", time_windows[trace])]:
            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(win)]
            rows.append(
                {
                    "phase": phase,
                    "trace": trace,
                    "algorithm": "NeuFSD",
                    "memory_kb": 16,
                    "n_windows": len(ours),
                    "mean_wmrd": float(ours["wmrd"].mean()),
                    "mean_mrd": float(ours["mrd"].mean()),
                    "mean_decode_ms_h800": timing.get("h800_ms", math.nan),
                    "mean_decode_ms_cpu": timing.get("cpu_ms", math.nan),
                    "source_csv": str(chosen[trace][1]),
                    "optimization": chosen[trace][0],
                }
            )
            for alg in comp.BASELINE_ALGS:
                mems = comp.MEM_KB if phase == "memory10" else [comp.TIME_BASELINE_KB]
                for kb in mems:
                    sub = df[
                        (df.trace == trace)
                        & (df.algorithm == alg)
                        & (df.memory_kb == kb)
                        & (df.minute.isin(win))
                        & (df.status == "ok")
                    ]
                    if sub.empty:
                        continue
                    rows.append(
                        {
                            "phase": phase,
                            "trace": trace,
                            "algorithm": comp.ALG_LABEL[alg],
                            "memory_kb": kb,
                            "n_windows": len(sub),
                            "mean_wmrd": pd.to_numeric(sub["wmrd"], errors="coerce").mean(),
                            "mean_mrd": pd.to_numeric(sub["mrd"], errors="coerce").mean(),
                            "mean_decode_ms_cpu": pd.to_numeric(sub["decode_ms"], errors="coerce").mean(),
                        }
                    )
    pd.DataFrame(rows).to_csv(plot_dir / "summary_compare_latest_imc.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=comp.RUN_ROOT_DEFAULT)
    ap.add_argument("--start-minute", type=int, default=5)
    ap.add_argument("--sweep-windows", type=int, default=10)
    ap.add_argument("--time-windows", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run_root = args.run_root.resolve()
    plot_dir = args.out_dir.resolve() if args.out_dir else run_root / "plots_single_latest_imc"
    plot_dir.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(run_root / "baseline_detail.csv")
    sweep = windows(args.start_minute, args.sweep_windows)
    time = windows(args.start_minute, args.time_windows)
    ours_cache, chosen = load_ours_latest(run_root)
    timing = load_timing(run_root)

    draw_legend(plot_dir)
    plot_memory_sweep(run_root, df, plot_dir, sweep, ours_cache, timing)
    plot_time_series(df, plot_dir, time, ours_cache)
    write_latest_summary(run_root, df, plot_dir, sweep, time, ours_cache, chosen, timing)

    metadata = {
        "plot_dir": str(plot_dir),
        "latest_imc_source": str(chosen["imc"][1]),
        "latest_imc_optimization": chosen["imc"][0],
        "forward_timing": timing,
    }
    (plot_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote single-panel plots to {plot_dir}")


if __name__ == "__main__":
    main()
