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


TRACE_TITLES = {
    "caida_2016": "CAIDA 2016",
    "caida_2018": "CAIDA 2018",
    "caida_2018_new": "CAIDA 2018 new",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build online-safe adaptive light-mass gate curves from existing "
            "original/full-gate curve CSVs."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", default="pretrain_continuous")
    parser.add_argument("--res", nargs="+", default=["64_64", "128_128"])
    parser.add_argument("--traces", nargs="+", default=["caida_2016", "caida_2018"])
    parser.add_argument("--source-tag", default="t1p02_c1p20_mainonly")
    parser.add_argument("--regime-window", type=int, default=80)
    parser.add_argument("--regime-min-history", type=int, default=10)
    parser.add_argument(
        "--regime-stat",
        choices=["rolling-median", "expanding-quantile", "rolling-max"],
        default="rolling-median",
    )
    parser.add_argument("--regime-quantile", type=float, default=0.75)
    parser.add_argument("--heavy-frac-threshold", type=float, default=0.50)
    parser.add_argument("--light-window", type=int, default=80)
    parser.add_argument("--light-min-history", type=int, default=10)
    parser.add_argument("--light-quantile", type=float, default=0.50)
    parser.add_argument("--light-ratio", type=float, default=1.03)
    parser.add_argument("--light-delta", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def source_dir(run_root: Path, source_tag: str, mode: str, res: str) -> Path:
    return run_root / "plots" / f"gate_full_curves_{source_tag}_{mode}_{res}_two_traces"


def past_rolling_quantile(series: pd.Series, window: int, min_history: int, quantile: float) -> pd.Series:
    shifted = series.shift(1)
    rolling = shifted.rolling(window, min_periods=min_history).quantile(quantile)
    expanding = shifted.expanding(min_periods=min_history).quantile(quantile)
    return rolling.fillna(expanding)


def past_rolling_median(series: pd.Series, window: int, min_history: int) -> pd.Series:
    return past_rolling_quantile(series, window, min_history, 0.50)


def past_regime_stat(series: pd.Series, args: argparse.Namespace) -> pd.Series:
    shifted = series.shift(1)
    if args.regime_stat == "rolling-median":
        return past_rolling_median(series, args.regime_window, args.regime_min_history)
    if args.regime_stat == "expanding-quantile":
        return shifted.expanding(min_periods=args.regime_min_history).quantile(args.regime_quantile)
    if args.regime_stat == "rolling-max":
        rolling = shifted.rolling(args.regime_window, min_periods=args.regime_min_history).max()
        expanding = shifted.expanding(min_periods=args.regime_min_history).max()
        return rolling.fillna(expanding)
    raise ValueError(args.regime_stat)


def apply_adaptive_gate(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.sort_values("minute").reset_index(drop=True).copy()
    packet_count = out["heavy_packet_mass"] + out["residual_light_mass"]
    out["heavy_frac"] = np.divide(
        out["heavy_packet_mass"],
        packet_count,
        out=np.zeros(len(out), dtype=float),
        where=packet_count.to_numpy() > 0,
    )
    out["light_frac"] = np.divide(
        out["residual_light_mass"],
        packet_count,
        out=np.zeros(len(out), dtype=float),
        where=packet_count.to_numpy() > 0,
    )

    out["past_heavy_frac_regime_stat"] = past_regime_stat(out["heavy_frac"], args)
    out["past_light_frac_quantile"] = past_rolling_quantile(
        out["light_frac"], args.light_window, args.light_min_history, args.light_quantile
    )

    normal_heavy_regime = out["past_heavy_frac_regime_stat"] >= args.heavy_frac_threshold
    light_residual_spike = (
        (out["light_frac"] >= out["past_light_frac_quantile"] * args.light_ratio)
        & (out["light_frac"] >= out["past_light_frac_quantile"] + args.light_delta)
    )
    full_gate_available = out["gated_seed_fraction"] > 0
    out["adaptive_use_gate"] = (normal_heavy_regime | light_residual_spike).fillna(False) & full_gate_available
    out["adaptive_regime"] = np.where(normal_heavy_regime.fillna(False), "normal-heavy", "conservative")
    out["adaptive_mrd"] = np.where(out["adaptive_use_gate"], out["gate_mrd"], out["original_mrd"])
    out["adaptive_wmrd"] = np.where(out["adaptive_use_gate"], out["gate_wmrd"], out["original_wmrd"])
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(res: str, trace: str, df: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "res": res,
        "trace": trace,
        "n": int(len(df)),
        "adaptive_gated_windows": int(df["adaptive_use_gate"].sum()),
        "normal_heavy_windows": int((df["adaptive_regime"] == "normal-heavy").sum()),
    }
    for prefix in ["original", "gate", "adaptive"]:
        for metric in ["mrd", "wmrd"]:
            values = df[f"{prefix}_{metric}"].to_numpy(dtype=float)
            row[f"{prefix}_{metric}"] = float(np.mean(values))
            row[f"max_{prefix}_{metric}"] = float(np.max(values))
            row[f"max_{prefix}_{metric}_minute"] = int(df.iloc[int(np.argmax(values))]["minute"])
    row["adaptive_delta_wmrd_vs_original"] = row["adaptive_wmrd"] - row["original_wmrd"]
    row["adaptive_delta_wmrd_vs_gate"] = row["adaptive_wmrd"] - row["gate_wmrd"]
    row["adaptive_delta_mrd_vs_original"] = row["adaptive_mrd"] - row["original_mrd"]
    row["adaptive_delta_mrd_vs_gate"] = row["adaptive_mrd"] - row["gate_mrd"]
    return row


def plot_trace(path: Path, res: str, trace: str, df: pd.DataFrame) -> None:
    minutes = df["minute"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    for ax, metric in [(axes[0], "wmrd"), (axes[1], "mrd")]:
        ax.plot(minutes, df[f"original_{metric}"], label="Original", linewidth=1.15)
        ax.plot(minutes, df[f"gate_{metric}"], label="Light residual gate", linewidth=1.15)
        ax.plot(minutes, df[f"adaptive_{metric}"], label="Adaptive light gate", linewidth=1.35)
        ax.set_ylabel(f"{metric.upper()} error")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[0].set_title(f"{res} {TRACE_TITLES.get(trace, trace)}: light gate variants")
    axes[1].set_xlabel("Time (minute)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_grid(path: Path, curves: dict[tuple[str, str], pd.DataFrame], metric: str) -> None:
    keys = list(curves)
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 2.45 * len(keys)), sharex=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, (res, trace) in zip(axes, keys):
        df = curves[(res, trace)]
        minutes = df["minute"].to_numpy(dtype=int)
        ax.plot(minutes, df[f"original_{metric}"], label="Original", linewidth=1.0)
        ax.plot(minutes, df[f"gate_{metric}"], label="Light residual gate", linewidth=1.0)
        ax.plot(minutes, df[f"adaptive_{metric}"], label="Adaptive light gate", linewidth=1.25)
        ax.set_title(f"{res} {TRACE_TITLES.get(trace, trace)}")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (minute)")
    axes[0].legend(loc="upper left", ncols=3)
    fig.suptitle(f"{metric.upper()} over time", y=0.995)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = args.out_dir or (
        run_root
        / "plots"
        / (
            "adaptive_lightmass_gate_"
            f"{args.regime_stat}_h{args.heavy_frac_threshold:g}_lfq{args.light_quantile:g}_"
            f"w{args.light_window}_r{args.light_ratio:g}_{args.mode}_mainonly"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for res in args.res:
        src = source_dir(run_root, args.source_tag, args.mode, res)
        for trace in args.traces:
            csv_path = src / f"{res}_{trace}_gate_full_curves.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            df = apply_adaptive_gate(pd.read_csv(csv_path), args)
            curves[(res, trace)] = df
            df.to_csv(out_dir / f"{res}_{trace}_adaptive_gate_curves.csv", index=False)
            plot_trace(out_dir / f"{res}_{trace}_original_vs_gate_vs_adaptive", res, trace, df)
            summary_rows.append(summarize(res, trace, df))

    write_csv(out_dir / "adaptive_gate_summary.csv", summary_rows)
    plot_metric_grid(out_dir / "wmrd_time_original_vs_gate_vs_adaptive", curves, "wmrd")
    plot_metric_grid(out_dir / "mrd_time_original_vs_gate_vs_adaptive", curves, "mrd")
    print(f"wrote adaptive gate curves to {out_dir}")


if __name__ == "__main__":
    main()
