#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_paper_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sliding full-stack online runs and plot error vs time.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing per-run subdirectories.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--glob", default="*")
    parser.add_argument(
        "--copy-single-epoch-style",
        action="store_true",
        help="When multiple epochs exist, still draw all epochs on the per strategy plot.",
    )
    parser.add_argument(
        "--prefer-imc-default",
        action="store_true",
        help="When *_imcdefault IMC runs exist, use them instead of older IMC runs for the same strategy.",
    )
    parser.add_argument(
        "--imc-default-sample-shape-gate",
        action="store_true",
        help="For IMC runs with sample_shape_* columns, plot/sample summarize sample_shape as the default gate.",
    )
    return parser.parse_args()


def read_args(path: Path) -> dict[str, object]:
    with path.open() as f:
        return json.load(f)


def run_dirs(root: Path, pattern: str) -> list[Path]:
    out = []
    for path in sorted(root.glob(pattern)):
        if path.is_dir() and (path / "window_metrics.csv").exists() and (path / "args.json").exists():
            out.append(path)
    return out


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "p50": float(np.nanpercentile(values, 50)),
        "p95": float(np.nanpercentile(values, 95)),
        "max": float(np.nanmax(values)),
    }


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


def summarize_run(run_dir: Path, cfg: dict[str, object], df: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "run": run_dir.name,
        "trace": cfg.get("trace", ""),
        "strategy": cfg.get("strategy", ""),
        "history_size": cfg.get("history_size", ""),
        "epochs": int(cfg.get("epochs", 0)),
        "batch_size": cfg.get("batch_size", ""),
        "lr": cfg.get("lr", ""),
        "start_seed": cfg.get("start_seed", ""),
        "end_seed": cfg.get("end_seed", ""),
        "seed_count": int(cfg.get("end_seed", 0)) - int(cfg.get("start_seed", 0)),
        "last_replicas": cfg.get("last_replicas", ""),
        "avg_replicas": cfg.get("avg_replicas", ""),
        "n_windows": int(len(df)),
        "deadline_miss": int(df["deadline_miss"].astype(str).str.lower().isin(["true", "1"]).sum())
        if "deadline_miss" in df
        else 0,
    }
    for metric in ["mrd", "wmrd", "gate_mrd", "gate_wmrd", "control_train_sec", "sft_total_sec"]:
        if metric in df:
            stats = summarize_values(df[metric].astype(float).to_numpy())
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
    if "wmrd" in df and "gate_wmrd" in df:
        row["gate_wmrd_delta_mean"] = row["gate_wmrd_mean"] - row["wmrd_mean"]
        row["gate_wmrd_delta_max"] = row["gate_wmrd_max"] - row["wmrd_max"]
    if "mrd" in df and "gate_mrd" in df:
        row["gate_mrd_delta_mean"] = row["gate_mrd_mean"] - row["mrd_mean"]
        row["gate_mrd_delta_max"] = row["gate_mrd_max"] - row["mrd_max"]
    for metric in ["mrd", "wmrd"]:
        sample_col = f"sample_shape_{metric}"
        if sample_col in df:
            stats = summarize_values(df[sample_col].astype(float).to_numpy())
            for key, value in stats.items():
                row[f"{sample_col}_{key}"] = value
    return row


def effective_gate_column(cfg: dict[str, object], df: pd.DataFrame, metric: str, use_imc_default: bool) -> tuple[str, str]:
    sample_col = f"sample_shape_{metric}"
    if use_imc_default and cfg.get("trace") == "imc" and sample_col in df:
        return sample_col, "gate"
    return f"gate_{metric}", "gate"


def prefer_imc_default_runs(
    loaded: list[tuple[Path, dict[str, object], pd.DataFrame]],
) -> list[tuple[Path, dict[str, object], pd.DataFrame]]:
    has_default: set[tuple[str, str]] = set()
    for run_dir, cfg, _ in loaded:
        if cfg.get("trace") == "imc" and run_dir.name.endswith("_imcdefault"):
            has_default.add((str(cfg.get("trace", "")), str(cfg.get("strategy", ""))))
    if not has_default:
        return loaded
    filtered = []
    for run_dir, cfg, df in loaded:
        key = (str(cfg.get("trace", "")), str(cfg.get("strategy", "")))
        if key in has_default and cfg.get("trace") == "imc" and not run_dir.name.endswith("_imcdefault"):
            continue
        filtered.append((run_dir, cfg, df))
    return filtered


def plot_group(
    out_dir: Path,
    trace: str,
    strategy: str,
    metric: str,
    runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
    use_imc_default_gate: bool,
) -> None:
    tick_size = 19
    label_size = 24
    line_width = 2.5
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    multiple = len(runs) > 1
    for run_dir, cfg, df in sorted(runs, key=lambda item: int(item[1].get("epochs", 0))):
        minutes = df["minute"].astype(int).to_numpy()
        epochs = int(cfg.get("epochs", 0))
        suffix = f" E{epochs}" if multiple else ""
        gate_col, gate_label = effective_gate_column(cfg, df, metric, use_imc_default_gate)
        ax.plot(
            minutes,
            df[metric].astype(float).to_numpy(),
            linewidth=line_width,
            label=f"no gate{suffix}",
            color="C0",
        )
        ax.plot(
            minutes,
            df[gate_col].astype(float).to_numpy(),
            linewidth=line_width,
            label=f"{gate_label}{suffix}",
            color="C1",
        )
    xmin = min(float(df["minute"].astype(int).min()) for _, _, df in runs)
    xmax = max(float(df["minute"].astype(int).max()) for _, _, df in runs)
    pad = 0.05 * max(1.0, xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="y")
    ax.grid(True, linestyle="--", axis="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2 if multiple else 1,
        handlelength=3,
    )
    leg = ax.get_legend()
    if leg is not None:
        plt.setp(leg.get_texts(), fontweight="bold", fontsize=20)
    fig.tight_layout()
    stem = f"{trace}_{strategy}_{metric}_vs_time"
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    out_dir = args.out_dir or (args.root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[Path, dict[str, object], pd.DataFrame]] = []
    for run_dir in run_dirs(args.root, args.glob):
        cfg = read_args(run_dir / "args.json")
        df = pd.read_csv(run_dir / "window_metrics.csv")
        loaded.append((run_dir, cfg, df))
    if args.prefer_imc_default:
        loaded = prefer_imc_default_runs(loaded)

    rows: list[dict[str, object]] = []
    for run_dir, cfg, df in loaded:
        rows.append(summarize_run(run_dir, cfg, df))

    write_csv(out_dir / "summary_by_run.csv", rows)

    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, object], pd.DataFrame]]] = {}
    for item in loaded:
        _, cfg, _ = item
        key = (str(cfg.get("trace", "")), str(cfg.get("strategy", "")))
        grouped.setdefault(key, []).append(item)
    for (trace, strategy), group in sorted(grouped.items()):
        for metric in ["wmrd", "mrd"]:
            if all(
                metric in df and effective_gate_column(cfg, df, metric, args.imc_default_sample_shape_gate)[0] in df
                for _, cfg, df in group
            ):
                plot_group(out_dir, trace, strategy, metric, group, args.imc_default_sample_shape_gate)
    print(f"wrote {out_dir / 'summary_by_run.csv'}")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
