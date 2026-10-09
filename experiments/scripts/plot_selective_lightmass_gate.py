#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    ("caida2016", "last", "caida2016_last_e20"),
    ("caida2016", "window5", "caida2016_window5_e5"),
    ("caida2016", "avg5", "caida2016_avg5_e20"),
    ("caida2018", "last", "caida2018_last_e20"),
    ("caida2018", "window5", "caida2018_window5_e5_tune"),
    ("caida2018", "avg5", "caida2018_avg5_e20"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a selective light-residual gate from existing window_metrics.csv files. "
            "The rule is online-safe: enable the original gate in heavy-dominant regimes, "
            "or for light-mass underprediction spikes."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--heavy-median-threshold", type=float, default=0.50)
    parser.add_argument("--scale-spike-threshold", type=float, default=1.08)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {"minute": int(row["minute"])}
            for key, value in row.items():
                if key == "minute":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    rows.sort(key=lambda r: int(r["minute"]))
    return rows


def add_selective_gate(rows: list[dict[str, object]], heavy_threshold: float, scale_threshold: float) -> None:
    heavy_history: list[float] = []
    for row in rows:
        heavy_history.append(float(row["heavy_frac"]))
        heavy_regime = statistics.median(heavy_history) >= heavy_threshold
        scale_spike = float(row["raw_scale_mean"]) >= scale_threshold
        use_gate = heavy_regime or scale_spike
        row["selective_use_gate"] = bool(use_gate)
        for metric in ["wmrd", "mrd"]:
            row[f"selective_{metric}"] = float(row[f"gate_{metric}"] if use_gate else row[metric])


def summarize(trace: str, strategy: str, run_dir: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "trace": trace,
        "strategy": strategy,
        "run_dir": str(run_dir),
        "n_windows": len(rows),
        "selective_use_gate_fraction": float(np.mean([bool(r["selective_use_gate"]) for r in rows])),
    }
    for metric in ["wmrd", "mrd"]:
        for prefix in ["", "gate_", "selective_"]:
            key = f"{prefix}{metric}"
            values = np.array([float(r[key]) for r in rows], dtype=float)
            label = "raw" if prefix == "" else prefix[:-1]
            out[f"{label}_{metric}_mean"] = float(np.mean(values))
            out[f"{label}_{metric}_p95"] = float(np.percentile(values, 95))
            out[f"{label}_{metric}_max"] = float(np.max(values))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_run(out_dir: Path, trace: str, strategy: str, rows: list[dict[str, object]], metric: str) -> None:
    minutes = np.array([int(r["minute"]) for r in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(minutes, [float(r[metric]) for r in rows], label="no gate", linewidth=1.15)
    ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="fixed gate", linewidth=1.15)
    ax.plot(minutes, [float(r[f"selective_{metric}"]) for r in rows], label="selective gate", linewidth=1.35)
    ax.set_title(f"{trace} {strategy}: {metric.upper()} vs time")
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{trace}_{strategy}_{metric}_selective_gate.png", dpi=180)
    fig.savefig(out_dir / f"{trace}_{strategy}_{metric}_selective_gate.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for trace, strategy, run_name in DEFAULT_RUNS:
        run_dir = args.run_root / run_name
        rows = read_rows(run_dir / "window_metrics.csv")
        add_selective_gate(rows, args.heavy_median_threshold, args.scale_spike_threshold)
        summaries.append(summarize(trace, strategy, run_dir, rows))
        write_csv(args.out_dir / f"{trace}_{strategy}_selective_window_metrics.csv", rows)
        for metric in ["wmrd", "mrd"]:
            plot_run(args.out_dir, trace, strategy, rows, metric)
    write_csv(args.out_dir / "selective_gate_summary.csv", summaries)
    print(args.out_dir / "selective_gate_summary.csv")


if __name__ == "__main__":
    main()
