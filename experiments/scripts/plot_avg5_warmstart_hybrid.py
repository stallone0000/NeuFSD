#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an avg5 warm-start hybrid from selective-window CSVs: use window5 "
            "for the first measured window, then avg5 for steady state."
        )
    )
    parser.add_argument("--selective-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--traces", nargs="+", default=["caida2016", "caida2018"])
    return parser.parse_args()


def read_rows(path: Path) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
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
            rows[int(row["minute"])] = parsed
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(trace: str, rows: list[dict[str, object]], warmup_windows: int) -> dict[str, object]:
    out: dict[str, object] = {
        "trace": trace,
        "strategy": "avg5_warmstart",
        "warmup_windows": warmup_windows,
        "n_windows": len(rows),
        "window5_bootstrap_fraction": float(np.mean([r["hybrid_source"] == "window5_bootstrap" for r in rows])),
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


def plot_trace(out_dir: Path, trace: str, rows: list[dict[str, object]], metric: str) -> None:
    minutes = np.array([int(r["minute"]) for r in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(minutes, [float(r[metric]) for r in rows], label="no gate", linewidth=1.15)
    ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="fixed gate", linewidth=1.15)
    ax.plot(minutes, [float(r[f"selective_{metric}"]) for r in rows], label="selective gate", linewidth=1.35)
    ax.set_title(f"{trace} avg5 warm-start: {metric.upper()} vs time")
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{trace}_avg5_warmstart_{metric}.png", dpi=180)
    fig.savefig(out_dir / f"{trace}_avg5_warmstart_{metric}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for trace in args.traces:
        avg = read_rows(args.selective_dir / f"{trace}_avg5_selective_window_metrics.csv")
        window = read_rows(args.selective_dir / f"{trace}_window5_selective_window_metrics.csv")
        minutes = sorted(avg)
        warmup = set(minutes[: args.warmup_windows])
        rows: list[dict[str, object]] = []
        for minute in minutes:
            row = dict(window[minute] if minute in warmup else avg[minute])
            row["hybrid_source"] = "window5_bootstrap" if minute in warmup else "avg5_steady"
            rows.append(row)
        write_csv(args.out_dir / f"{trace}_avg5_warmstart_window_metrics.csv", rows)
        summaries.append(summarize(trace, rows, args.warmup_windows))
        for metric in ["wmrd", "mrd"]:
            plot_trace(args.out_dir, trace, rows, metric)
    write_csv(args.out_dir / "avg5_warmstart_summary.csv", summaries)
    print(args.out_dir / "avg5_warmstart_summary.csv")


if __name__ == "__main__":
    main()
