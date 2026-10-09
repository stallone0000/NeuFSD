#!/usr/bin/env python3
"""Plot sliding-window stability from existing summary_metrics.csv files.

This script does not run training. It joins train/test window definitions from
train_test_name_key.json with per-minute MRD/WMRD summaries and produces
window-level stability plots for the current independent-from-scratch runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


DEFAULT_JSON_ROOT = Path(__file__).resolve().parents[1] / "data_full"
DATASET_RE = re.compile(r"(?:fine_)?dataset_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("zipf_fsd_release"), help="Release root containing results/")
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT, help="Root containing <trace>/train_test_name_key.json")
    parser.add_argument("--out-dir", type=Path, default=Path("zipf_fsd_release/reproduced_figures/window_stability"))
    parser.add_argument("--traces", nargs="+", default=["caida_2016", "caida_2018"])
    parser.add_argument("--res", nargs="+", default=["128_128", "64_64"])
    parser.add_argument("--mode-label", default="independent", help="Label for the plotted run mode")
    return parser.parse_args()


def dataset_minute(name: str) -> int:
    match = DATASET_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse dataset minute from {name!r}")
    return int(match.group(1))


def load_windows(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        items = list(json.load(f).items())

    windows = []
    i = 0
    while i < len(items):
        train, test = [], []
        while i < len(items) and items[i][1]:
            train.append(items[i][0])
            i += 1
        while i < len(items) and not items[i][1]:
            test.append(items[i][0])
            i += 1
        if not train and not test:
            continue
        train_minutes = [dataset_minute(x) for x in train]
        test_minutes = [dataset_minute(x) for x in test]
        windows.append(
            {
                "block_index": len(windows),
                "train_start": min(train_minutes) if train_minutes else None,
                "train_end": max(train_minutes) if train_minutes else None,
                "test_start": min(test_minutes) if test_minutes else None,
                "test_end": max(test_minutes) if test_minutes else None,
                "train_files": train,
                "test_files": test,
                "test_minutes": test_minutes,
            }
        )
    return windows


def load_summary(summary_path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = row["dataset_id"]
            if dataset_id == "OVERALL_AVG":
                continue
            rows[dataset_minute(dataset_id)] = row
    return rows


def find_summary(root: Path, res: str, trace: str) -> Path:
    candidates = [
        root / "results" / f"{res}_{trace}" / "summary_metrics.csv",
        root / "run_full_matrix" / f"{res}_{trace}_final" / "plots" / "pipeline_eval" / "summary_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No summary_metrics.csv found for {res}/{trace}; tried {candidates}")


def metric_stats(values: list[float]) -> tuple[float, float, float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0, min(values), max(values)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_trace_outputs(args: argparse.Namespace, trace: str) -> None:
    json_path = args.json_root / trace / "train_test_name_key.json"
    windows = load_windows(json_path)
    detail_rows = []
    window_rows = []

    for res in args.res:
        summary_path = find_summary(args.root, res, trace)
        metrics_by_minute = load_summary(summary_path)
        print(f"{trace}/{res}: {summary_path}")

        for window in windows:
            per_minute = []
            for minute in window["test_minutes"]:
                metric_row = metrics_by_minute.get(minute)
                if metric_row is None:
                    continue
                row = {
                    "mode": args.mode_label,
                    "trace": trace,
                    "res": res,
                    "block_index": window["block_index"],
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "test_minute": minute,
                    "mrd": float(metric_row["mrd_avg"]),
                    "wmrd": float(metric_row["wmrd_avg"]),
                }
                per_minute.append(row)
                detail_rows.append(row)

            if not per_minute:
                continue
            mrds = [row["mrd"] for row in per_minute]
            wmrds = [row["wmrd"] for row in per_minute]
            mrd_mean, mrd_std, mrd_min, mrd_max = metric_stats(mrds)
            wmrd_mean, wmrd_std, wmrd_min, wmrd_max = metric_stats(wmrds)
            window_rows.append(
                {
                    "mode": args.mode_label,
                    "trace": trace,
                    "res": res,
                    "block_index": window["block_index"],
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "test_count": len(per_minute),
                    "mrd_mean": mrd_mean,
                    "mrd_std": mrd_std,
                    "mrd_min": mrd_min,
                    "mrd_max": mrd_max,
                    "wmrd_mean": wmrd_mean,
                    "wmrd_std": wmrd_std,
                    "wmrd_min": wmrd_min,
                    "wmrd_max": wmrd_max,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / f"{trace}_{args.mode_label}_window_detail.csv",
        detail_rows,
        [
            "mode",
            "trace",
            "res",
            "block_index",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "test_minute",
            "mrd",
            "wmrd",
        ],
    )
    write_csv(
        args.out_dir / f"{trace}_{args.mode_label}_window_summary.csv",
        window_rows,
        [
            "mode",
            "trace",
            "res",
            "block_index",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "test_count",
            "mrd_mean",
            "mrd_std",
            "mrd_min",
            "mrd_max",
            "wmrd_mean",
            "wmrd_std",
            "wmrd_min",
            "wmrd_max",
        ],
    )
    plot_trace(args, trace, window_rows)


def plot_trace(args: argparse.Namespace, trace: str, rows: list[dict]) -> None:
    by_res = defaultdict(list)
    for row in rows:
        by_res[row["res"]].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for res in args.res:
        series = sorted(by_res[res], key=lambda r: r["train_start"])
        if not series:
            continue
        xs = [row["train_start"] for row in series]
        axes[0].errorbar(xs, [row["mrd_mean"] for row in series], yerr=[row["mrd_std"] for row in series], marker="o", capsize=2, label=res)
        axes[1].errorbar(xs, [row["wmrd_mean"] for row in series], yerr=[row["wmrd_std"] for row in series], marker="o", capsize=2, label=res)

    axes[0].set_ylabel("MRD")
    axes[1].set_ylabel("WMRD")
    axes[1].set_xlabel("Training window start minute")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(title="Counter image")
    fig.suptitle(f"{trace}: {args.mode_label} sliding-window test stability")
    fig.tight_layout()
    png = args.out_dir / f"{trace}_{args.mode_label}_window_stability.png"
    pdf = args.out_dir / f"{trace}_{args.mode_label}_window_stability.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png}")


def main() -> None:
    args = parse_args()
    for trace in args.traces:
        build_trace_outputs(args, trace)


if __name__ == "__main__":
    main()
