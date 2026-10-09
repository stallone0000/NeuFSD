#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixed 0-4 no-gate baseline against continuous SFT raw and gated curves."
        )
    )
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument("--continuous-csv", type=Path, required=True)
    parser.add_argument("--trace-label", required=True)
    parser.add_argument("--continuous-label", default="continuous")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--gate-column-prefix",
        choices=["gate", "adaptive"],
        default="gate",
        help="continuous gated curve to draw; fixed baseline is always raw/no-gate.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "minute" not in df.columns:
        raise ValueError(f"missing minute column: {path}")
    return df.sort_values("minute").reset_index(drop=True)


def plot_metric(args: argparse.Namespace, fixed: pd.DataFrame, continuous: pd.DataFrame, metric: str) -> None:
    gate_col = f"{args.gate_column_prefix}_{metric}"
    if metric not in fixed.columns:
        raise ValueError(f"missing fixed {metric}: {args.fixed_csv}")
    for col in [metric, gate_col]:
        if col not in continuous.columns:
            raise ValueError(f"missing continuous {col}: {args.continuous_csv}")

    ylabel = metric.upper()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        fixed["minute"],
        fixed[metric],
        label="fixed 0-4 no gate",
        color="#64748B",
        linewidth=1.35,
    )
    ax.plot(
        continuous["minute"],
        continuous[metric],
        label=f"{args.continuous_label} no gate",
        color="#0F766E",
        linewidth=1.35,
    )
    ax.plot(
        continuous["minute"],
        continuous[gate_col],
        label=f"{args.continuous_label} + gate",
        color="#DC2626",
        linewidth=1.35,
    )
    ax.set_title(f"{args.trace_label}: fixed 0-4 vs continuous {ylabel}")
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / f"{args.stem}_{metric}.png", dpi=args.dpi)
    fig.savefig(args.out_dir / f"{args.stem}_{metric}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fixed = load_metrics(args.fixed_csv)
    continuous = load_metrics(args.continuous_csv)
    for metric in ["wmrd", "mrd"]:
        plot_metric(args, fixed, continuous, metric)
    print(args.out_dir)


if __name__ == "__main__":
    main()
