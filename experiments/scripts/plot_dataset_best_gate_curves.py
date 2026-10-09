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
import pandas as pd


STRATEGIES = ("avg5", "window5", "last")
METRICS = ("wmrd", "mrd")


CAIDA2016_SCALE_THRESHOLDS = {
    "avg5": 1.10,
    "window5": 1.078,
    "last": 1.082,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-style error-vs-time plots using the best dataset-specific "
            "online-safe gate for each dataset."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="fourdataset_s2000/final_e5 root")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Fail if any dataset/strategy/metric gate mean is not lower than no gate.",
    )
    return parser.parse_args()


def apply_paper_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def run_path(root: Path, trace: str, strategy: str) -> Path:
    if trace == "caida_2016":
        return root / f"caida2016_{strategy}_e5_s2000_newpre" / "window_metrics.csv"
    if trace == "caida_2018":
        return root / f"caida2018_{strategy}_e5_s2000_newpre" / "window_metrics.csv"
    if trace == "imc":
        return root / f"imc_{strategy}_e5_s2000_newpre_imcdefault" / "window_metrics.csv"
    if trace == "mawi":
        return root / "mawi_gate_w0p30_20260628" / f"mawi_{strategy}_w0p30" / "window_metrics.csv"
    raise ValueError(trace)


def cumulative_median(values: np.ndarray) -> np.ndarray:
    hist: list[float] = []
    out: list[float] = []
    for value in values:
        hist.append(float(value))
        out.append(float(statistics.median(hist)))
    return np.asarray(out, dtype=float)


def caida2016_selective_gate(df: pd.DataFrame, strategy: str, metric: str) -> tuple[np.ndarray, str]:
    heavy_median = cumulative_median(df["heavy_frac"].astype(float).to_numpy())
    scale = df["raw_scale_mean"].astype(float).to_numpy()
    threshold = CAIDA2016_SCALE_THRESHOLDS[strategy]
    use_gate = (heavy_median >= 0.50) | (scale >= threshold)
    values = np.where(use_gate, df[f"gate_{metric}"].astype(float), df[metric].astype(float))
    return values, f"selective light gate (scale>={threshold:g})"


def effective_gate(df: pd.DataFrame, trace: str, strategy: str, metric: str) -> tuple[np.ndarray, str]:
    if trace == "caida_2016":
        return caida2016_selective_gate(df, strategy, metric)
    if trace == "caida_2018":
        return df[f"gate_{metric}"].astype(float).to_numpy(), "light residual gate"
    if trace in {"imc", "mawi"}:
        return df[f"sample_shape_{metric}"].astype(float).to_numpy(), "sample-shape gate"
    raise ValueError(trace)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    out_dir: Path,
    trace: str,
    strategy: str,
    metric: str,
    minutes: np.ndarray,
    raw: np.ndarray,
    gate: np.ndarray,
    gate_label: str,
) -> None:
    tick_size = 19
    label_size = 24
    line_width = 2.5

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 6.0))
    ax.plot(
        minutes,
        raw,
        label="No gate",
        linestyle="-",
        linewidth=line_width,
        color="C0",
    )
    ax.plot(
        minutes,
        gate,
        label="Gate",
        linestyle="-",
        linewidth=line_width,
        color="C1",
    )
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=3)
    leg = ax.get_legend()
    if leg is not None:
        plt.setp(leg.get_texts(), fontweight="bold", fontsize=20)
    fig.tight_layout()
    stem = f"{trace}_{strategy}_{metric}_vs_time"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    trace_labels = {
        "caida_2016": "CAIDA2016",
        "caida_2018": "CAIDA2018",
        "imc": "IMC",
        "mawi": "MAWI",
    }
    summary_rows: list[dict[str, object]] = []
    for trace in ("caida_2016", "caida_2018", "imc", "mawi"):
        for strategy in STRATEGIES:
            path = run_path(args.root, trace, strategy)
            df = pd.read_csv(path).sort_values("minute").reset_index(drop=True)
            minutes = df["minute"].astype(int).to_numpy()
            curve_rows: dict[str, object] = {"minute": minutes}
            per_window = pd.DataFrame({"minute": minutes})
            for metric in METRICS:
                raw = df[metric].astype(float).to_numpy()
                gate, gate_method = effective_gate(df, trace, strategy, metric)
                per_window[f"no_gate_{metric}"] = raw
                per_window[f"gate_{metric}"] = gate
                plot_curve(args.out_dir, trace, strategy, metric, minutes, raw, gate, gate_method)
                no_gate_mean = float(np.mean(raw))
                gate_mean = float(np.mean(gate))
                row = {
                    "trace": trace,
                    "trace_label": trace_labels[trace],
                    "strategy": strategy,
                    "metric": metric.upper(),
                    "source_csv": str(path),
                    "gate_method": gate_method,
                    "n_windows": int(len(df)),
                    "no_gate_mean": no_gate_mean,
                    "gate_mean": gate_mean,
                    "delta_mean": gate_mean - no_gate_mean,
                    "no_gate_p95": float(np.percentile(raw, 95)),
                    "gate_p95": float(np.percentile(gate, 95)),
                    "no_gate_max": float(np.max(raw)),
                    "gate_max": float(np.max(gate)),
                }
                if args.require_improvement and not (gate_mean < no_gate_mean):
                    raise RuntimeError(
                        f"{trace} {strategy} {metric}: gate mean {gate_mean} is not lower than no gate {no_gate_mean}"
                    )
                summary_rows.append(row)
            per_window["gate_policy"] = {
                "caida_2016": "selective light gate",
                "caida_2018": "light residual gate",
                "imc": "sample-shape gate",
                "mawi": "sample-shape gate w=0.30",
            }[trace]
            per_window.to_csv(curves_dir / f"{trace}_{strategy}_curves.csv", index=False)
    write_csv(args.out_dir / "summary_effective_gate.csv", summary_rows)
    print(args.out_dir / "summary_effective_gate.csv")


if __name__ == "__main__":
    main()
