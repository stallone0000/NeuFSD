#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
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
METRICS = ("wmrd", "mrd")
CAIDA2016_SCALE_THRESHOLD = 1.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-ready original-vs-residual-calibrated plots for the "
            "default Avg-5 online SFT strategy."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="fourdataset_s2000/final_e5 root")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zoom-half-width", type=int, default=8)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Fail if a calibrated curve does not reduce mean error.",
    )
    return parser.parse_args()


def apply_paper_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def run_path(root: Path, trace: str) -> Path:
    if trace == "caida_2016":
        return root / "caida2016_avg5_e5_s2000_newpre" / "window_metrics.csv"
    if trace == "caida_2018":
        return root / "caida2018_avg5_e5_s2000_newpre" / "window_metrics.csv"
    if trace == "imc":
        return root / "imc_avg5_e5_s2000_newpre_imcdefault" / "window_metrics.csv"
    if trace == "mawi":
        return root / "mawi_gate_w0p30_20260628" / "mawi_avg5_w0p30" / "window_metrics.csv"
    raise ValueError(trace)


def cumulative_median(values: np.ndarray) -> np.ndarray:
    hist: list[float] = []
    out: list[float] = []
    for value in values:
        hist.append(float(value))
        out.append(float(statistics.median(hist)))
    return np.asarray(out, dtype=float)


def caida2016_selective_gate(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, str]:
    heavy_median = cumulative_median(df["heavy_frac"].astype(float).to_numpy())
    scale = df["raw_scale_mean"].astype(float).to_numpy()
    use_gate = (heavy_median >= 0.50) | (scale >= CAIDA2016_SCALE_THRESHOLD)
    values = np.where(use_gate, df[f"gate_{metric}"].astype(float), df[metric].astype(float))
    return values, "selective residual calibration"


def calibrated_values(df: pd.DataFrame, trace: str, metric: str) -> tuple[np.ndarray, str]:
    if trace == "caida_2016":
        return caida2016_selective_gate(df, metric)
    if trace == "caida_2018":
        return df[f"gate_{metric}"].astype(float).to_numpy(), "residual-mass calibration"
    if trace in {"imc", "mawi"}:
        return df[f"sample_shape_{metric}"].astype(float).to_numpy(), "sample-shape residual calibration"
    raise ValueError(trace)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    series = pd.Series(values.astype(float))
    return series.rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def zoom_slice(raw: np.ndarray, calibrated: np.ndarray, half_width: int) -> slice:
    improvement = raw - calibrated
    if float(np.max(improvement)) > 0:
        center = int(np.argmax(improvement))
    else:
        center = int(np.argmax(raw))
    start = max(0, center - half_width)
    stop = min(len(raw), center + half_width + 1)
    return slice(start, stop)


def panel_zoom_half_width(trace: str, metric: str, default: int) -> int:
    if (trace, metric) == ("caida_2018", "wmrd"):
        return max(default, 28)
    return default


def save_legend(out_dir: Path) -> None:
    handles = [
        Line2D([0], [0], color="C0", linewidth=5.2, linestyle="-", label="Original"),
        Line2D([0], [0], color="C1", linewidth=5.2, linestyle="-", label="Calibrated"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 0.82))
    ax.axis("off")
    legend = ax.legend(handles=handles, loc="center", ncol=2, frameon=True, handlelength=2.6)
    plt.setp(legend.get_texts(), fontweight="bold", fontsize=18)
    fig.savefig(out_dir / "legend_residual_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def set_main_ylim(ax: plt.Axes, trace: str, metric: str, raw: np.ndarray, calibrated: np.ndarray, show_inset: bool) -> None:
    values = np.concatenate([raw, calibrated])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = max(1e-9, vmax - vmin)
    if metric == "mrd" and trace in {"caida_2016", "caida_2018"}:
        pad = 0.14 * vrange
        ax.set_ylim(max(0.0, vmin - pad), vmax + pad)
        return
    y_expand = 1.85 if metric == "mrd" and trace in {"imc", "mawi"} else 1.18
    ax.set_ylim(0.0, vmax * y_expand if vmax > 0 else 1.0)


def style_zoom_indicator(indicator: object) -> None:
    def style_artist(artist: object) -> None:
        for method, value in (("set_edgecolor", "C2"), ("set_color", "C2"), ("set_linewidth", 3.0)):
            if hasattr(artist, method):
                try:
                    getattr(artist, method)(value)
                except Exception:
                    pass

    if isinstance(indicator, tuple):
        for item in indicator:
            if isinstance(item, (tuple, list)):
                for subitem in item:
                    style_artist(subitem)
            else:
                style_artist(item)
    else:
        style_artist(indicator)


def plot_one(
    out_dir: Path,
    trace: str,
    metric: str,
    minutes: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    zoom: slice,
) -> None:
    tick_size = 24
    label_size = 31
    line_width = 5.2
    show_inset = not (metric == "mrd" and trace in {"caida_2016", "caida_2018"})

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
    ax.plot(minutes, raw, color="C0", linewidth=line_width, linestyle="-")
    ax.plot(minutes, calibrated, color="C1", linewidth=line_width, linestyle="-")
    xmin = float(np.min(minutes))
    xmax = float(np.max(minutes))
    pad = 0.05 * max(1.0, xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    set_main_ylim(ax, trace, metric, raw, calibrated, show_inset)
    ax.set_xlabel("Time (minute)", fontweight="bold", fontsize=label_size)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="y")
    ax.grid(True, linestyle="--", axis="x")

    if show_inset:
        inset_bounds = [0.12, 0.54, 0.38, 0.38] if (trace, metric) == ("caida_2018", "wmrd") else [0.55, 0.52, 0.40, 0.39]
        inset = ax.inset_axes(inset_bounds)
        z_minutes = minutes[zoom]
        z_raw = raw[zoom]
        z_cal = calibrated[zoom]
        inset.plot(z_minutes, z_raw, color="C0", linewidth=2.4, linestyle="-")
        inset.plot(z_minutes, z_cal, color="C1", linewidth=2.4, linestyle="-")
        inset.set_xlim(float(np.min(z_minutes)), float(np.max(z_minutes)))
        z_min = float(min(np.min(z_raw), np.min(z_cal)))
        z_max = float(max(np.max(z_raw), np.max(z_cal)))
        z_pad = 0.12 * max(1e-9, z_max - z_min)
        inset.set_ylim(max(0.0, z_min - z_pad), z_max + z_pad)
        inset.set_xticks([])
        inset.set_yticks([])
        try:
            indicator = ax.indicate_inset_zoom(inset, edgecolor="C2", linewidth=3.0)
            style_zoom_indicator(indicator)
        except Exception:
            pass

    fig.tight_layout()
    stem = f"{trace}_avg5_{metric}_residual_calibration"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for trace, label in TRACE_LABELS.items():
        path = run_path(args.root, trace)
        df = pd.read_csv(path).sort_values("minute").reset_index(drop=True)
        minutes = df["minute"].astype(int).to_numpy()
        curve_out = pd.DataFrame({"minute": minutes})
        for metric in METRICS:
            raw = df[metric].astype(float).to_numpy()
            calibrated, method = calibrated_values(df, trace, metric)
            zoom = zoom_slice(raw, calibrated, panel_zoom_half_width(trace, metric, args.zoom_half_width))
            plot_one(args.out_dir, trace, metric, minutes, raw, calibrated, zoom)
            curve_out[f"original_{metric}"] = raw
            curve_out[f"calibrated_{metric}"] = calibrated
            row = {
                "trace": trace,
                "trace_label": label,
                "strategy": "avg5",
                "metric": metric.upper(),
                "source_csv": str(path),
                "calibration_method": method,
                "n_windows": int(len(df)),
                "original_mean": float(np.mean(raw)),
                "calibrated_mean": float(np.mean(calibrated)),
                "delta_mean": float(np.mean(calibrated) - np.mean(raw)),
                "original_p95": float(np.percentile(raw, 95)),
                "calibrated_p95": float(np.percentile(calibrated, 95)),
                "original_max": float(np.max(raw)),
                "calibrated_max": float(np.max(calibrated)),
                "burst_start_minute": int(minutes[zoom][0]),
                "burst_end_minute": int(minutes[zoom][-1]),
                "burst_best_reduction": float(np.max(raw - calibrated)),
            }
            if args.require_improvement and not (row["calibrated_mean"] < row["original_mean"]):
                raise RuntimeError(
                    f"{trace} {metric}: calibrated mean {row['calibrated_mean']} "
                    f"is not lower than original mean {row['original_mean']}"
                )
            summary_rows.append(row)
        curve_out["calibration_method"] = calibrated_values(df, trace, "wmrd")[1]
        curve_out.to_csv(curves_dir / f"{trace}_avg5_residual_calibration_curves.csv", index=False)

    save_legend(args.out_dir)
    write_csv(args.out_dir / "summary_residual_calibration.csv", summary_rows)
    print(args.out_dir / "summary_residual_calibration.csv")


if __name__ == "__main__":
    main()
