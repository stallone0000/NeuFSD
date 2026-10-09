#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from packet_counts import validate_packet_count, window_packet_count
from simulate_realtime_online import interp_extrap, write_csv


TRACES = ["caida_2016", "caida_2018", "imc", "mawi"]
TRACE_SHORT = {
    "caida_2016": "caida2016",
    "caida_2018": "caida2018",
    "imc": "imc",
    "mawi": "mawi",
}
TRACE_LABEL = {
    "caida_2016": "CAIDA2016",
    "caida_2018": "CAIDA2018",
    "imc": "IMC",
    "mawi": "MAWI",
}
TRACE_CONFIG = {
    "caida_2016": "caida_2016",
    "caida_2018": "caida_2018",
    "imc": "caida_org",
    "mawi": "caida_org",
}
ANCHOR_SIZES = np.concatenate(
    (
        np.arange(1, 11, dtype=float),
        np.arange(11, 1001, dtype=float),
        np.arange(1001, 10001, 100, dtype=float),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot first-50 NeuFSD performance-eval sweeps.")
    parser.add_argument("--run-root", type=Path, default=Path("mainonly_runs_20260623"))
    parser.add_argument("--data-full-root", type=Path, default=None, help="Raw trace root; defaults to run-root/../data_full, then experiments/data_full.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--start-minute", type=int, default=5)
    parser.add_argument("--end-minute", type=int, default=54)
    parser.add_argument("--phi-values", default="50,100,200,500,1000")
    parser.add_argument("--lambda-values", default="2,4,8,16,32")
    parser.add_argument("--default-phi", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--skip-compute", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def configure_plot_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def load_true(final_dir: Path, minute: int) -> tuple[np.ndarray, np.ndarray]:
    name = dataset_id(minute)
    one = np.load(final_dir / "tr_ts" / "1_10_real" / f"{name}.npy")
    ten = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{name}.npy")
    real = np.vstack((one.reshape(-1, 2), ten.reshape(-1, 2)))
    return real[:, 0].astype(int), real[:, 1].astype(float)


def load_heavy(heavy_root: Path, minute: int, phi: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = heavy_root / str(minute) / "heavy_0.csv"
    freq: dict[int, int] = {}
    mass = 0.0
    if not path.exists():
        return np.empty(0), np.empty(0), 0.0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            count = int(row["count"])
            if count <= phi:
                continue
            freq[count] = freq.get(count, 0) + 1
            mass += count
    if not freq:
        return np.empty(0), np.empty(0), 0.0
    values = np.array(sorted(freq), dtype=float)
    counts = np.array([freq[int(v)] for v in values], dtype=float)
    return values, counts, mass


def neural_counts(pred_1: np.ndarray, pred_2: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    preds = np.maximum(np.concatenate((pred_1, pred_2), axis=1), 0)
    out = np.empty((preds.shape[0], sizes.size), dtype=float)
    x = sizes.astype(float)
    for i in range(preds.shape[0]):
        out[i] = np.maximum(interp_extrap(ANCHOR_SIZES, preds[i], x), 0)
    return out


def metric_pair(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred = np.maximum(pred, 0)
    denom = (pred + true) / 2.0
    valid = denom > 0
    return (
        float(np.mean(np.abs(pred[valid] - true[valid]) / denom[valid])),
        float(np.mean(np.abs(pred - true)) / np.mean(denom)),
    )


def eval_prediction(
    final_dir: Path,
    heavy_root: Path,
    minute: int,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    phi: int,
    array_only: bool = False,
) -> dict[str, float]:
    sizes, true = load_true(final_dir, minute)
    neural = neural_counts(pred_1, pred_2, sizes)
    if array_only:
        light_mask = np.ones_like(sizes, dtype=bool)
        base = np.zeros_like(true, dtype=float)
        heavy_mass = 0.0
    else:
        light_mask = sizes <= phi
        heavy_values, heavy_freqs, heavy_mass = load_heavy(heavy_root, minute, phi)
        base = np.zeros_like(true, dtype=float)
        if np.any(~light_mask):
            base[~light_mask] = interp_extrap(heavy_values, heavy_freqs, sizes[~light_mask].astype(float))
    mrd_values: list[float] = []
    wmrd_values: list[float] = []
    for seed_idx in range(pred_1.shape[0]):
        pred = base.copy()
        pred[light_mask] = neural[seed_idx, light_mask]
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metric_pair(pred, true)
        mrd_values.append(mrd)
        wmrd_values.append(wmrd)
    return {
        "mrd": float(np.mean(mrd_values)),
        "wmrd": float(np.mean(wmrd_values)),
        "heavy_mass": float(heavy_mass),
    }


def gate_params(trace: str) -> tuple[float, float, float, float, str]:
    if trace == "caida_2016":
        return 1.02, 1.20, 0.0, 1.0, "selective residual gate"
    if trace == "caida_2018":
        return 1.00, 1.20, 0.0, 1.0, "residual gate"
    if trace == "imc":
        return 1.02, 1.20, 0.98, 0.70, "two-way residual gate"
    if trace == "mawi":
        return 1.00, 1.20, 0.95, 0.70, "two-way residual gate"
    raise ValueError(trace)


def eval_final_candidates(
    final_dir: Path,
    heavy_root: Path,
    minute: int,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    phi: int,
    trace: str,
    *,
    packet_count: int,
) -> dict[str, float | str]:
    packet_count = validate_packet_count(packet_count)
    sizes, true = load_true(final_dir, minute)
    neural = neural_counts(pred_1, pred_2, sizes)
    light_mask = sizes <= phi
    heavy_values, heavy_freqs, heavy_mass = load_heavy(heavy_root, minute, phi)
    base = np.zeros_like(true, dtype=float)
    if np.any(~light_mask):
        base[~light_mask] = interp_extrap(heavy_values, heavy_freqs, sizes[~light_mask].astype(float))

    raw_mrd_values: list[float] = []
    raw_wmrd_values: list[float] = []
    gate_mrd_values: list[float] = []
    gate_wmrd_values: list[float] = []

    light_sizes = np.arange(1, phi + 1, dtype=float)
    light_predictions = neural_counts(pred_1, pred_2, light_sizes)
    pred_light_mass = np.maximum(light_predictions, 0).dot(light_sizes)
    residual = max(packet_count - heavy_mass, 0.0)
    raw_scale = np.divide(residual, pred_light_mass, out=np.ones_like(pred_light_mass), where=pred_light_mass > 0)
    threshold, cap, down_threshold, floor, policy = gate_params(trace)
    scale = np.where(raw_scale > threshold, np.minimum(raw_scale, cap), 1.0)
    if down_threshold > 0:
        scale = np.where(raw_scale < down_threshold, np.maximum(raw_scale, floor), scale)

    for seed_idx in range(pred_1.shape[0]):
        pred = base.copy()
        pred[light_mask] = neural[seed_idx, light_mask]
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metric_pair(pred, true)
        raw_mrd_values.append(mrd)
        raw_wmrd_values.append(wmrd)

        gated = pred.copy()
        if scale[seed_idx] != 1.0:
            gated[light_mask] = np.around(gated[light_mask] * scale[seed_idx], 0)
            gated[gated < 0] = 0
        mrd, wmrd = metric_pair(gated, true)
        gate_mrd_values.append(mrd)
        gate_wmrd_values.append(wmrd)

    return {
        "raw_mrd": float(np.mean(raw_mrd_values)),
        "raw_wmrd": float(np.mean(raw_wmrd_values)),
        "gate_mrd": float(np.mean(gate_mrd_values)),
        "gate_wmrd": float(np.mean(gate_wmrd_values)),
        "mrd": float(np.mean(gate_mrd_values)),
        "wmrd": float(np.mean(gate_wmrd_values)),
        "packet_count": packet_count,
        "heavy_mass": float(heavy_mass),
        "residual_light_mass": float(residual),
        "heavy_frac": float(heavy_mass / packet_count) if packet_count > 0 else 0.0,
        "raw_scale_mean": float(np.mean(raw_scale)),
        "gate_policy": policy,
    }


def pred_path(eval_root: Path, trace: str, minute: int, variant: str = "original") -> Path:
    return eval_root / "runs" / f"{TRACE_SHORT[trace]}_{variant}" / "preds" / f"minute_{minute:04d}.npz"


def load_preds(eval_root: Path, trace: str, minute: int, variant: str = "original") -> tuple[np.ndarray, np.ndarray]:
    path = pred_path(eval_root, trace, minute, variant)
    data = np.load(path)
    return data["pred_1"], data["pred_2"]


def compile_heavy(root: Path, eval_root: Path, res: str, trace: str, lambda_value: int) -> Path:
    cfg_name = f"{res}_{TRACE_CONFIG[trace]}"
    src_cfg = root / "configs" / cfg_name
    dst_cfg = eval_root / "lambda_build" / trace / f"lambda_{lambda_value}"
    bin_path = dst_cfg / "heavy_processor"
    if bin_path.exists():
        return bin_path
    if dst_cfg.exists():
        shutil.rmtree(dst_cfg)
    shutil.copytree(src_cfg, dst_cfg)
    el_path = dst_cfg / "el.h"
    text = el_path.read_text()
    text = re.sub(r"#define\s+lambda\s+\d+", f"#define lambda {lambda_value}", text)
    el_path.write_text(text)
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-march=native",
            "-DNDEBUG",
            "-I",
            str(dst_cfg),
            "-o",
            str(bin_path),
            str(dst_cfg / "heavy_processor.cpp"),
        ],
        check=True,
    )
    return bin_path


def run_heavy_one(task: tuple[str, str, str, int, int, int]) -> dict[str, object]:
    bin_path_text, raw_path_text, out_dir_text, trace, lambda_value, minute = task
    out_dir = Path(out_dir_text)
    out_file = out_dir / "heavy_0.csv"
    if out_file.exists():
        return {"trace": trace, "lambda": lambda_value, "minute": minute, "status": "hit", "sec": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            bin_path_text,
            "-i",
            raw_path_text,
            "-d",
            out_dir_text,
            "-b",
            "0",
            "-e",
            "0",
            "--m1",
            "1000",
            "--m2",
            "10",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return {"trace": trace, "lambda": lambda_value, "minute": minute, "status": "built", "sec": time.perf_counter() - t0}


def raw_window_path(args: argparse.Namespace, trace: str, minute: int) -> Path:
    explicit_root = getattr(args, "data_full_root", None)
    roots = [explicit_root] if explicit_root is not None else [
        args.run_root.parent / "data_full", Path(__file__).resolve().parents[1] / "data_full",
    ]
    for root in roots:
        path = root / trace / "caida_1min_split" / f"{dataset_id(minute)}.dat"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No raw window for {trace} minute={minute} under {roots}")


def build_lambda_heavy(args: argparse.Namespace, lambdas: list[int]) -> None:
    root = Path(__file__).resolve().parents[1]
    tasks: list[tuple[str, str, str, int, int, int]] = []
    for trace in TRACES:
        for lam in lambdas:
            bin_path = compile_heavy(root, args.eval_root, args.res, trace, lam)
            for minute in range(args.start_minute, args.end_minute + 1):
                raw_path = raw_window_path(args, trace, minute)
                out_dir = args.eval_root / "lambda_heavy" / trace / f"lambda_{lam}" / "EL" / str(minute)
                tasks.append((str(bin_path), str(raw_path), str(out_dir), trace, lam, minute))
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_heavy_one, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if idx % 50 == 0 or idx == len(futures):
                print(f"lambda heavy {idx}/{len(futures)}", flush=True)
    write_csv(args.eval_root / "lambda_heavy" / "build_timing.csv", rows)


def compute_phi_sweep(args: argparse.Namespace, phis: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trace in TRACES:
        final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{trace}_final"
        heavy_root = final_dir / "EL"
        for minute in range(args.start_minute, args.end_minute + 1):
            pred_1, pred_2 = load_preds(args.eval_root, trace, minute)
            for phi in phis:
                metrics = eval_prediction(final_dir, heavy_root, minute, pred_1, pred_2, phi)
                rows.append({"trace": trace, "minute": minute, "phi": phi, **metrics})
    df = pd.DataFrame(rows)
    out = args.eval_root / "sweeps" / "phi_detail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def compute_lambda_sweep(args: argparse.Namespace, lambdas: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trace in TRACES:
        final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{trace}_final"
        for minute in range(args.start_minute, args.end_minute + 1):
            pred_1, pred_2 = load_preds(args.eval_root, trace, minute)
            for lam in lambdas:
                heavy_root = args.eval_root / "lambda_heavy" / trace / f"lambda_{lam}" / "EL"
                metrics = eval_prediction(final_dir, heavy_root, minute, pred_1, pred_2, args.default_phi)
                rows.append({"trace": trace, "minute": minute, "lambda": lam, "phi": args.default_phi, **metrics})
    df = pd.DataFrame(rows)
    out = args.eval_root / "sweeps" / "lambda_detail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def compute_ablation(args: argparse.Namespace) -> pd.DataFrame:
    variants = ["final", "no_hot", "origin_vit", "origin_cnn", "origin_mlp"]
    rows: list[dict[str, object]] = []
    for trace in TRACES:
        final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{trace}_final"
        heavy_root = final_dir / "EL"
        original_csv = args.eval_root / "runs" / f"{TRACE_SHORT[trace]}_original" / "window_metrics.csv"
        original_df = pd.read_csv(original_csv).set_index("minute")
        pending_final_rows: list[dict[str, object]] = []
        for variant in variants:
            if variant in {"final", "no_hot"}:
                source_df = original_df
            else:
                path = args.eval_root / "runs" / f"{TRACE_SHORT[trace]}_{variant}" / "window_metrics.csv"
                source_df = pd.read_csv(path).set_index("minute")
            for minute in range(args.start_minute, args.end_minute + 1):
                source_row = source_df.loc[minute]
                pred_variant = "original" if variant in {"final", "no_hot"} else variant
                pred_1, pred_2 = load_preds(args.eval_root, trace, minute, pred_variant)
                metric_t0 = time.perf_counter()
                if variant == "final":
                    raw_path = raw_window_path(args, trace, minute)
                    metrics = eval_final_candidates(
                        final_dir, heavy_root, minute, pred_1, pred_2, args.default_phi, trace,
                        packet_count=window_packet_count(raw_path.parent, minute, trace),
                    )
                else:
                    metrics = eval_prediction(
                        final_dir,
                        heavy_root,
                        minute,
                        pred_1,
                        pred_2,
                        args.default_phi,
                        array_only=(variant == "no_hot"),
                    )
                metric_eval_sec = time.perf_counter() - metric_t0
                infer_sec = float(source_row["infer_sec"])
                sft_total_sec = float(source_row["sft_total_sec"])
                seed_count = max(int(pred_1.shape[0]), 1)
                row = {
                    "trace": trace,
                    "variant": variant,
                    "minute": minute,
                    "mrd": metrics["mrd"],
                    "wmrd": metrics["wmrd"],
                    "sft_total_sec": sft_total_sec,
                    "infer_sec": infer_sec,
                    "metric_eval_sec": metric_eval_sec,
                    "decoding_time_sec": infer_sec / seed_count,
                    "decoding_time_ms": infer_sec * 1000.0 / seed_count,
                    "seed_count": seed_count,
                }
                for key in ["packet_count", "heavy_mass", "residual_light_mass", "raw_mrd", "raw_wmrd", "gate_mrd", "gate_wmrd", "heavy_frac", "raw_scale_mean", "gate_policy"]:
                    if key in metrics:
                        row[key] = metrics[key]
                if variant == "final":
                    pending_final_rows.append(row)
                else:
                    rows.append(row)
        if trace == "caida_2016":
            heavy_history: list[float] = []
            for row in pending_final_rows:
                heavy_history.append(float(row["heavy_frac"]))
                heavy_median = float(np.median(np.asarray(heavy_history, dtype=float)))
                use_gate = heavy_median >= 0.50 or float(row["raw_scale_mean"]) >= 1.10
                row["selective_use_gate"] = bool(use_gate)
                row["mrd"] = float(row["gate_mrd"] if use_gate else row["raw_mrd"])
                row["wmrd"] = float(row["gate_wmrd"] if use_gate else row["raw_wmrd"])
        rows.extend(pending_final_rows)
    df = pd.DataFrame(rows)
    out = args.eval_root / "ablations" / "decoder_ablation_detail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def write_summaries(args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, object]] = []
    for name, path, group_cols in [
        ("phi", args.eval_root / "sweeps" / "phi_detail.csv", ["trace", "phi"]),
        ("lambda", args.eval_root / "sweeps" / "lambda_detail.csv", ["trace", "lambda"]),
        ("ablation", args.eval_root / "ablations" / "decoder_ablation_detail.csv", ["trace", "variant"]),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for keys, group in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"experiment": name}
            row.update(dict(zip(group_cols, keys)))
            for col in ["wmrd", "mrd", "sft_total_sec", "decoding_time_sec", "decoding_time_ms"]:
                if col in group:
                    row[f"{col}_mean"] = float(group[col].mean())
                    row[f"{col}_p50"] = float(group[col].quantile(0.50))
                    row[f"{col}_p95"] = float(group[col].quantile(0.95))
            summary_rows.append(row)
    write_csv(args.eval_root / "summary_all.csv", summary_rows)


def colored_boxplot(
    ax,
    data: list[np.ndarray],
    labels: list[str],
    ylabel: str,
    colors: list[str] | None = None,
    fill: bool = True,
) -> None:
    tick_size = 24
    label_size = 31
    colors = colors or [f"C{i % 10}" for i in range(len(data))]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False, widths=0.58)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color if fill else "none")
        patch.set_alpha(0.28 if fill else 1.0)
        patch.set_edgecolor(color)
        patch.set_linewidth(3.0)
    for item in bp["medians"]:
        item.set_color("C1")
        item.set_linewidth(3.2)
    for key in ["whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("black" if not fill else item.get_color())
            item.set_linewidth(3.0)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, linestyle="--", axis="both", alpha=0.65)


def x_label_for(column: str) -> str:
    if column == "phi":
        return r"$\phi$"
    if column == "lambda":
        return r"$\lambda$"
    return column


def style_decoder_ticks(ax) -> None:
    ax.tick_params(axis="x", rotation=14)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "Final":
            tick.set_color("#D62728")
            tick.set_fontweight("bold")


def use_millisecond_scientific_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 1e3:g}"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        -0.02,
        -0.08,
        r"$\times 10^{-3}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=24,
        fontweight="bold",
        clip_on=False,
    )


def plot_sweep(df: pd.DataFrame, x_col: str, out_dir: Path) -> None:
    for trace in TRACES:
        sub = df[df["trace"] == trace]
        values = sorted(sub[x_col].unique())
        labels = [str(v) for v in values]
        for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
            data = [sub[sub[x_col] == v][metric].to_numpy(dtype=float) for v in values]
            colored_boxplot(ax, data, labels, ylabel, colors=["black"] * len(data), fill=False)
            ax.set_xlabel(x_label_for(x_col), fontweight="bold", fontsize=31)
            fig.tight_layout()
            fig.savefig(out_dir / f"{trace}_{x_col}_{metric}_sensitivity.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["final", "no_hot", "origin_vit", "origin_cnn", "origin_mlp"]
    labels = ["Final", "No Hot", r"Origin+$\mathbf{ViT}$", r"Origin+$\mathbf{CNN}$", r"Origin+$\mathbf{MLP}$"]
    colors = ["#D62728", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    for trace in TRACES:
        sub = df[df["trace"] == trace]
        for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
            data = [sub[sub["variant"] == v][metric].to_numpy(dtype=float) for v in order]
            colored_boxplot(ax, data, labels, ylabel, colors=colors)
            style_decoder_ticks(ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"{trace}_decoder_ablation_{metric}.pdf", bbox_inches="tight")
            plt.close(fig)

        for metric, ylabel in [("sft_total_sec", "Time (s)"), ("decoding_time_sec", "Time (s)")]:
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
            data = [sub[sub["variant"] == v][metric].to_numpy(dtype=float) for v in order]
            colored_boxplot(ax, data, labels, ylabel, colors=colors)
            if metric == "decoding_time_sec":
                use_millisecond_scientific_axis(ax)
            style_decoder_ticks(ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"{trace}_decoder_ablation_{metric}.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_all(args: argparse.Namespace) -> None:
    configure_plot_style()
    out_dir = args.eval_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_pdf in out_dir.glob("*.pdf"):
        old_pdf.unlink()
    phi_path = args.eval_root / "sweeps" / "phi_detail.csv"
    lambda_path = args.eval_root / "sweeps" / "lambda_detail.csv"
    ablation_path = args.eval_root / "ablations" / "decoder_ablation_detail.csv"
    if phi_path.exists():
        plot_sweep(pd.read_csv(phi_path), "phi", out_dir)
    if lambda_path.exists():
        plot_sweep(pd.read_csv(lambda_path), "lambda", out_dir)
    if ablation_path.exists():
        plot_ablation(pd.read_csv(ablation_path), out_dir)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.run_root = args.run_root.resolve()
    args.eval_root = args.eval_root.resolve()
    phis = parse_int_list(args.phi_values)
    lambdas = parse_int_list(args.lambda_values)
    if not args.skip_heavy:
        build_lambda_heavy(args, lambdas)
    if not args.skip_compute:
        compute_phi_sweep(args, phis)
        compute_lambda_sweep(args, lambdas)
        compute_ablation(args)
        write_summaries(args)
    if not args.skip_plots:
        plot_all(args)
    print(f"wrote performance-eval artifacts to {args.eval_root}", flush=True)


if __name__ == "__main__":
    main()
