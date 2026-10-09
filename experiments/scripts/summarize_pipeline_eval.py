#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from packet_counts import count_raw_packets, validate_packet_count


TARGET_FREQS = np.concatenate((np.arange(1, 1001, 1), np.arange(1001, 10001, 100)))
SPLICE_POINT = 100
PLOT_SAVE_DIR = Path("plots/pipeline_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast MRAC pipeline summary and minute-error curves.")
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--packet-size", type=int)
    parser.add_argument(
        "--light-mass-correction",
        choices=["none", "up-only"],
        default="none",
        help="Optional online-safe correction for the neural light FSD using packet-count residual after the heavy part.",
    )
    parser.add_argument(
        "--light-mass-threshold",
        type=float,
        default=1.02,
        help="Apply up-only light-mass correction only when residual/predicted light packet mass exceeds this value.",
    )
    parser.add_argument(
        "--light-mass-cap",
        type=float,
        default=1.20,
        help="Maximum multiplicative correction for light bins.",
    )
    return parser.parse_args()


def task_mapping(final_dir: Path) -> dict[str, str]:
    with (final_dir / "train_test_name_key.json").open() as f:
        tasks = json.load(f)
    mapping: dict[str, str] = {}
    current_train_block = None
    in_train_block = False
    for filename, is_train in tasks.items():
        ds_name = Path(filename).stem
        if is_train:
            if not in_train_block:
                current_train_block = ds_name
                in_train_block = True
        else:
            in_train_block = False
            if current_train_block is not None:
                mapping[ds_name] = current_train_block
    return mapping


def load_el_freq(final_dir: Path, dataset_id: str) -> np.ndarray:
    el_folder_idx = str(int(dataset_id.split("_")[-1]))
    df = pd.read_csv(final_dir / "EL" / el_folder_idx / "heavy_0.csv")
    count_nums = df["count"].to_numpy()
    count_nums = count_nums[count_nums > 99]
    count_nums.sort()
    if count_nums.size == 0:
        return np.empty((0, 2), dtype=float)
    values, frequencies = np.unique(count_nums, return_counts=True)
    return np.column_stack((values, frequencies)).astype(float)


def large_el_predictions(freq: np.ndarray, flow_sizes: np.ndarray) -> np.ndarray:
    if flow_sizes.size == 0:
        return np.empty(0, dtype=float)
    if freq.size == 0:
        return np.zeros(flow_sizes.size, dtype=float)
    if len(freq) == 1:
        return np.full(flow_sizes.size, freq[0, 1], dtype=float)
    interp = interp1d(freq[:, 0], freq[:, 1], kind="linear", fill_value="extrapolate")
    return np.asarray(interp(flow_sizes), dtype=float)


def compute_dataset(
    final_dir: Path,
    dataset_id: str,
    block_name: str,
    light_mass_correction: str = "none",
    light_mass_threshold: float = 1.05,
    light_mass_cap: float = 1.25,
    *,
    packet_count: int | None = None,
) -> tuple[float, float]:
    root = final_dir / "0_finetuned_results"
    pred_1 = np.load(root / "ViT_1_10_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy")
    pred_2 = np.load(root / "ViT_10_1e4_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy")
    corr_preds = np.column_stack((pred_1, pred_2))

    real_1 = np.load(final_dir / "tr_ts" / "1_10_real" / f"{dataset_id}.npy")
    real_2 = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{dataset_id}.npy")
    real = np.vstack((real_1, real_2))
    flow_sizes = real[:, 0].astype(int)
    true_counts = real[:, 1].astype(float)

    small_mask = flow_sizes <= SPLICE_POINT
    large_mask = ~small_mask
    small_sizes = flow_sizes[small_mask]
    base_preds = np.zeros(flow_sizes.shape, dtype=float)
    el_freq = load_el_freq(final_dir, dataset_id)
    base_preds[large_mask] = large_el_predictions(el_freq, flow_sizes[large_mask])

    correction_scale = 1.0
    if light_mass_correction == "up-only":
        if packet_count is None:
            raise ValueError("Light-mass correction requires the raw window packet count")
        packet_count = validate_packet_count(packet_count)
        heavy_packet_mass = float(np.sum(el_freq[:, 0] * el_freq[:, 1])) if el_freq.size else 0.0
        residual_light_mass = max(packet_count - heavy_packet_mass, 0.0)

    mrd_values: list[float] = []
    wmrd_values: list[float] = []
    for sep_preds in corr_preds:
        sep_preds = np.maximum(sep_preds, 0)
        final_preds = base_preds.copy()
        if small_sizes.size:
            final_preds[small_mask] = np.interp(small_sizes, TARGET_FREQS, sep_preds)

        if light_mass_correction == "up-only" and small_sizes.size:
            light_sizes = np.arange(1, SPLICE_POINT + 1)
            light_preds = np.interp(light_sizes, TARGET_FREQS, sep_preds)
            light_mass = float(np.sum(light_sizes * np.maximum(light_preds, 0)))
            if light_mass > 0:
                raw_scale = residual_light_mass / light_mass
                if raw_scale > light_mass_threshold:
                    correction_scale = min(raw_scale, light_mass_cap)
                    final_preds[small_mask] *= correction_scale

        final_preds = np.around(final_preds, 0)
        final_preds[final_preds < 0] = 0
        denom = (final_preds + true_counts) / 2
        mrd_values.append(float(np.around(np.mean(np.abs(final_preds - true_counts) / denom), 6)))
        wmrd_values.append(float(np.around(np.mean(np.abs(final_preds - true_counts)) / np.mean(denom), 6)))

    return float(np.mean(mrd_values)), float(np.mean(wmrd_values))


def write_plots(out_dir: Path, minute_rows: list[tuple[int, float, float]]) -> None:
    minutes = np.array([row[0] for row in minute_rows], dtype=int)
    mrds = np.array([row[1] for row in minute_rows], dtype=float)
    wmrds = np.array([row[2] for row in minute_rows], dtype=float)

    for name, values, ylabel in [
        ("mrd_vs_minute.png", mrds, "MRD"),
        ("wmrd_vs_minute.png", wmrds, "WMRD"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(minutes, values, linewidth=1.4)
        ax.set_xlabel("Minute")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(minutes, mrds, linewidth=1.4)
    axes[0].set_ylabel("MRD")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(minutes, wmrds, linewidth=1.4)
    axes[1].set_xlabel("Minute")
    axes[1].set_ylabel("WMRD")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "minute_error_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.light_mass_correction != "none" and (args.raw_dir is None or args.packet_size is None):
        raise ValueError("Light-mass correction requires --raw-dir and --packet-size")
    final_dir = args.final_dir.resolve()
    if args.light_mass_correction == "none":
        out_dir = final_dir / PLOT_SAVE_DIR
    else:
        suffix = f"{args.light_mass_correction}_thr{args.light_mass_threshold:g}_cap{args.light_mass_cap:g}"
        suffix = suffix.replace(".", "p")
        out_dir = final_dir / f"plots/pipeline_eval_lightmass_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float]] = []
    minute_rows: list[tuple[int, float, float]] = []
    for dataset_id, block_name in task_mapping(final_dir).items():
        mrd, wmrd = compute_dataset(
            final_dir,
            dataset_id,
            block_name,
            light_mass_correction=args.light_mass_correction,
            light_mass_threshold=args.light_mass_threshold,
            light_mass_cap=args.light_mass_cap,
            packet_count=(
                count_raw_packets(args.raw_dir / f"{dataset_id}.dat", args.packet_size)
                if args.light_mass_correction != "none" else None
            ),
        )
        rows.append((dataset_id, mrd, wmrd))
        minute_rows.append((int(dataset_id.split("_")[-1]), mrd, wmrd))

    if not rows:
        raise RuntimeError(f"no test datasets found in {final_dir}")

    with (out_dir / "summary_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_id", "mrd_avg", "wmrd_avg"])
        writer.writerows(rows)
        writer.writerow(["OVERALL_AVG", np.mean([row[1] for row in rows]), np.mean([row[2] for row in rows])])

    with (out_dir / "minute_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["minute", "dataset_id", "mrd_avg", "wmrd_avg"])
        for (dataset_id, mrd, wmrd), (minute, _, _) in zip(rows, minute_rows):
            writer.writerow([minute, dataset_id, mrd, wmrd])

    write_plots(out_dir, minute_rows)
    print(f"{final_dir}: MRD={np.mean([row[1] for row in rows]):.6f} WMRD={np.mean([row[2] for row in rows]):.6f}")


if __name__ == "__main__":
    main()
