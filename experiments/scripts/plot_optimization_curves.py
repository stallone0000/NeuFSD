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
from scipy.interpolate import interp1d

from packet_counts import validate_packet_count, window_packet_count


TARGET_FREQS = np.concatenate((np.arange(1, 1001, 1), np.arange(1001, 10001, 100)))
SPLICE_POINT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot original/gated/final-active optimization curves.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--packet-size", type=int, default=0)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--traces", nargs="+", default=["caida_2018", "caida_2018_new"])
    parser.add_argument("--base-mode", default="pretrain_continuous")
    parser.add_argument("--final-active-mode", default="pretrain_continuous_final_active_loss")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=1.02)
    parser.add_argument("--cap", type=float, default=1.20)
    parser.add_argument(
        "--focus",
        nargs="*",
        default=[],
        help="Optional trace:start-end filters, e.g. caida_2018:120-180 caida_2018_new:60-90.",
    )
    return parser.parse_args()


def parse_focus(items: list[str]) -> dict[str, list[tuple[int, int]]]:
    focus: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        trace, _, ranges_text = item.partition(":")
        ranges = []
        for part in ranges_text.split(","):
            lo_text, _, hi_text = part.partition("-")
            lo = int(lo_text)
            hi = int(hi_text or lo_text)
            ranges.append((lo, hi))
        focus[trace] = ranges
    return focus


def in_focus(focus: dict[str, list[tuple[int, int]]], trace: str, minute: int) -> bool:
    ranges = focus.get(trace)
    if not ranges:
        return True
    return any(lo <= minute <= hi for lo, hi in ranges)


def grouped_tasks(final_dir: Path) -> dict[str, list[str]]:
    with (final_dir / "train_test_name_key.json").open() as f:
        items = list(json.load(f).items())
    mapping: dict[str, list[str]] = {}
    i = 0
    while i < len(items):
        train: list[str] = []
        test: list[str] = []
        while i < len(items) and items[i][1]:
            train.append(Path(items[i][0]).stem)
            i += 1
        while i < len(items) and not items[i][1]:
            test.append(Path(items[i][0]).stem)
            i += 1
        for dataset_id in test:
            mapping[dataset_id] = train
    return mapping


def load_el_freq(final_dir: Path, dataset_id: str) -> np.ndarray:
    el_folder_idx = str(int(dataset_id.split("_")[-1]))
    df = pd.read_csv(final_dir / "EL" / el_folder_idx / "heavy_0.csv")
    counts = df["count"].to_numpy()
    counts = counts[counts > 99]
    counts.sort()
    if counts.size == 0:
        return np.empty((0, 2), dtype=float)
    values, freqs = np.unique(counts, return_counts=True)
    return np.column_stack((values, freqs)).astype(float)


def large_el_predictions(freq: np.ndarray, flow_sizes: np.ndarray) -> np.ndarray:
    if flow_sizes.size == 0:
        return np.empty(0, dtype=float)
    if freq.size == 0:
        return np.zeros(flow_sizes.size, dtype=float)
    if len(freq) == 1:
        return np.full(flow_sizes.size, freq[0, 1], dtype=float)
    interp = interp1d(freq[:, 0], freq[:, 1], kind="linear", fill_value="extrapolate")
    return np.asarray(interp(flow_sizes), dtype=float)


def load_true(final_dir: Path, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    real_1 = np.load(final_dir / "tr_ts" / "1_10_real" / f"{dataset_id}.npy")
    real_2 = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{dataset_id}.npy")
    real = np.vstack((real_1.reshape(-1, 2), real_2.reshape(-1, 2)))
    return real[:, 0].astype(int), real[:, 1].astype(float)


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred = np.maximum(pred, 0)
    denom = (pred + true) / 2
    valid = denom > 0
    mrd = float(np.mean(np.abs(pred[valid] - true[valid]) / denom[valid]))
    wmrd = float(np.mean(np.abs(pred - true)) / np.mean(denom))
    return mrd, wmrd


def compute_dataset(
    final_dir: Path, dataset_id: str, block_name: str, threshold: float, cap: float,
    *, packet_count: int,
) -> dict[str, float]:
    root = final_dir / "0_finetuned_results"
    pred_1_path = root / "ViT_1_10_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy"
    pred_2_path = root / "ViT_10_1e4_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy"
    if not pred_1_path.exists() or not pred_2_path.exists():
        raise FileNotFoundError(f"missing predictions for {dataset_id} block={block_name} under {final_dir}")
    pred_1 = np.load(pred_1_path)
    pred_2 = np.load(pred_2_path)
    corr_preds = np.column_stack((pred_1, pred_2))

    flow_sizes, true_counts = load_true(final_dir, dataset_id)
    small_mask = flow_sizes <= SPLICE_POINT
    large_mask = ~small_mask
    small_sizes = flow_sizes[small_mask]
    el_freq = load_el_freq(final_dir, dataset_id)
    base_large = np.zeros(flow_sizes.shape, dtype=float)
    base_large[large_mask] = large_el_predictions(el_freq, flow_sizes[large_mask])

    packet_count = validate_packet_count(packet_count)
    heavy_packet_mass = float(np.sum(el_freq[:, 0] * el_freq[:, 1])) if el_freq.size else 0.0
    residual_light_mass = max(packet_count - heavy_packet_mass, 0.0)

    raw_mrd, raw_wmrd, gated_mrd, gated_wmrd, scales = [], [], [], [], []
    for sep_preds in corr_preds:
        sep_preds = np.maximum(sep_preds, 0)
        pred = base_large.copy()
        if small_sizes.size:
            pred[small_mask] = np.interp(small_sizes, TARGET_FREQS, sep_preds)
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metrics(pred, true_counts)
        raw_mrd.append(mrd)
        raw_wmrd.append(wmrd)

        gated = pred.copy()
        light_sizes = np.arange(1, SPLICE_POINT + 1)
        light_preds = np.interp(light_sizes, TARGET_FREQS, sep_preds)
        light_mass = float(np.sum(light_sizes * np.maximum(light_preds, 0)))
        scale = residual_light_mass / light_mass if light_mass > 0 else 1.0
        scales.append(scale)
        if scale > threshold and small_sizes.size:
            gated[small_mask] = np.around(gated[small_mask] * min(scale, cap), 0)
        mrd, wmrd = metrics(gated, true_counts)
        gated_mrd.append(mrd)
        gated_wmrd.append(wmrd)

    return {
        "raw_mrd": float(np.mean(raw_mrd)),
        "raw_wmrd": float(np.mean(raw_wmrd)),
        "gated_mrd": float(np.mean(gated_mrd)),
        "gated_wmrd": float(np.mean(gated_wmrd)),
        "scale": float(np.mean(scales)),
    }


def load_raw_summary(final_dir: Path) -> dict[int, tuple[float, float]]:
    path = final_dir / "plots" / "pipeline_eval" / "summary_metrics.csv"
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["dataset_id"] == "OVERALL_AVG":
                continue
            minute = int(row["dataset_id"].split("_")[-1])
            out[minute] = (float(row["mrd_avg"]), float(row["wmrd_avg"]))
    return out


def compute_curves(
    final_dir: Path,
    trace: str,
    threshold: float,
    cap: float,
    focus: dict[str, list[tuple[int, int]]],
    *,
    raw_dir: Path,
    packet_size: int = 0,
) -> list[dict[str, float | int]]:
    mapping = grouped_tasks(final_dir)
    rows = []
    for dataset_id, train_names in mapping.items():
        minute = int(dataset_id.split("_")[-1])
        if not in_focus(focus, trace, minute):
            continue
        packet_count = window_packet_count(raw_dir, minute, trace, packet_size)
        try:
            values = compute_dataset(
                final_dir, dataset_id, train_names[0], threshold, cap, packet_count=packet_count,
            )
        except FileNotFoundError:
            continue
        rows.append({"minute": minute, **values})
    return sorted(rows, key=lambda r: int(r["minute"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(out_dir: Path, trace: str, rows: list[dict[str, object]]) -> None:
    minutes = np.array([int(r["minute"]) for r in rows])
    series = [
        ("original", np.array([float(r["original_wmrd"]) for r in rows])),
        ("gated", np.array([float(r["gated_wmrd"]) for r in rows])),
        ("gated + final-active loss", np.array([float(r["final_active_gated_wmrd"]) for r in rows])),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, values in series:
        mask = np.isfinite(values)
        ax.plot(minutes[mask], values[mask], linewidth=1.6, label=label)
    if trace == "caida_2018":
        for lo, hi in [(133, 139), (154, 169)]:
            ax.axvspan(lo, hi, color="tab:red", alpha=0.08)
    elif trace == "caida_2018_new":
        ax.axvspan(74, 79, color="tab:red", alpha=0.08)
    ax.set_title(f"{trace} WMRD vs minute")
    ax.set_xlabel("minute")
    ax.set_ylabel("WMRD")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{trace}_wmrd_optimization_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = args.out_dir or (run_root / "plots" / "optimization_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    focus = parse_focus(args.focus)

    for trace in args.traces:
        base_final = run_root / "online" / args.base_mode / f"{args.res}_{trace}_final"
        final_active_final = run_root / "online" / args.final_active_mode / f"{args.res}_{trace}_final"
        raw_dir = args.data_full_root / trace / "caida_1min_split"
        base_rows = compute_curves(
            base_final, trace, args.threshold, args.cap, focus,
            raw_dir=raw_dir, packet_size=args.packet_size,
        )
        fa_rows = compute_curves(
            final_active_final, trace, args.threshold, args.cap, focus,
            raw_dir=raw_dir, packet_size=args.packet_size,
        )
        fa_by_minute = {int(r["minute"]): r for r in fa_rows}
        merged = []
        for row in base_rows:
            minute = int(row["minute"])
            fa = fa_by_minute.get(minute)
            merged.append(
                {
                    "minute": minute,
                    "original_mrd": row["raw_mrd"],
                    "original_wmrd": row["raw_wmrd"],
                    "gated_mrd": row["gated_mrd"],
                    "gated_wmrd": row["gated_wmrd"],
                    "gated_scale": row["scale"],
                    "final_active_mrd": fa["raw_mrd"] if fa else np.nan,
                    "final_active_wmrd": fa["raw_wmrd"] if fa else np.nan,
                    "final_active_gated_mrd": fa["gated_mrd"] if fa else np.nan,
                    "final_active_gated_wmrd": fa["gated_wmrd"] if fa else np.nan,
                    "final_active_gated_scale": fa["scale"] if fa else np.nan,
                }
            )
        write_csv(out_dir / f"{args.res}_{trace}_curves.csv", merged)
        plot_trace(out_dir, trace, merged)
    print(f"wrote optimization curves to {out_dir}")


if __name__ == "__main__":
    main()
