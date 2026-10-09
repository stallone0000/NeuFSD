#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from packet_counts import validate_packet_count, window_packet_count


TARGET_FREQS = np.concatenate((np.arange(1, 1001, 1), np.arange(1001, 10001, 100)))
SPLICE_POINT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep online-safe light-mass gate variants on existing predictions.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--packet-size", type=int, default=0)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--mode", default="pretrain_continuous")
    parser.add_argument("--traces", nargs="+", default=["caida_2018", "caida_2018_new"])
    parser.add_argument("--focus", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
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


def variants(raw_scale: float) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = [("original", 1.0, 1.0, 0.0)]
    for threshold in [1.02, 1.05, 1.08, 1.10]:
        for cap in [1.10, 1.20, 1.25, 1.35]:
            capped = min(raw_scale, cap) if raw_scale > threshold else 1.0
            rows.append((f"uniform_t{threshold:g}_c{cap:g}", capped, capped, threshold))
            rows.append((f"sqrt_t{threshold:g}_c{cap:g}", capped ** 0.5, capped ** 0.5, threshold))
            rows.append((f"le10_t{threshold:g}_c{cap:g}", capped, 1.0, threshold))
            rows.append((f"split_t{threshold:g}_c{cap:g}", capped, capped ** 0.5, threshold))
            for alpha in [0.5, 0.75]:
                blended = 1.0 + alpha * (capped - 1.0)
                rows.append((f"blend{alpha:g}_t{threshold:g}_c{cap:g}", blended, blended, threshold))
    return rows


def dataset_rows(final_dir: Path, dataset_id: str, block_name: str, *, packet_count: int) -> list[dict[str, object]]:
    root = final_dir / "0_finetuned_results"
    pred_1 = np.load(root / "ViT_1_10_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy")
    pred_2 = np.load(root / "ViT_10_1e4_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy")
    corr_preds = np.column_stack((pred_1, pred_2))
    flow_sizes, true_counts = load_true(final_dir, dataset_id)
    small_mask = flow_sizes <= SPLICE_POINT
    large_mask = ~small_mask
    le10_mask = flow_sizes <= 10
    mid_mask = (flow_sizes > 10) & (flow_sizes <= SPLICE_POINT)
    small_sizes = flow_sizes[small_mask]

    el_freq = load_el_freq(final_dir, dataset_id)
    base_large = np.zeros(flow_sizes.shape, dtype=float)
    base_large[large_mask] = large_el_predictions(el_freq, flow_sizes[large_mask])
    packet_count = validate_packet_count(packet_count)
    heavy_packet_mass = float(np.sum(el_freq[:, 0] * el_freq[:, 1])) if el_freq.size else 0.0
    residual_light_mass = max(packet_count - heavy_packet_mass, 0.0)

    metric_sum: dict[str, list[float]] = {}
    for sep_preds in corr_preds:
        sep_preds = np.maximum(sep_preds, 0)
        base = base_large.copy()
        if small_sizes.size:
            base[small_mask] = np.interp(small_sizes, TARGET_FREQS, sep_preds)
        light_sizes = np.arange(1, SPLICE_POINT + 1)
        light_preds = np.interp(light_sizes, TARGET_FREQS, sep_preds)
        light_mass = float(np.sum(light_sizes * np.maximum(light_preds, 0)))
        raw_scale = residual_light_mass / light_mass if light_mass > 0 else 1.0
        for name, le10_scale, mid_scale, _ in variants(raw_scale):
            pred = base.copy()
            pred[le10_mask] *= le10_scale
            pred[mid_mask] *= mid_scale
            pred = np.around(pred, 0)
            pred[pred < 0] = 0
            mrd, wmrd = metrics(pred, true_counts)
            metric_sum.setdefault(name, [0.0, 0.0, 0.0])
            metric_sum[name][0] += mrd
            metric_sum[name][1] += wmrd
            metric_sum[name][2] += 1.0
    minute = int(dataset_id.split("_")[-1])
    rows = []
    for name, (mrd_sum, wmrd_sum, n) in metric_sum.items():
        rows.append({"minute": minute, "dataset_id": dataset_id, "variant": name, "mrd": mrd_sum / n, "wmrd": wmrd_sum / n})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    focus = parse_focus(args.focus)
    out_dir = args.out_dir or (run_root / "plots" / "lightmass_gate_sweep")
    all_rows: list[dict[str, object]] = []
    for trace in args.traces:
        final_dir = run_root / "online" / args.mode / f"{args.res}_{trace}_final"
        for dataset_id, train_names in grouped_tasks(final_dir).items():
            minute = int(dataset_id.split("_")[-1])
            if not in_focus(focus, trace, minute):
                continue
            packet_count = window_packet_count(
                args.data_full_root / trace / "caida_1min_split", minute, trace, args.packet_size,
            )
            for row in dataset_rows(final_dir, dataset_id, train_names[0], packet_count=packet_count):
                row["trace"] = trace
                all_rows.append(row)
    write_csv(out_dir / f"{args.res}_{args.mode}_detail.csv", all_rows)
    summary: list[dict[str, object]] = []
    variants_seen = sorted({str(r["variant"]) for r in all_rows})
    for variant in variants_seen:
        rows = [r for r in all_rows if r["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "n": len(rows),
                "mrd": float(np.mean([float(r["mrd"]) for r in rows])),
                "wmrd": float(np.mean([float(r["wmrd"]) for r in rows])),
                "max_wmrd": float(np.max([float(r["wmrd"]) for r in rows])),
            }
        )
    summary.sort(key=lambda r: (float(r["wmrd"]), float(r["mrd"])))
    write_csv(out_dir / f"{args.res}_{args.mode}_summary.csv", summary)
    print(f"wrote {out_dir} variants={len(summary)} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
