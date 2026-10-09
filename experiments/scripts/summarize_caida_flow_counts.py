#!/usr/bin/env python3
"""Summarize exact CAIDA flow counts from existing FSD label npy files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_full",
    )
    parser.add_argument("--traces", nargs="+", default=["caida_2016", "caida_2018", "caida_2018_new"])
    parser.add_argument("--limit", type=int, default=0, help="Optional max datasets per trace, after sorting")
    parser.add_argument("--dataset-ids", nargs="*", default=None, help="Optional explicit dataset ids, e.g. dataset_0010")
    parser.add_argument("--out", type=Path, default=Path("zipf_fsd_release/reproduced_figures/flow_counts/caida_flow_counts.csv"))
    return parser.parse_args()


def load_real_fsd(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N,2) real FSD array: {path}, got {arr.shape}")
    return arr.astype(np.int64, copy=False)


def summarize_dataset(trace_root: Path, dataset_id: str) -> dict:
    low = load_real_fsd(trace_root / "tr_ts" / "1_10_real" / f"{dataset_id}.npy")
    high = load_real_fsd(trace_root / "tr_ts" / "10_1e4_real" / f"{dataset_id}.npy")
    full = np.vstack([low, high])
    return {
        "dataset_id": dataset_id,
        "flow_count": int(full[:, 1].sum()),
        "packet_count": int((full[:, 0] * full[:, 1]).sum()),
        "low_flow_count": int(low[:, 1].sum()),
        "high_flow_count": int(high[:, 1].sum()),
        "unique_frequencies": int(full.shape[0]),
    }


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    summary_rows = []

    for trace in args.traces:
        trace_root = args.data_root / trace
        label_dir = trace_root / "tr_ts" / "1_10_real"
        if not label_dir.exists():
            print(f"skip missing {trace}: {label_dir}")
            continue
        if args.dataset_ids:
            dataset_ids = list(args.dataset_ids)
        else:
            dataset_ids = sorted(path.stem for path in label_dir.glob("dataset_*.npy"))
            if args.limit:
                dataset_ids = dataset_ids[: args.limit]
        for dataset_id in dataset_ids:
            row = {"trace": trace, **summarize_dataset(trace_root, dataset_id)}
            detail_rows.append(row)

        flow_counts = [row["flow_count"] for row in detail_rows if row["trace"] == trace]
        packet_counts = [row["packet_count"] for row in detail_rows if row["trace"] == trace]
        summary_rows.append(
            {
                "trace": trace,
                "datasets": len(flow_counts),
                "flow_count_mean": float(np.mean(flow_counts)),
                "flow_count_p10": percentile(flow_counts, 10),
                "flow_count_p50": percentile(flow_counts, 50),
                "flow_count_p90": percentile(flow_counts, 90),
                "flow_count_min": min(flow_counts),
                "flow_count_max": max(flow_counts),
                "packet_count_mean": float(np.mean(packet_counts)),
                "packet_count_min": min(packet_counts),
                "packet_count_max": max(packet_counts),
            }
        )

    fieldnames = [
        "trace",
        "dataset_id",
        "flow_count",
        "packet_count",
        "low_flow_count",
        "high_flow_count",
        "unique_frequencies",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = args.out.with_name("caida_flow_count_summary.csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    for row in summary_rows:
        print(
            f"{row['trace']}: n={row['datasets']} "
            f"flows mean={row['flow_count_mean']:.0f} p50={row['flow_count_p50']:.0f} "
            f"p10={row['flow_count_p10']:.0f} p90={row['flow_count_p90']:.0f} "
            f"packets mean={row['packet_count_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
