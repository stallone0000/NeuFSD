#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ALPHAS = [0.0, 0.2] + [round(x / 10, 1) for x in range(10, 21)] + [2.5, 3.0]
FORMAT = {
    "caida2016": {"packet_size": 16, "key_offset": 8, "key_size": 8},
    "caida2018": {"packet_size": 21, "key_offset": 0, "key_size": 13},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact Zipf pretraining slices and labels.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["caida2016", "caida2018"], choices=sorted(FORMAT))
    parser.add_argument("--flows", nargs="+", type=int, default=[80000, 100000])
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--packets", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-ensure-all-flows", action="store_true")
    parser.add_argument("--write-raw", action="store_true", default=True)
    return parser.parse_args()


def alpha_name(alpha: float) -> str:
    return f"{alpha:.1f}".replace(".", "p")


def write_packets(path: Path, ranks: np.ndarray, fmt: str, flow_seed: int) -> None:
    cfg = FORMAT[fmt]
    packet_size = cfg["packet_size"]
    key_offset = cfg["key_offset"]
    key_size = cfg["key_size"]
    chunk_size = 200_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for start in range(0, len(ranks), chunk_size):
            chunk = ranks[start : start + chunk_size].astype(">u8", copy=False)
            key8 = chunk.view("u1").reshape(-1, 8)
            packets = np.zeros((len(chunk), packet_size), dtype=np.uint8)
            if key_size == 8:
                packets[:, key_offset : key_offset + 8] = key8
            else:
                prefix = np.array([(flow_seed >> shift) & 0xFF for shift in (32, 24, 16, 8, 0)], dtype=np.uint8)
                packets[:, key_offset : key_offset + 5] = prefix
                packets[:, key_offset + 5 : key_offset + 13] = key8
            f.write(packets.tobytes())


def exact_labels(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    positive = counts[counts > 0].astype(np.int64)
    fsd = np.bincount(positive)
    freqs = np.nonzero(fsd)[0].astype(np.int64)
    nums = fsd[freqs].astype(np.int64)
    full = np.column_stack([freqs, nums])
    low_real = full[full[:, 0] <= 10]
    high_real = full[(full[:, 0] > 10) & (full[:, 0] <= 10000)]

    exact = {int(freq): int(num) for freq, num in full}
    low_chazhi = np.array([exact.get(freq, 0) for freq in range(1, 11)], dtype=np.float64)
    high_targets = np.concatenate((np.arange(11, 1001, 1), np.arange(1001, 10001, 100)))
    high_chazhi = np.array([exact.get(int(freq), 0) for freq in high_targets], dtype=np.float64)
    meta = {
        "observed_flows": int(positive.shape[0]),
        "packet_count": int((freqs * nums).sum()),
        "max_flow_size": int(freqs.max()),
        "unique_frequencies": int(freqs.shape[0]),
    }
    return full, low_real, high_real, low_chazhi, high_chazhi, meta


def deterministic_counts(packet_count: int, flow_count: int, alpha: float) -> np.ndarray:
    if flow_count > packet_count:
        raise ValueError("flow_count cannot exceed packet_count")
    remaining = packet_count - flow_count
    if alpha == 0:
        weights = np.ones(flow_count, dtype=np.float64)
    else:
        ranks = np.arange(1, flow_count + 1, dtype=np.float64)
        weights = np.power(ranks, -alpha)
    weights /= weights.sum()
    expected = weights * remaining
    extra = np.floor(expected).astype(np.int64)
    leftover = int(remaining - extra.sum())
    if leftover:
        order = np.argsort(-(expected - extra), kind="mergesort")
        extra[order[:leftover]] += 1
    counts = extra + 1
    if counts.shape[0] != flow_count:
        raise AssertionError("flow support mismatch")
    if int(counts.sum()) != packet_count:
        raise AssertionError("packet count mismatch")
    return counts


def expand_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ranks = np.repeat(np.arange(1, counts.shape[0] + 1, dtype=np.uint64), counts.astype(np.int64))
    rng.shuffle(ranks)
    return ranks


def save_labels(root: Path, dataset_id: str, counts: np.ndarray, manifest_row: dict) -> None:
    full_real, low_real, high_real, low_chazhi, high_chazhi, meta = exact_labels(counts)
    for sub in ["full_real", "1_10_real", "10_1e4_real", "1_10_chazhi", "10_1e4_chazhi"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    np.save(root / "full_real" / f"{dataset_id}.npy", full_real)
    np.save(root / "1_10_real" / f"{dataset_id}.npy", low_real)
    np.save(root / "10_1e4_real" / f"{dataset_id}.npy", high_real)
    np.save(root / "1_10_chazhi" / f"{dataset_id}.npy", low_chazhi)
    np.save(root / "10_1e4_chazhi" / f"{dataset_id}.npy", high_chazhi)
    manifest_row.update(meta)


def main() -> None:
    args = parse_args()
    ensure_all = not args.no_ensure_all_flows
    rng = np.random.default_rng(args.seed)
    rows = []
    jobs = [(fmt, flows, alpha) for fmt in args.formats for flows in args.flows for alpha in args.alphas]
    if args.limit:
        jobs = jobs[: args.limit]
    for fmt, flows, alpha in jobs:
        dataset_id = f"zipf_{fmt}_flows{flows}_alpha{alpha_name(alpha)}"
        print(f"generating {dataset_id}")
        fmt_root = args.out_root / fmt
        raw_path = fmt_root / "caida_1min_split" / f"{dataset_id}.dat"
        labels_root = fmt_root / "tr_ts"
        counts = deterministic_counts(args.packets, flows, alpha) if ensure_all else None
        if counts is None:
            raise RuntimeError("non-exact Zipf generation is disabled for pretraining")
        ranks = expand_counts(counts, rng)
        if args.write_raw:
            write_packets(raw_path, ranks, fmt, args.seed)
        row = {
            "dataset_id": dataset_id,
            "format": fmt,
            "target_flows": flows,
            "alpha": alpha,
            "raw_path": str(raw_path),
            "ensure_all_flows": ensure_all,
            "generation": "deterministic_largest_remainder",
        }
        save_labels(labels_root, dataset_id, counts, row)
        rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "manifest.json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    print(f"wrote {args.out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
