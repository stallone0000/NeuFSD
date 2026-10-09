#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from timestamp_windows import MANIFEST, add_window_arguments, prepare_windows, source_files


ONE_TARGETS = np.arange(1, 11, dtype=float)
TEN_TARGETS = np.concatenate((np.arange(11, 1001, dtype=float), np.arange(1001, 10001, 100, dtype=float)))
LABEL_DIRS = ["1_10_chazhi", "10_1e4_chazhi", "1_10_real", "10_1e4_real", "full_real"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare upstream timestamped IMC/MAWI captures in actual 60-second windows: "
            "build exact FSD labels, write exact heavy CSVs, and optionally build measured counters."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace", required=True, choices=["imc", "mawi"])
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--run-root", type=Path, default=Path("mainonly_runs_20260623"))
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--packet-size", type=int, default=13)
    parser.add_argument("--key-offset", type=int, default=0)
    parser.add_argument("--key-length", type=int, default=13)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--build-counter-store", action="store_true")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=400)
    parser.add_argument("--counter-threads", type=int, default=4)
    parser.add_argument("--gen-bin", type=Path, default=Path("run_tools/gen_counter_store"))
    add_window_arguments(parser)
    return parser.parse_args()


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def counter_len_for_res(res: str) -> int:
    if res == "64_64":
        return 4096
    if res == "128_128":
        return 16384
    raise ValueError(f"unsupported --res={res}")


def interp_dense(sorted_fsd: list[tuple[int, float]], targets: np.ndarray) -> np.ndarray:
    if not sorted_fsd:
        return np.zeros(targets.shape, dtype=np.float32)
    x = np.array([v for v, _ in sorted_fsd], dtype=float)
    y = np.array([c for _, c in sorted_fsd], dtype=float)
    if len(x) == 1:
        return np.full(targets.shape, y[0], dtype=np.float32)
    func = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    out = np.asarray(func(targets), dtype=np.float32)
    out[out < 0] = 0
    return out


def ensure_dirs(trace_root: Path) -> None:
    for folder in LABEL_DIRS:
        (trace_root / "tr_ts" / folder).mkdir(parents=True, exist_ok=True)
    (trace_root / "EL").mkdir(parents=True, exist_ok=True)


def split_windows(args: argparse.Namespace, raw_dir: Path) -> tuple[int, int]:
    manifest = prepare_windows(
        source_files(args.source), raw_dir, window_mode=args.window_mode,
        window_seconds=args.window_seconds, window_records=args.window_records,
        last_window=args.last_window, input_format=args.input_format,
        record_size=args.packet_size, timestamp_format=args.timestamp_format,
        timestamp_offset=args.timestamp_offset, output_key=True, key_offset=args.key_offset,
        pcap_policy=args.pcap_policy or ("ipv4" if args.trace == "imc" else "tcp-udp"),
        max_windows=args.max_windows,
    )
    return len(manifest["windows"]), manifest["discarded_records"]


def labels_complete(trace_root: Path, minute: int) -> bool:
    name = dataset_id(minute)
    for folder in LABEL_DIRS:
        if not (trace_root / "tr_ts" / folder / f"{name}.npy").exists():
            return False
    return (trace_root / "EL" / str(minute) / "heavy_0.csv").exists()


def build_one_window(
    raw_path_text: str,
    trace_root_text: str,
    minute: int,
    packet_size: int,
    key_offset: int,
    key_length: int,
    force: bool,
) -> dict[str, object]:
    raw_path = Path(raw_path_text)
    trace_root = Path(trace_root_text)
    if labels_complete(trace_root, minute) and not force:
        return {"minute": minute, "cache_hit": True}

    t0 = time.perf_counter()
    if key_offset == 0 and key_length == packet_size:
        keys = np.fromfile(raw_path, dtype=np.dtype(f"S{packet_size}"))
    else:
        packets = np.fromfile(raw_path, dtype=np.dtype(f"S{packet_size}"))
        keys = np.array([bytes(packet)[key_offset : key_offset + key_length] for packet in packets], dtype=f"S{key_length}")
    _, flow_counts = np.unique(keys, return_counts=True)
    parse_sec = time.perf_counter() - t0

    fsd_t0 = time.perf_counter()
    freqs, counts = np.unique(flow_counts.astype(np.int64, copy=False), return_counts=True)
    sorted_fsd = [(int(freq), float(count)) for freq, count in zip(freqs, counts)]
    fsd_sec = time.perf_counter() - fsd_t0

    label_t0 = time.perf_counter()
    name = dataset_id(minute)
    tr_ts = trace_root / "tr_ts"
    for folder in LABEL_DIRS:
        (tr_ts / folder).mkdir(parents=True, exist_ok=True)
    np.save(tr_ts / "1_10_chazhi" / f"{name}.npy", interp_dense(sorted_fsd, ONE_TARGETS))
    np.save(tr_ts / "10_1e4_chazhi" / f"{name}.npy", interp_dense(sorted_fsd, TEN_TARGETS))
    np.save(
        tr_ts / "1_10_real" / f"{name}.npy",
        np.array([(f, c) for f, c in sorted_fsd if f <= 10], dtype=np.float32),
    )
    np.save(
        tr_ts / "10_1e4_real" / f"{name}.npy",
        np.array([(f, c) for f, c in sorted_fsd if 10 < f <= 10000], dtype=np.float32),
    )
    np.save(tr_ts / "full_real" / f"{name}.npy", np.array(sorted_fsd, dtype=np.float32))

    heavy_dir = trace_root / "EL" / str(minute)
    heavy_dir.mkdir(parents=True, exist_ok=True)
    heavy_rows = 0
    heavy_mass = 0
    with (heavy_dir / "heavy_0.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "slot", "fingerprint", "count", "flag"])
        writer.writeheader()
        row_id = 0
        for freq, count in sorted_fsd:
            if freq < 100:
                continue
            for _ in range(int(round(count))):
                writer.writerow(
                    {
                        "bucket": row_id // 4,
                        "slot": row_id % 4,
                        "fingerprint": row_id,
                        "count": int(freq),
                        "flag": 1,
                    }
                )
                row_id += 1
            heavy_rows += int(round(count))
            heavy_mass += int(freq) * int(round(count))
    label_sec = time.perf_counter() - label_t0

    return {
        "minute": minute,
        "cache_hit": False,
        "parse_unique_sec": parse_sec,
        "fsd_build_sec": fsd_sec,
        "label_heavy_write_sec": label_sec,
        "raw_packets": int(raw_path.stat().st_size // packet_size),
        "full_flows": int(flow_counts.size),
        "fsd_points": len(sorted_fsd),
        "heavy_rows": heavy_rows,
        "heavy_mass": heavy_mass,
    }


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


def build_labels(args: argparse.Namespace, trace_root: Path, raw_dir: Path, windows: int) -> list[dict[str, object]]:
    ensure_dirs(trace_root)
    pending = [m for m in range(windows) if args.force_labels or not labels_complete(trace_root, m)]
    if not pending:
        print("labels/heavy already complete", flush=True)
        return []
    workers = args.workers or max(1, min(32, os.cpu_count() or 1))
    rows: list[dict[str, object]] = []
    print(f"building exact labels/heavy for {len(pending)} windows with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                build_one_window,
                str(raw_dir / f"{dataset_id(minute)}.dat"),
                str(trace_root),
                minute,
                13,
                0,
                13,
                args.force_labels,
            )
            for minute in pending
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if done % 25 == 0 or done == len(futures):
                print(f"  labels {done}/{len(futures)}", flush=True)
    rows = sorted(rows, key=lambda row: int(row["minute"]))
    write_csv(trace_root / "prepare_key13_labels_timing.csv", rows)
    return rows


def write_train_test_json(trace_root: Path, windows: int) -> None:
    mapping: dict[str, bool] = {}
    for start in range(0, windows, 10):
        train_end = min(start + 5, windows)
        test_start = start + 10
        if train_end <= start or test_start >= windows:
            break
        for minute in range(start, train_end):
            mapping[f"fine_dataset_{minute:04d}.dat"] = True
        for minute in range(test_start, min(test_start + 10, windows)):
            mapping[f"dataset_{minute:04d}.dat"] = False
    with (trace_root / "train_test_name_key.json").open("w") as f:
        json.dump(mapping, f, indent=4)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise ValueError(f"refusing to replace existing {dst}; use a fresh --run-root")
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_final_links(args: argparse.Namespace, trace_root: Path) -> Path:
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(trace_root / "tr_ts", final_dir / "tr_ts")
    link_or_copy(trace_root / "EL", final_dir / "EL")
    source_mapping = trace_root / "train_test_name_key.json"
    target_mapping = final_dir / "train_test_name_key.json"
    if target_mapping.exists() and target_mapping.read_bytes() != source_mapping.read_bytes():
        raise ValueError(f"refusing to replace existing {target_mapping}; use a fresh --run-root")
    if not target_mapping.exists():
        shutil.copy2(source_mapping, target_mapping)
    return final_dir


def build_measured_counter_store(args: argparse.Namespace, raw_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    out_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    cmd = [
        sys.executable,
        str(root / "scripts" / "build_counter_store.py"),
        "--input-dir",
        str(raw_dir),
        "--out-root",
        str(out_root),
        "--trace-format",
        args.trace,
        "--counter-len",
        str(counter_len_for_res(args.res)),
        "--start-seed",
        str(args.start_seed),
        "--end-seed",
        str(args.end_seed),
        "--gen-bin",
        str(args.gen_bin),
        "--tmp-dir",
        str(out_root / "_counter_blocks"),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.counter_threads)
    subprocess.run(cmd, cwd=root, env=env, check=True)
    return out_root


def main() -> None:
    args = parse_args()
    if args.key_length != 13 or args.key_offset < 0 or args.key_offset + 13 > args.packet_size:
        raise ValueError("This helper requires a 13-byte key inside each fixed-size input record.")
    trace_root = args.data_full_root / args.trace
    raw_dir = trace_root / "caida_1min_split"
    if not (raw_dir / MANIFEST).exists() and trace_root.exists():
        derived = (trace_root / "tr_ts", trace_root / "EL", trace_root / "train_test_name_key.json")
        if any(p.exists() and (p.is_file() or any(p.iterdir())) for p in derived):
            raise ValueError(f"{trace_root}: existing derived data lacks window provenance; use a fresh --data-full-root")
    t0 = time.perf_counter()
    windows, discarded = split_windows(args, raw_dir)
    window_manifest = json.loads((raw_dir / MANIFEST).read_text())
    rows = build_labels(args, trace_root, raw_dir, windows)
    write_train_test_json(trace_root, windows)
    final_dir = prepare_final_links(args, trace_root)
    counter_root = None
    if args.build_counter_store:
        counter_root = build_measured_counter_store(args, raw_dir)
    manifest = {
        "trace": args.trace,
        "source": str(args.source),
        "windows": windows,
        "window_mode": args.window_mode,
        "window_seconds": str(args.window_seconds) if args.window_mode == "time" else None,
        "window_records": args.window_records if args.window_mode == "records" else None,
        "last_window": args.last_window,
        "window_manifest": str(raw_dir / MANIFEST),
        "records_per_window": [w["records"] for w in window_manifest["windows"]],
        "discarded_records": discarded,
        "input_packet_size": args.packet_size,
        "packet_size": 13,
        "key_offset": 0,
        "key_length": 13,
        "label_rows_written": len(rows),
        "final_dir": str(final_dir),
        "counter_root": str(counter_root) if counter_root else None,
        "wall_sec": time.perf_counter() - t0,
    }
    with (trace_root / "prepare_key13_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
