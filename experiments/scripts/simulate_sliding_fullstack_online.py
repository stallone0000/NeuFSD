#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from packet_counts import window_packet_count
from simulate_fullstack_online import build_counter_store, counter_len_for_res, fake_packet
from simulate_realtime_online import (
    HEADS,
    cached_counter,
    default_packet_format,
    final_metrics,
    fine_id,
    infer_head,
    interp_dense,
    make_model,
    parse_packet_keys,
    setup_imports,
    stable_sample,
    train_head,
    write_csv,
)


LABEL_DIRS = ["1_10_chazhi", "10_1e4_chazhi", "1_10_real", "10_1e4_real"]
ONE_TARGETS = np.arange(1, 11, dtype=float)
TEN_TARGETS = np.concatenate((np.arange(11, 1001, dtype=float), np.arange(1001, 10001, 100, dtype=float)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-minute sliding full-stack continuous SFT. The control plane keeps "
            "sample-restored FSDs for recent one-minute windows and fine-tunes once "
            "per minute before measuring the current minute."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", required=True, choices=["caida_2016", "caida_2018", "caida_2018_new", "imc", "mawi"])
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--sample-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=["last", "window5", "avg5"], default="window5")
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--start-minute", type=int, default=-1)
    parser.add_argument("--end-minute", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["full", "head-only"], default="full")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=400)
    parser.add_argument(
        "--last-replicas",
        type=int,
        default=1,
        help=(
            "For strategy=last, synthesize this many replicas from the previous window's restored FSD. "
            "Use this to match window5's total training sample count without changing the one-window semantics."
        ),
    )
    parser.add_argument(
        "--last-replica-stride",
        type=int,
        default=10000,
        help="Minute-id stride used to name last-strategy replica fine_dataset files.",
    )
    parser.add_argument(
        "--avg-replicas",
        type=int,
        default=1,
        help=(
            "For strategy=avg5, synthesize this many fine_dataset replicas from the same averaged FSD. "
            "This preserves the avg5 label semantics while increasing counter/key diversity."
        ),
    )
    parser.add_argument(
        "--avg-replica-stride",
        type=int,
        default=10000,
        help="Minute-id stride used to name avg5 replica fine_dataset files.",
    )
    parser.add_argument("--gpu-deadline-sec", type=float, default=60.0)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--counter-threads", type=int, default=0)
    parser.add_argument("--gen-bin", type=Path, default=Path("run_tools/gen_counter_store"))
    parser.add_argument("--gate-threshold", type=float, default=1.02)
    parser.add_argument("--gate-cap", type=float, default=1.20)
    parser.add_argument("--gate-down-threshold", type=float, default=0.0)
    parser.add_argument("--gate-floor", type=float, default=0.50)
    parser.add_argument("--sample-shape-weight", type=float, default=0.0)
    parser.add_argument("--sample-shape-max-freq", type=int, default=100)
    parser.add_argument("--adaptive-gate", action="store_true")
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        default=None,
        help="Optional root containing ViT_* pretrained checkpoint directories. Defaults to run-root/pretrained/res.",
    )
    parser.add_argument(
        "--init-from",
        choices=["pretrained", "scratch"],
        default="pretrained",
        help="Initialize models from pretrained checkpoints or random scratch weights.",
    )
    parser.add_argument(
        "--runtime-train-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional cache for per-minute runtime training roots. If a cached root already has "
            "input_store/index.json, labels, and fine_dat files, reuse it across epoch sweeps."
        ),
    )
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--force-cache-rebuild", action="store_true")
    parser.add_argument(
        "--use-sample-counter-cache",
        action="store_true",
        help=(
            "For last/window5, read training counters directly from sample-cache-dir/input_store. "
            "Build that store once with scripts/build_counter_store.py over sample-cache-dir/fine_dat."
        ),
    )
    parser.add_argument("--keep-train-roots", action="store_true")
    return parser.parse_args()


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def numeric_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits or 0)


def last_raw_minute(raw_dir: Path) -> int:
    files = sorted(raw_dir.glob("dataset_*.dat"), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no dataset_*.dat files under {raw_dir}")
    return numeric_key(files[-1])


def ensure_label_dirs(root: Path) -> None:
    for folder in LABEL_DIRS:
        (root / folder).mkdir(parents=True, exist_ok=True)


def fsd_paths(root: Path, minute: int) -> dict[str, Path]:
    name = fine_id(minute)
    return {
        "fine_dat": root / "fine_dat" / f"{name}.dat",
        "fsd": root / "sampled_fsd" / f"{name}.npz",
        "one": root / "1_10_chazhi" / f"{name}.npy",
        "ten": root / "10_1e4_chazhi" / f"{name}.npy",
        "one_real": root / "1_10_real" / f"{name}.npy",
        "ten_real": root / "10_1e4_real" / f"{name}.npy",
    }


def cache_complete(root: Path, minute: int) -> bool:
    paths = fsd_paths(root, minute)
    return all(path.exists() for path in paths.values())


def write_artifacts_from_fsd(
    root: Path,
    name: str,
    minute_for_packet_key: int,
    sorted_fsd: list[tuple[int, float]],
    trace: str,
    save_fsd: bool,
) -> dict[str, object]:
    packet_size, key_offset, key_length = default_packet_format(trace)
    ensure_label_dirs(root)
    (root / "fine_dat").mkdir(parents=True, exist_ok=True)
    if save_fsd:
        (root / "sampled_fsd").mkdir(parents=True, exist_ok=True)

    label_t0 = time.perf_counter()
    one = interp_dense(sorted_fsd, ONE_TARGETS)
    ten = interp_dense(sorted_fsd, TEN_TARGETS)
    np.save(root / "1_10_chazhi" / f"{name}.npy", one)
    np.save(root / "10_1e4_chazhi" / f"{name}.npy", ten)
    real_small = np.array([(freq, count) for freq, count in sorted_fsd if freq <= 10], dtype=np.float32)
    real_large = np.array([(freq, count) for freq, count in sorted_fsd if 10 < freq <= 10000], dtype=np.float32)
    np.save(root / "1_10_real" / f"{name}.npy", real_small)
    np.save(root / "10_1e4_real" / f"{name}.npy", real_large)
    if save_fsd:
        np.savez_compressed(
            root / "sampled_fsd" / f"{name}.npz",
            freq=np.array([freq for freq, _ in sorted_fsd], dtype=np.int32),
            count=np.array([count for _, count in sorted_fsd], dtype=np.float32),
        )
    label_sec = time.perf_counter() - label_t0

    synth_t0 = time.perf_counter()
    fine_path = root / "fine_dat" / f"{name}.dat"
    fine_packets = 0
    fine_flows = 0
    flow_index = 1
    with fine_path.open("wb") as out:
        for size, flow_count in sorted_fsd:
            integer_flows = int(round(max(float(flow_count), 0.0)))
            if integer_flows <= 0 or size <= 0:
                continue
            packet = fake_packet(flow_index, minute_for_packet_key, packet_size, key_offset, key_length)
            for _ in range(integer_flows):
                out.write(packet * int(size))
                fine_packets += int(size)
                fine_flows += 1
                flow_index += 1
                packet = fake_packet(flow_index, minute_for_packet_key, packet_size, key_offset, key_length)
    synth_sec = time.perf_counter() - synth_t0

    return {
        "label_write_sec": label_sec,
        "synth_stream_write_sec": synth_sec,
        "fine_flows": fine_flows,
        "fine_packets": fine_packets,
        "fine_bytes": fine_path.stat().st_size,
    }


def prepare_sample_cache_one(
    raw_path_text: str,
    cache_root_text: str,
    minute: int,
    trace: str,
    sample_rate: float,
    sample_seed: int,
    force: bool,
) -> dict[str, object]:
    raw_path = Path(raw_path_text)
    cache_root = Path(cache_root_text)
    if cache_complete(cache_root, minute) and not force:
        return {
            "minute": minute,
            "raw_path": str(raw_path),
            "cache_hit": True,
            "parse_sec": 0.0,
            "sample_reconstruct_sec": 0.0,
            "label_write_sec": 0.0,
            "synth_stream_write_sec": 0.0,
            "full_flows": np.nan,
            "sampled_flows": np.nan,
            "sampled_packets": np.nan,
            "fine_flows": np.nan,
            "fine_packets": np.nan,
            "raw_packets": np.nan,
            "fine_bytes": np.nan,
        }

    packet_size, key_offset, key_length = default_packet_format(trace)
    parse_t0 = time.perf_counter()
    flow_counts = parse_packet_keys(raw_path, packet_size, key_offset, key_length)
    parse_sec = time.perf_counter() - parse_t0

    sample_t0 = time.perf_counter()
    multiplier = 1.0 / sample_rate
    sampled_fsd: dict[int, float] = defaultdict(float)
    sampled_flows = 0
    sampled_packets = 0
    for key, count in flow_counts.items():
        if stable_sample(key, minute, sample_rate, sample_seed):
            sampled_fsd[int(count)] += multiplier
            sampled_flows += 1
            sampled_packets += int(count)
    sample_sec = time.perf_counter() - sample_t0

    name = fine_id(minute)
    artifact_info = write_artifacts_from_fsd(
        cache_root,
        name,
        minute,
        sorted(sampled_fsd.items()),
        trace,
        save_fsd=True,
    )

    return {
        "minute": minute,
        "raw_path": str(raw_path),
        "cache_hit": False,
        "parse_sec": parse_sec,
        "sample_reconstruct_sec": sample_sec,
        "full_flows": len(flow_counts),
        "sampled_flows": sampled_flows,
        "sampled_packets": sampled_packets,
        "raw_packets": raw_path.stat().st_size // packet_size,
        **artifact_info,
    }


def prepare_cache_range(args: argparse.Namespace, raw_dir: Path, sample_minutes: list[int]) -> list[dict[str, object]]:
    args.sample_cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    existing_path = args.sample_cache_dir / "sample_cache_timing.csv"
    if existing_path.exists() and not args.force_cache_rebuild:
        with existing_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    pending = [minute for minute in sample_minutes if args.force_cache_rebuild or not cache_complete(args.sample_cache_dir, minute)]
    if not pending:
        return rows

    workers = args.prep_workers or max(1, min(32, os.cpu_count() or 1))
    t0 = time.perf_counter()
    print(f"preparing sample cache: {len(pending)} missing windows with {workers} workers", flush=True)
    new_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for minute in pending:
            raw_path = raw_dir / f"{dataset_id(minute)}.dat"
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            futures.append(
                pool.submit(
                    prepare_sample_cache_one,
                    str(raw_path),
                    str(args.sample_cache_dir),
                    minute,
                    args.trace,
                    args.sample_rate,
                    args.sample_seed,
                    args.force_cache_rebuild,
                )
            )
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            new_rows.append(row)
            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"  sample cache prepared {done}/{len(futures)}", flush=True)

    combined: dict[int, dict[str, object]] = {}
    for row in rows:
        combined[int(row["minute"])] = row
    for row in new_rows:
        combined[int(row["minute"])] = row
    rows = [combined[k] for k in sorted(combined)]
    write_csv(args.sample_cache_dir / "sample_cache_timing.csv", rows)
    with (args.sample_cache_dir / "sample_cache_summary.json").open("w") as f:
        json.dump(
            {
                "trace": args.trace,
                "sample_rate": args.sample_rate,
                "sample_seed": args.sample_seed,
                "minutes": [min(sample_minutes), max(sample_minutes)] if sample_minutes else [],
                "new_windows": len(new_rows),
                "wall_sec": time.perf_counter() - t0,
            },
            f,
            indent=2,
        )
    return rows


def load_cached_fsd(cache_root: Path, minute: int) -> list[tuple[int, float]]:
    path = cache_root / "sampled_fsd" / f"{fine_id(minute)}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    freq = data["freq"].astype(int)
    count = data["count"].astype(float)
    return [(int(f), float(c)) for f, c in zip(freq, count) if c > 0]


def average_fsd(cache_root: Path, minutes: list[int]) -> list[tuple[int, float]]:
    total: dict[int, float] = defaultdict(float)
    for minute in minutes:
        for freq, count in load_cached_fsd(cache_root, minute):
            total[int(freq)] += float(count)
    scale = 1.0 / max(1, len(minutes))
    return sorted((freq, count * scale) for freq, count in total.items() if count > 0)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_labels(cache_root: Path, train_root: Path, minute: int) -> None:
    name = fine_id(minute)
    for folder in LABEL_DIRS:
        src = cache_root / folder / f"{name}.npy"
        dst = train_root / folder / f"{name}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def train_root_complete(root: Path, train_minutes: list[int]) -> bool:
    if not (root / "input_store" / "index.json").exists():
        return False
    for minute in train_minutes:
        name = fine_id(minute)
        if not (root / "fine_dat" / f"{name}.dat").exists():
            return False
        for folder in LABEL_DIRS:
            if not (root / folder / f"{name}.npy").exists():
                return False
    return True


def aggregate_artifacts(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "label_write_sec": float(sum(float(row["label_write_sec"]) for row in rows)),
        "synth_stream_write_sec": float(sum(float(row["synth_stream_write_sec"]) for row in rows)),
        "fine_flows": float(sum(float(row["fine_flows"]) for row in rows)),
        "fine_packets": float(sum(float(row["fine_packets"]) for row in rows)),
        "fine_bytes": float(sum(float(row["fine_bytes"]) for row in rows)),
    }


def build_training_root(args: argparse.Namespace, test_minute: int, history: list[int]) -> tuple[Path, list[int], dict[str, object]]:
    if args.use_sample_counter_cache and args.strategy in {"last", "window5"} and args.last_replicas == 1:
        if not (args.sample_cache_dir / "input_store" / "index.json").exists():
            raise FileNotFoundError(
                f"{args.sample_cache_dir / 'input_store' / 'index.json'} is required by --use-sample-counter-cache"
            )
        train_minutes = [history[-1]] if args.strategy == "last" else list(history)
        return (
            args.sample_cache_dir,
            train_minutes,
            {
                "train_root_build_sec": 0.0,
                "avg_source_start": np.nan,
                "avg_source_end": np.nan,
                "used_sample_counter_cache": True,
                "counter_ready": True,
            },
        )

    if args.runtime_train_cache_dir is not None:
        train_root = args.runtime_train_cache_dir / f"minute_{test_minute:04d}_{args.strategy}"
    else:
        train_root = args.out_dir / "runtime_train_roots" / f"minute_{test_minute:04d}_{args.strategy}"

    if args.strategy == "last" and args.last_replicas > 1:
        train_minutes = [history[-1] + replica * args.last_replica_stride for replica in range(args.last_replicas)]
    elif args.strategy == "avg5":
        train_minutes = [test_minute + replica * args.avg_replica_stride for replica in range(args.avg_replicas)]
    else:
        train_minutes = [history[-1]] if args.strategy == "last" else list(history)

    if train_root_complete(train_root, train_minutes):
        return (
            train_root,
            train_minutes,
            {
                "train_root_build_sec": 0.0,
                "avg_source_start": min(history) if args.strategy == "avg5" else np.nan,
                "avg_source_end": max(history) if args.strategy == "avg5" else np.nan,
                "avg_freq_points": np.nan,
                "last_source_minute": history[-1] if args.strategy == "last" else np.nan,
                "last_replicas": args.last_replicas if args.strategy == "last" else np.nan,
                "avg_replicas": args.avg_replicas if args.strategy == "avg5" else np.nan,
                "used_sample_counter_cache": True,
                "counter_ready": True,
            },
        )

    if train_root.exists():
        shutil.rmtree(train_root)
    ensure_label_dirs(train_root)
    (train_root / "fine_dat").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.strategy == "last":
        if args.last_replicas == 1:
            for minute in train_minutes:
                link_or_copy(
                    args.sample_cache_dir / "fine_dat" / f"{fine_id(minute)}.dat",
                    train_root / "fine_dat" / f"{fine_id(minute)}.dat",
                )
                copy_labels(args.sample_cache_dir, train_root, minute)
            extra = {
                "avg_source_start": np.nan,
                "avg_source_end": np.nan,
                "last_source_minute": history[-1],
                "last_replicas": args.last_replicas,
                "used_sample_counter_cache": False,
                "counter_ready": False,
            }
        else:
            source_minute = history[-1]
            source_fsd = load_cached_fsd(args.sample_cache_dir, source_minute)
            artifact_rows = []
            for replica, train_minute in enumerate(train_minutes):
                artifact_rows.append(
                    write_artifacts_from_fsd(
                        train_root,
                        fine_id(train_minute),
                        train_minute,
                        source_fsd,
                        args.trace,
                        save_fsd=False,
                    )
                    | {"replica": replica}
                )
            artifact_info = aggregate_artifacts(artifact_rows)
            extra = {
                "avg_source_start": np.nan,
                "avg_source_end": np.nan,
                "last_source_minute": source_minute,
                "last_replicas": args.last_replicas,
                "used_sample_counter_cache": False,
                "counter_ready": False,
                **{f"last_{k}": v for k, v in artifact_info.items()},
            }
    elif args.strategy == "window5":
        for minute in train_minutes:
            link_or_copy(
                args.sample_cache_dir / "fine_dat" / f"{fine_id(minute)}.dat",
                train_root / "fine_dat" / f"{fine_id(minute)}.dat",
            )
            copy_labels(args.sample_cache_dir, train_root, minute)
        extra = {
            "avg_source_start": np.nan,
            "avg_source_end": np.nan,
            "last_source_minute": np.nan,
            "last_replicas": np.nan,
            "used_sample_counter_cache": False,
            "counter_ready": False,
        }
    else:
        avg = average_fsd(args.sample_cache_dir, history)
        artifact_rows = []
        for replica, train_minute in enumerate(train_minutes):
            artifact_rows.append(
                write_artifacts_from_fsd(
                    train_root,
                    fine_id(train_minute),
                    train_minute,
                    avg,
                    args.trace,
                    save_fsd=False,
                )
                | {"replica": replica}
            )
        artifact_info = aggregate_artifacts(artifact_rows)
        extra = {
            "avg_source_start": min(history),
            "avg_source_end": max(history),
            "avg_freq_points": len(avg),
            "avg_replicas": args.avg_replicas,
            "last_source_minute": np.nan,
            "last_replicas": np.nan,
            "used_sample_counter_cache": False,
            "counter_ready": False,
            **{f"avg_{k}": v for k, v in artifact_info.items()},
        }
    return train_root, train_minutes, {"train_root_build_sec": time.perf_counter() - t0, **extra}


def configure_runtime(args: argparse.Namespace) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Sliding full-stack simulation requires exactly one visible GPU; got {torch.cuda.device_count()}."
        )
    if bool(int(os.environ.get("ENABLE_TF32", "1"))):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    random.seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    torch.manual_seed(args.torch_seed)
    return torch.device("cuda:0")


def plot_outputs(out_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not rows:
        return
    minutes = np.array([int(r["minute"]) for r in rows], dtype=int)
    title_prefix = f"{args.trace} {args.strategy} K={args.history_size} E={args.epochs}"
    for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
        fig, ax = plt.subplots(figsize=(13, 4.6))
        ax.plot(minutes, [float(r[metric]) for r in rows], label="no gate", linewidth=1.25)
        ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="light residual gate", linewidth=1.25)
        if f"twoway_gate_{metric}" in rows[0]:
            ax.plot(minutes, [float(r[f"twoway_gate_{metric}"]) for r in rows], label="two-way mass gate", linewidth=1.25)
        if f"sample_shape_{metric}" in rows[0]:
            ax.plot(minutes, [float(r[f"sample_shape_{metric}"]) for r in rows], label="sample-shape blend", linewidth=1.25)
        ax.set_title(f"{title_prefix}: {ylabel} vs time")
        ax.set_xlabel("Time (minute)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"sliding_{args.strategy}_{metric}_vs_time.png", dpi=180)
        fig.savefig(out_dir / f"sliding_{args.strategy}_{metric}_vs_time.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(minutes, [float(r["train_root_build_sec"]) for r in rows], label="train root build")
    ax.plot(minutes, [float(r["counter_build_sec"]) for r in rows], label="counter build")
    ax.plot(minutes, [float(r["sft_total_sec"]) for r in rows], label="SFT")
    ax.plot(minutes, [float(r["control_train_sec"]) for r in rows], label="train path total")
    ax.axhline(args.gpu_deadline_sec, color="red", linestyle="--", linewidth=1.0, label="60s deadline")
    ax.set_title(f"{title_prefix}: control-plane timing")
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"sliding_{args.strategy}_timing_vs_time.png", dpi=180)
    fig.savefig(out_dir / f"sliding_{args.strategy}_timing_vs_time.pdf")
    plt.close(fig)


def summarize_sliding(rows: list[dict[str, object]], out_dir: Path, args: argparse.Namespace) -> None:
    summary: dict[str, object] = {
        "res": args.res,
        "trace": args.trace,
        "strategy": args.strategy,
        "history_size": args.history_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mode": args.mode,
        "sample_rate": args.sample_rate,
        "start_seed": args.start_seed,
        "end_seed": args.end_seed,
        "seed_count": args.end_seed - args.start_seed,
        "gpu_deadline_sec": args.gpu_deadline_sec,
        "use_sample_counter_cache": bool(args.use_sample_counter_cache),
        "n_windows": len(rows),
        "deadline_miss": int(sum(bool(r["deadline_miss"]) for r in rows)),
    }
    if rows:
        for key in [
            "train_root_build_sec",
            "counter_build_sec",
            "sft_total_sec",
            "control_train_sec",
            "infer_sec",
            "metric_eval_sec",
            "mrd",
            "wmrd",
            "gate_mrd",
            "gate_wmrd",
            "twoway_gate_mrd",
            "twoway_gate_wmrd",
            "sample_shape_mrd",
            "sample_shape_wmrd",
            "adaptive_mrd",
            "adaptive_wmrd",
            "gated_seed_fraction",
            "downscaled_seed_fraction",
            "raw_scale_mean",
            "twoway_scale_mean",
        ]:
            values = np.array([float(r[key]) for r in rows], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_p50"] = float(np.nanpercentile(values, 50))
            summary[f"{key}_p95"] = float(np.nanpercentile(values, 95))
            summary[f"{key}_max"] = float(np.nanmax(values))
    write_csv(out_dir / "summary.csv", [summary])


def main() -> None:
    args = parse_args()
    if args.history_size < 1:
        raise ValueError("--history-size must be >= 1")
    if args.strategy in {"window5", "avg5"} and args.history_size != 5:
        print(f"warning: strategy={args.strategy} with history_size={args.history_size}", flush=True)
    if args.sample_rate <= 0 or args.sample_rate > 1:
        raise ValueError("--sample-rate must be in (0, 1]")
    if args.last_replicas < 1:
        raise ValueError("--last-replicas must be >= 1")
    if args.last_replica_stride < 1:
        raise ValueError("--last-replica-stride must be >= 1")
    if args.avg_replicas < 1:
        raise ValueError("--avg-replicas must be >= 1")
    if args.avg_replica_stride < 1:
        raise ValueError("--avg-replica-stride must be >= 1")
    args.prep_workers = args.prep_workers or max(1, min(32, os.cpu_count() or 1))
    args.counter_threads = args.counter_threads or max(1, min(32, os.cpu_count() or 1))

    root = Path(__file__).resolve().parents[1]
    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    end_minute = last_raw_minute(raw_dir) if args.end_minute < 0 else args.end_minute
    start_minute = args.history_size if args.start_minute < 0 else args.start_minute
    if start_minute < args.history_size:
        raise ValueError("start minute must be >= history size so X-K..X-1 exists")
    if end_minute < start_minute:
        raise ValueError("end minute must be >= start minute")
    sample_end_exclusive = end_minute + 1 if args.sample_shape_weight > 0 else end_minute
    sample_minutes = list(range(start_minute - args.history_size, sample_end_exclusive))

    cache_rows = prepare_cache_range(args, raw_dir, sample_minutes)
    if args.prepare_cache_only:
        print(f"sample cache ready at {args.sample_cache_dir} rows={len(cache_rows)}", flush=True)
        return

    device = configure_runtime(args)
    counter_len = counter_len_for_res(args.res)
    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    pretrained_root = args.pretrained_root or (args.run_root / "pretrained" / args.res)
    if args.init_from == "pretrained":
        ckpts = {
            head: sorted((pretrained_root / spec["model_dir"]).glob("best_model_*.pth"))[0]
            for head, spec in HEADS.items()
        }
    else:
        ckpts = {head: None for head in HEADS}

    train_args = argparse.Namespace(
        train_source="fine-cache",
        holdout_frac=args.holdout_frac,
        torch_seed=args.torch_seed,
        mode=args.mode,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    metric_args = argparse.Namespace(
        gate_threshold=args.gate_threshold,
        gate_cap=args.gate_cap,
        gate_down_threshold=args.gate_down_threshold,
        gate_floor=args.gate_floor,
        sample_shape_weight=args.sample_shape_weight,
        sample_shape_max_freq=args.sample_shape_max_freq,
        sample_cache_dir=args.sample_cache_dir,
        adaptive_gate=args.adaptive_gate,
    )

    with (args.out_dir / "args.json").open("w") as f:
        json.dump(
            vars(args)
            | {
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "start_minute_effective": start_minute,
                "end_minute_effective": end_minute,
                "counter_len": counter_len,
                "pretrained_ckpts": {k: str(v) for k, v in ckpts.items()},
            },
            f,
            indent=2,
            default=str,
        )

    print(f"using one visible GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"trace={args.trace} strategy={args.strategy} K={args.history_size} "
        f"minutes={start_minute}:{end_minute} epochs={args.epochs} "
        f"seeds={args.start_seed}:{args.end_seed}",
        flush=True,
    )
    print(f"sample cache: {args.sample_cache_dir}", flush=True)
    print(f"init_from={args.init_from} ckpts: {ckpts}", flush=True)

    models = {head: make_model(HEADS[head]["out_dim"], ckpts[head], device) for head in HEADS}
    rows: list[dict[str, object]] = []
    gate_state: dict[str, list[float]] = {}

    for test_minute in range(start_minute, end_minute + 1):
        history = list(range(test_minute - args.history_size, test_minute))
        print("=" * 80, flush=True)
        print(f"minute={test_minute} history={history[0]}..{history[-1]}", flush=True)

        train_root, train_minutes, root_info = build_training_root(args, test_minute, history)
        if bool(root_info.get("counter_ready", False)) or bool(root_info.get("used_sample_counter_cache", False)):
            counter_sec = 0.0
        else:
            counter_sec = build_counter_store(
                root,
                train_root,
                args.trace,
                counter_len,
                args.start_seed,
                args.end_seed,
                args.gen_bin,
                args.counter_threads,
            )

        sft_info: dict[str, float] = {}
        sft_t0 = time.perf_counter()
        for head in ["1_10", "10_1e4"]:
            sft_info.update(train_head(models[head], counter_root, train_root, train_root, train_minutes, head, train_args, device))
        sft_total_sec = time.perf_counter() - sft_t0

        data_t0 = time.perf_counter()
        test_x = cached_counter(str(counter_root), test_minute)
        snapshot_load_sec = time.perf_counter() - data_t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_t0 = time.perf_counter()
        pred_1 = infer_head(models["1_10"], test_x, args.batch_size * 2, device)
        pred_2 = infer_head(models["10_1e4"], test_x, args.batch_size * 2, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_sec = time.perf_counter() - infer_t0

        metric_t0 = time.perf_counter()
        metric_row = final_metrics(
            final_dir, test_minute, pred_1, pred_2, metric_args, gate_state,
            packet_count=window_packet_count(raw_dir, test_minute, args.trace),
        )
        metric_sec = time.perf_counter() - metric_t0

        control_train_sec = float(root_info["train_root_build_sec"]) + counter_sec + sft_total_sec
        deadline_miss = control_train_sec > args.gpu_deadline_sec
        row: dict[str, object] = {
            "minute": test_minute,
            "history_start": min(history),
            "history_end": max(history),
            "history_count": len(history),
            "train_minutes": " ".join(str(v) for v in train_minutes),
            "train_count": len(train_minutes),
            "snapshot_load_sec": snapshot_load_sec,
            "counter_build_sec": counter_sec,
            "sft_total_sec": sft_total_sec,
            "control_train_sec": control_train_sec,
            "infer_sec": infer_sec,
            "metric_eval_sec": metric_sec,
            "deadline_miss": bool(deadline_miss),
            **root_info,
            **metric_row,
            **sft_info,
        }
        rows.append(row)
        write_csv(args.out_dir / "window_metrics.csv", rows)
        print(
            f"  train_root={root_info['train_root_build_sec']:.3f}s "
            f"counter={counter_sec:.3f}s sft={sft_total_sec:.3f}s "
            f"control={control_train_sec:.3f}s infer={infer_sec:.3f}s "
            f"wmrd={metric_row['wmrd']:.4f} gate={metric_row['gate_wmrd']:.4f} "
            f"mrd={metric_row['mrd']:.4f} gate_mrd={metric_row['gate_mrd']:.4f} "
            f"miss={deadline_miss}",
            flush=True,
        )

        if (
            not args.keep_train_roots
            and args.runtime_train_cache_dir is None
            and not bool(root_info.get("used_sample_counter_cache", False))
        ):
            shutil.rmtree(train_root, ignore_errors=True)

    summarize_sliding(rows, args.out_dir, args)
    plot_outputs(args.out_dir, rows, args)
    print(f"wrote sliding full-stack simulation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
