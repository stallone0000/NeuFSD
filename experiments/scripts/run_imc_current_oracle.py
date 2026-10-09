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
from pathlib import Path

import numpy as np
import torch

from packet_counts import window_packet_count
from perf_eval_first50_runner import (
    TRACE_SHORT,
    checkpoint_for,
    default_config_dir,
    make_model,
)
from simulate_fullstack_online import build_counter_store, counter_len_for_res
from simulate_realtime_online import HEADS, cached_counter, final_metrics, infer_head, setup_imports, train_head, write_csv
from simulate_sliding_fullstack_online import (
    LABEL_DIRS,
    cache_complete,
    copy_labels,
    fine_id,
    link_or_copy,
    load_cached_fsd,
    prepare_cache_range,
    train_root_complete,
    write_artifacts_from_fsd,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "IMC current-window oracle: sample the current window, train on that "
            "sample-restored FSD, and test on the same current window."
        )
    )
    ap.add_argument("--run-root", type=Path, default=Path("mainonly_runs_20260623"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--res", default="64_64")
    ap.add_argument("--trace", default="imc", choices=["imc"])
    ap.add_argument("--start-minute", type=int, default=5)
    ap.add_argument("--end-minute", type=int, default=97)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--sample-rate", type=float, default=0.1)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--end-seed", type=int, default=400)
    ap.add_argument("--current-replicas", type=int, default=5)
    ap.add_argument("--current-replica-stride", type=int, default=10000)
    ap.add_argument("--prep-workers", type=int, default=32)
    ap.add_argument("--counter-threads", type=int, default=8)
    ap.add_argument("--gen-bin", type=Path, default=Path("run_tools/gen_counter_store"))
    ap.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    ap.add_argument("--sample-cache-dir", type=Path, default=None)
    ap.add_argument("--runtime-train-cache-dir", type=Path, default=None)
    ap.add_argument("--pretrained-root", type=Path, default=None)
    ap.add_argument("--phi", type=int, default=1000)
    ap.add_argument("--input-mode", choices=["raw", "log1p", "log1p_total"], default="raw")
    ap.add_argument("--gate-threshold", type=float, default=1.02)
    ap.add_argument("--gate-cap", type=float, default=1.20)
    ap.add_argument("--gate-down-threshold", type=float, default=0.98)
    ap.add_argument("--gate-floor", type=float, default=0.70)
    ap.add_argument("--sample-shape-weight", type=float, default=0.5)
    ap.add_argument("--sample-shape-max-freq", type=int, default=100)
    ap.add_argument("--adaptive-gate", action="store_true")
    ap.add_argument("--prepare-cache-only", action="store_true")
    ap.add_argument("--force-cache-rebuild", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def configure(args: argparse.Namespace, root: Path) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    os.environ["COUNTER_BACKEND"] = "memmap"
    os.environ["COUNTER_INPUT_MODE"] = args.input_mode
    os.environ["COUNTER_FEATURE_MODE"] = "three"
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


def build_current_train_root(args: argparse.Namespace, minute: int, root: Path) -> tuple[Path, dict[str, object]]:
    train_root = args.runtime_train_cache_dir / f"minute_{minute:04d}_current"
    train_minutes = [minute + replica * args.current_replica_stride for replica in range(args.current_replicas)]
    if train_root_complete(train_root, train_minutes):
        return train_root, {
            "train_root_build_sec": 0.0,
            "counter_ready": True,
            "used_train_cache": True,
            "current_replicas": args.current_replicas,
        }

    if train_root.exists():
        shutil.rmtree(train_root)
    for folder in LABEL_DIRS:
        (train_root / folder).mkdir(parents=True, exist_ok=True)
    (train_root / "fine_dat").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.current_replicas == 1:
        link_or_copy(
            args.sample_cache_dir / "fine_dat" / f"{fine_id(minute)}.dat",
            train_root / "fine_dat" / f"{fine_id(minute)}.dat",
        )
        copy_labels(args.sample_cache_dir, train_root, minute)
    else:
        source_fsd = load_cached_fsd(args.sample_cache_dir, minute)
        for train_minute in train_minutes:
            write_artifacts_from_fsd(
                train_root,
                fine_id(train_minute),
                train_minute,
                source_fsd,
                args.trace,
                save_fsd=False,
            )
    return train_root, {
        "train_root_build_sec": time.perf_counter() - t0,
        "counter_ready": False,
        "used_train_cache": False,
        "current_replicas": args.current_replicas,
    }


def train_eval_minute(args: argparse.Namespace, root: Path, device: torch.device, minute: int) -> dict[str, object]:
    if not cache_complete(args.sample_cache_dir, minute):
        raise FileNotFoundError(f"sample cache incomplete for minute {minute}: {args.sample_cache_dir}")

    counter_len = counter_len_for_res(args.res)
    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    ckpts = {head: checkpoint_for(args.pretrained_root, head) for head in HEADS}
    train_root, root_info = build_current_train_root(args, minute, root)
    train_minutes = [minute + replica * args.current_replica_stride for replica in range(args.current_replicas)]

    if bool(root_info["counter_ready"]):
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

    train_args = argparse.Namespace(
        train_source="fine-cache",
        holdout_frac=args.holdout_frac,
        torch_seed=args.torch_seed,
        mode="full",
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    models = {head: make_model(args, HEADS[head]["out_dim"], ckpts[head], device) for head in HEADS}
    sft_t0 = time.perf_counter()
    sft_info: dict[str, float] = {}
    for head in ["1_10", "10_1e4"]:
        sft_info.update(train_head(models[head], counter_root, train_root, train_root, train_minutes, head, train_args, device))
    sft_total_sec = time.perf_counter() - sft_t0

    load_t0 = time.perf_counter()
    test_x = cached_counter(str(counter_root), minute)
    snapshot_load_sec = time.perf_counter() - load_t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_t0 = time.perf_counter()
    pred_1 = infer_head(models["1_10"], test_x, args.batch_size * 2, device)
    pred_2 = infer_head(models["10_1e4"], test_x, args.batch_size * 2, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_sec = time.perf_counter() - infer_t0

    metric_t0 = time.perf_counter()
    metrics = final_metrics(
        final_dir, minute, pred_1, pred_2, args, {},
        packet_count=window_packet_count(args.data_full_root / args.trace / "caida_1min_split", minute, args.trace),
    )
    metric_sec = time.perf_counter() - metric_t0

    del models
    torch.cuda.empty_cache()

    return {
        "trace": args.trace,
        "strategy": "current_oracle",
        "minute": minute,
        "epochs": args.epochs,
        "sample_rate": args.sample_rate,
        "sample_shape_weight": args.sample_shape_weight,
        "train_minutes": " ".join(str(v) for v in train_minutes),
        "train_count": len(train_minutes),
        "counter_build_sec": counter_sec,
        "sft_total_sec": sft_total_sec,
        "snapshot_load_sec": snapshot_load_sec,
        "infer_sec": infer_sec,
        "metric_eval_sec": metric_sec,
        "decode_sec": infer_sec + metric_sec,
        **root_info,
        **metrics,
        **sft_info,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.run_root = args.run_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.config_dir = default_config_dir(root, args.res, args.trace)
    args.variant = "original"
    args.sample_cache_dir = args.sample_cache_dir or (
        args.run_root / "imc_current_oracle_20260630" / "sample_cache_imc_p10"
    )
    args.runtime_train_cache_dir = args.runtime_train_cache_dir or (
        args.run_root / "imc_current_oracle_20260630" / "train_cache_current"
    )
    args.pretrained_root = args.pretrained_root or (args.run_root / "pretrain_fourdatasets" / "pretrained" / args.res)
    args.gen_bin = (root / args.gen_bin).resolve() if not args.gen_bin.is_absolute() else args.gen_bin
    args.data_full_root = (root / args.data_full_root).resolve() if not args.data_full_root.is_absolute() else args.data_full_root

    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    sample_minutes = list(range(args.start_minute, args.end_minute + 1))
    args.sample_cache_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_train_cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_cache_range(args, raw_dir, sample_minutes)
    if args.prepare_cache_only:
        print(f"prepared sample cache: {args.sample_cache_dir}", flush=True)
        return

    device = configure(args, root)
    done = args.out_dir / "DONE"
    if done.exists() and not args.force:
        print(f"skip complete run: {args.out_dir}", flush=True)
        return

    rows: list[dict[str, object]] = []
    csv_path = args.out_dir / "window_metrics.csv"
    if csv_path.exists() and not args.force:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    completed = {int(row["minute"]) for row in rows}

    with (args.out_dir / "args.json").open("w") as f:
        json.dump(
            vars(args)
            | {
                "config_dir": str(args.config_dir),
                "visible_gpu": torch.cuda.get_device_name(0),
                "pretrained_root": str(args.pretrained_root),
            },
            f,
            indent=2,
            default=str,
        )

    print(
        f"current_oracle trace={args.trace} minutes={args.start_minute}-{args.end_minute} "
        f"epochs={args.epochs} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for minute in sample_minutes:
        if minute in completed:
            continue
        row = train_eval_minute(args, root, device, minute)
        rows.append(row)
        write_csv(csv_path, rows)
        print(
            f"minute={minute} sft={row['sft_total_sec']:.3f}s "
            f"wmrd={row['wmrd']:.4f} mrd={row['mrd']:.4f}",
            flush=True,
        )

    if rows:
        numeric = lambda key: [float(row[key]) for row in rows]
        write_csv(
            args.out_dir / "summary.csv",
            [
                {
                    "trace": args.trace,
                    "strategy": "current_oracle",
                    "epochs": args.epochs,
                    "sample_rate": args.sample_rate,
                    "n_windows": len(rows),
                    "wmrd_mean": float(np.mean(numeric("wmrd"))),
                    "mrd_mean": float(np.mean(numeric("mrd"))),
                    "sample_shape_wmrd_mean": float(np.mean(numeric("sample_shape_wmrd"))),
                    "sample_shape_mrd_mean": float(np.mean(numeric("sample_shape_mrd"))),
                    "twoway_gate_wmrd_mean": float(np.mean(numeric("twoway_gate_wmrd"))),
                    "twoway_gate_mrd_mean": float(np.mean(numeric("twoway_gate_mrd"))),
                    "sft_total_sec_mean": float(np.mean(numeric("sft_total_sec"))),
                    "decode_sec_mean": float(np.mean(numeric("decode_sec"))),
                }
            ],
        )
    done.write_text("done\n")


if __name__ == "__main__":
    main()
