#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from packet_counts import window_packet_count
from simulate_realtime_online import (
    HEADS,
    cached_counter,
    final_metrics,
    infer_head,
    make_model,
    setup_imports,
    summarize,
    train_head,
    write_csv,
)
from simulate_sliding_fullstack_online import last_raw_minute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train once from a prepared fine-cache train root, then infer all later windows. "
            "Used to align one-time SFT with the first step of sliding continuous SFT."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-minutes", nargs="+", type=int, required=True)
    parser.add_argument("--start-minute", type=int, default=5)
    parser.add_argument("--end-minute", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["full", "head-only"], default="full")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--init-from", choices=["pretrained", "scratch"], default="scratch")
    parser.add_argument("--pretrained-root", type=Path, default=None)
    parser.add_argument("--gate-threshold", type=float, default=1.02)
    parser.add_argument("--gate-cap", type=float, default=1.20)
    parser.add_argument("--gate-down-threshold", type=float, default=0.0)
    parser.add_argument("--gate-floor", type=float, default=0.50)
    parser.add_argument("--sample-shape-weight", type=float, default=0.0)
    parser.add_argument("--sample-shape-max-freq", type=int, default=100)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--adaptive-gate", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"requires exactly one visible GPU, got {torch.cuda.device_count()}")
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


def main() -> None:
    args = parse_args()
    device = configure(args)
    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    end_minute = last_raw_minute(raw_dir) if args.end_minute < 0 else args.end_minute
    if args.start_minute > end_minute:
        raise ValueError("--start-minute must be <= effective end minute")

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

    args.train_source = "fine-cache"
    args.fine_root = args.train_root
    args.policy = "fixed-train-root-once"
    args.gpu_deadline_sec = 60.0
    args.window_size = len(args.train_minutes)
    args.fixed_train_start = min(args.train_minutes)
    args.fixed_train_end = max(args.train_minutes)
    with (args.out_dir / "args.json").open("w") as f:
        json.dump(
            vars(args)
            | {
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "end_minute_effective": end_minute,
                "pretrained_ckpts": {k: str(v) for k, v in ckpts.items()},
            },
            f,
            indent=2,
            default=str,
        )

    print(f"using one visible GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"trace={args.trace} init_from={args.init_from} train_root={args.train_root} "
        f"train_minutes={args.train_minutes} epochs={args.epochs}",
        flush=True,
    )
    models = {head: make_model(HEADS[head]["out_dim"], ckpts[head], device) for head in HEADS}

    sft_info: dict[str, float] = {}
    sft_t0 = time.perf_counter()
    for head in ["1_10", "10_1e4"]:
        sft_info.update(train_head(models[head], counter_root, args.train_root, args.train_root, args.train_minutes, head, args, device))
    bootstrap_sft_sec = time.perf_counter() - sft_t0
    print(f"bootstrap_sft={bootstrap_sft_sec:.3f}s", flush=True)

    rows: list[dict[str, object]] = []
    gate_state: dict[str, list[float]] = {}
    for minute in range(args.start_minute, end_minute + 1):
        data_t0 = time.perf_counter()
        test_x = cached_counter(str(counter_root), minute)
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
            final_dir, minute, pred_1, pred_2, args, gate_state,
            packet_count=window_packet_count(raw_dir, minute, args.trace),
        )
        metric_sec = time.perf_counter() - metric_t0
        row: dict[str, object] = {
            "minute": minute,
            "history_start": min(args.train_minutes),
            "history_end": max(args.train_minutes),
            "history_count": len(args.train_minutes),
            "train_minutes": " ".join(str(v) for v in args.train_minutes),
            "train_count": len(args.train_minutes),
            "snapshot_load_sec": snapshot_load_sec,
            "counter_build_sec": 0.0,
            "sft_total_sec": bootstrap_sft_sec if minute == args.start_minute else 0.0,
            "bootstrap_sft_sec": bootstrap_sft_sec,
            "control_train_sec": bootstrap_sft_sec if minute == args.start_minute else 0.0,
            "infer_sec": infer_sec,
            "metric_eval_sec": metric_sec,
            "deadline_miss": bool(bootstrap_sft_sec > 60.0) if minute == args.start_minute else False,
            **metric_row,
            **sft_info,
        }
        rows.append(row)
        write_csv(args.out_dir / "window_metrics.csv", rows)
        print(f"minute={minute} wmrd={metric_row['wmrd']:.4f} mrd={metric_row['mrd']:.4f}", flush=True)

    summarize(rows, args.out_dir, args)
    print(f"wrote fixed train-root once simulation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
