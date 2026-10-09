#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Block-continuous online simulation using cached fine_dataset_* training data. "
            "This matches the original sample->fine counter training protocol."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--gpu-deadline-sec", type=float, default=60.0)
    parser.add_argument("--train-block-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["full", "head-only"], default="full")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--adaptive-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=1.02)
    parser.add_argument("--gate-cap", type=float, default=1.20)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def minute_from_name(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def task_groups(tasks: dict[str, bool]) -> list[tuple[list[str], list[str]]]:
    items = list(tasks.items())
    groups: list[tuple[list[str], list[str]]] = []
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
        if train or test:
            groups.append((train, test))
    return groups


def configure_runtime(args: argparse.Namespace) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Block-continuous simulation requires exactly one visible GPU; got {torch.cuda.device_count()}."
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


def main() -> None:
    args = parse_args()
    if args.train_block_size < 1:
        raise ValueError("--train-block-size must be >= 1")
    device = configure_runtime(args)
    args.policy = "block-continuous"
    args.window_size = args.train_block_size
    args.train_source = "fine-cache"
    args.fixed_train_start = 0
    args.fixed_train_end = args.train_block_size - 1
    args.packet_size = 0
    args.key_offset = -1
    args.key_length = 0

    exp_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_exp"
    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    fine_root = exp_dir / "tr_ts_finetuned_continue"
    args.fine_root = fine_root
    label_root = args.out_dir / "unused_sample_labels"
    pretrained_root = args.run_root / "pretrained" / args.res
    ckpts = {
        head: sorted((pretrained_root / spec["model_dir"]).glob("best_model_*.pth"))[0]
        for head, spec in HEADS.items()
    }
    with (exp_dir / "train_test_name_key.json").open() as f:
        groups = task_groups(json.load(f))
    if not groups:
        raise RuntimeError(f"no task groups in {exp_dir / 'train_test_name_key.json'}")
    max_train_names = max(len(train) for train, _ in groups)
    if args.train_block_size > max_train_names:
        raise ValueError(
            f"--train-block-size={args.train_block_size} exceeds cached train names per block ({max_train_names})."
        )

    with (args.out_dir / "args.json").open("w") as f:
        json.dump(vars(args) | {"visible_gpu_name": torch.cuda.get_device_name(0)}, f, indent=2, default=str)

    print(f"using one visible GPU: {torch.cuda.get_device_name(0)}")
    print(f"trace={args.trace} K={args.train_block_size} epochs={args.epochs} lr={args.lr}")
    print(f"fine cache root: {fine_root}")
    print(f"pretrained ckpts: {ckpts}")

    models = {head: make_model(HEADS[head]["out_dim"], ckpts[head], device) for head in HEADS}
    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    gate_state: dict[str, list[float]] = {}
    selected_groups = groups[: args.max_blocks] if args.max_blocks else groups

    for block_index, (train_names, test_names) in enumerate(selected_groups):
        train_minutes = [minute_from_name(name) for name in train_names[: args.train_block_size]]
        if not train_minutes:
            continue
        block_name = train_names[0]
        print("=" * 80, flush=True)
        print(
            f"block={block_index}/{len(selected_groups)-1} train={train_names[:args.train_block_size]} "
            f"test={test_names[:1]}..{test_names[-1:] if test_names else []}",
            flush=True,
        )

        sft_info: dict[str, float] = {}
        sft_t0 = time.perf_counter()
        for head in ["1_10", "10_1e4"]:
            sft_info.update(train_head(models[head], counter_root, label_root, fine_root, train_minutes, head, args, device))
        block_sft_sec = time.perf_counter() - sft_t0
        deadline_miss = block_sft_sec > args.gpu_deadline_sec
        block_row: dict[str, object] = {
            "block_index": block_index,
            "block_name": block_name,
            "train_start_minute": min(train_minutes),
            "train_end_minute": max(train_minutes),
            "train_count": len(train_minutes),
            "test_count": len(test_names),
            "block_sft_sec": block_sft_sec,
            "sft_deadline_miss": bool(deadline_miss),
            **sft_info,
        }
        block_rows.append(block_row)
        write_csv(args.out_dir / "block_timing.csv", block_rows)
        print(f"  block_sft={block_sft_sec:.3f}s miss={deadline_miss}", flush=True)

        for test_name in test_names:
            minute = minute_from_name(test_name)
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
            first_test = test_name == test_names[0]
            row: dict[str, object] = {
                "minute": minute,
                "block_index": block_index,
                "history_start": min(train_minutes),
                "history_end": max(train_minutes),
                "history_count": len(train_minutes),
                "sample_label_build_sec": 0.0,
                "sampled_flows": np.nan,
                "sampled_packets": np.nan,
                "full_flows": np.nan,
                "snapshot_load_sec": snapshot_load_sec,
                "infer_sec": infer_sec,
                "metric_eval_sec": metric_sec,
                "sft_total_sec": block_sft_sec if first_test else 0.0,
                "bootstrap_sft_sec": 0.0,
                "bootstrap_deadline_budget_sec": 0.0,
                "gpu_control_total_sec": infer_sec + (block_sft_sec if first_test else 0.0),
                "deadline_miss": bool(deadline_miss and first_test),
                "model_lag_windows": minute - max(train_minutes),
                **metric_row,
                **sft_info,
            }
            rows.append(row)
            write_csv(args.out_dir / "window_metrics.csv", rows)
            print(
                f"  test minute={minute} infer={infer_sec:.3f}s "
                f"wmrd={metric_row['wmrd']:.4f} gate={metric_row['adaptive_wmrd']:.4f}",
                flush=True,
            )

    summarize(rows, args.out_dir, args)
    print(f"wrote block-continuous simulation to {args.out_dir}")


if __name__ == "__main__":
    main()
