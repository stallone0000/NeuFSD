#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
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
from simulate_block_continuous_finecache import minute_from_name, task_groups
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-stack online simulation with no cached fine training data. "
            "Each block rebuilds sample-restored FSD labels, synthetic fine streams, "
            "fine counters, and SFT data from raw one-minute traces."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", required=True, choices=["caida_2016", "caida_2018", "caida_2018_new", "imc", "mawi"])
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--gpu-deadline-sec", type=float, default=60.0)
    parser.add_argument("--train-block-size", type=int, default=5)
    parser.add_argument("--max-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["full", "head-only"], default="full")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=400)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--counter-threads", type=int, default=0)
    parser.add_argument("--gen-bin", type=Path, default=Path("run_tools/gen_counter_store"))
    parser.add_argument("--adaptive-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=1.02)
    parser.add_argument("--gate-cap", type=float, default=1.20)
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def counter_len_for_res(res: str) -> int:
    if res == "64_64":
        return 4096
    if res == "128_128":
        return 16384
    raise ValueError(f"unsupported --res={res}")


def write_label_dirs(block_root: Path) -> None:
    for folder in ["1_10_chazhi", "10_1e4_chazhi", "1_10_real", "10_1e4_real"]:
        (block_root / folder).mkdir(parents=True, exist_ok=True)


def fake_packet(flow_index: int, minute: int, packet_size: int, key_offset: int, key_length: int) -> bytes:
    key_int = (int(minute) << 40) + int(flow_index)
    key = key_int.to_bytes(key_length, byteorder="big", signed=False)
    prefix = b"\x00" * key_offset
    suffix = b"\x00" * (packet_size - key_offset - key_length)
    return prefix + key + suffix


def prepare_one_window(
    raw_path_text: str,
    block_root_text: str,
    minute: int,
    trace: str,
    sample_rate: float,
    sample_seed: int,
) -> dict[str, object]:
    raw_path = Path(raw_path_text)
    block_root = Path(block_root_text)
    packet_size, key_offset, key_length = default_packet_format(trace)
    fine_name = fine_id(minute)
    fine_path = block_root / "fine_dat" / f"{fine_name}.dat"

    write_label_dirs(block_root)
    fine_path.parent.mkdir(parents=True, exist_ok=True)

    parse_t0 = time.perf_counter()
    flow_counts = parse_packet_keys(raw_path, packet_size, key_offset, key_length)
    parse_sec = time.perf_counter() - parse_t0

    sample_t0 = time.perf_counter()
    multiplier = 1.0 / sample_rate
    synth_multiplier = int(round(multiplier))
    if not np.isclose(multiplier, synth_multiplier):
        raise ValueError("full-stack fine stream synthesis expects 1/sample_rate to be close to an integer")
    sampled_fsd: dict[int, float] = defaultdict(float)
    sampled_flows = 0
    sampled_packets = 0
    for key, count in flow_counts.items():
        if stable_sample(key, minute, sample_rate, sample_seed):
            sampled_fsd[int(count)] += multiplier
            sampled_flows += 1
            sampled_packets += int(count)
    sample_sec = time.perf_counter() - sample_t0

    label_t0 = time.perf_counter()
    sorted_fsd = sorted(sampled_fsd.items())
    one_targets = np.arange(1, 11, dtype=float)
    ten_targets = np.concatenate((np.arange(11, 1001, dtype=float), np.arange(1001, 10001, 100, dtype=float)))
    one = interp_dense(sorted_fsd, one_targets)
    ten = interp_dense(sorted_fsd, ten_targets)
    np.save(block_root / "1_10_chazhi" / f"{fine_name}.npy", one)
    np.save(block_root / "10_1e4_chazhi" / f"{fine_name}.npy", ten)
    real_small = np.array([(freq, count) for freq, count in sorted_fsd if freq <= 10], dtype=np.float32)
    real_large = np.array([(freq, count) for freq, count in sorted_fsd if 10 < freq <= 10000], dtype=np.float32)
    np.save(block_root / "1_10_real" / f"{fine_name}.npy", real_small)
    np.save(block_root / "10_1e4_real" / f"{fine_name}.npy", real_large)
    label_sec = time.perf_counter() - label_t0

    synth_t0 = time.perf_counter()
    fine_packets = 0
    fine_flows = 0
    flow_index = 1
    with fine_path.open("wb") as out:
        for size, flow_count in sorted_fsd:
            integer_flows = int(round(float(flow_count)))
            for _ in range(integer_flows):
                packet = fake_packet(flow_index, minute, packet_size, key_offset, key_length)
                out.write(packet * int(size))
                fine_packets += int(size)
                fine_flows += 1
                flow_index += 1
    synth_sec = time.perf_counter() - synth_t0

    return {
        "minute": minute,
        "raw_path": str(raw_path),
        "fine_path": str(fine_path),
        "parse_sec": parse_sec,
        "sample_reconstruct_sec": sample_sec,
        "label_write_sec": label_sec,
        "synth_stream_write_sec": synth_sec,
        "full_flows": len(flow_counts),
        "sampled_flows": sampled_flows,
        "sampled_packets": sampled_packets,
        "fine_flows": fine_flows,
        "fine_packets": fine_packets,
        "raw_packets": raw_path.stat().st_size // packet_size,
        "fine_bytes": fine_path.stat().st_size,
    }


def build_counter_store(
    root: Path,
    block_root: Path,
    trace: str,
    counter_len: int,
    start_seed: int,
    end_seed: int,
    gen_bin: Path,
    threads: int,
) -> float:
    cmd = [
        sys.executable,
        str(root / "scripts" / "build_counter_store.py"),
        "--input-dir",
        str(block_root / "fine_dat"),
        "--out-root",
        str(block_root),
        "--trace-format",
        trace,
        "--counter-len",
        str(counter_len),
        "--start-seed",
        str(start_seed),
        "--end-seed",
        str(end_seed),
        "--gen-bin",
        str(gen_bin),
        "--tmp-dir",
        str(block_root / "_counter_blocks"),
    ]
    env = os.environ.copy()
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=root, env=env, check=True)
    return time.perf_counter() - t0


def configure_runtime(args: argparse.Namespace) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Full-stack online simulation requires exactly one visible GPU; got {torch.cuda.device_count()}."
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


def load_groups(run_root: Path, res: str, trace: str) -> list[tuple[list[str], list[str]]]:
    exp_dir = run_root / "run_full_matrix" / f"{res}_{trace}_exp"
    with (exp_dir / "train_test_name_key.json").open() as f:
        return task_groups(json.load(f))


def plot_outputs(out_dir: Path, rows: list[dict[str, object]], block_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    if rows:
        minutes = np.array([int(r["minute"]) for r in rows])
        for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
            fig, ax = plt.subplots(figsize=(12, 4.2))
            ax.plot(minutes, [float(r[metric]) for r in rows], label="raw")
            ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="gate")
            if "adaptive_mrd" in rows[0]:
                ax.plot(minutes, [float(r[f"adaptive_{metric}"]) for r in rows], label="adaptive gate")
            ax.set_xlabel("Time (minute)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(out_dir / f"fullstack_{metric}_vs_time.png", dpi=180)
            fig.savefig(out_dir / f"fullstack_{metric}_vs_time.pdf")
            plt.close(fig)

    if block_rows:
        blocks = np.array([int(r["block_index"]) for r in block_rows])
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(blocks, [float(r["prep_total_sec"]) for r in block_rows], label="sample/FSD/synth labels")
        ax.plot(blocks, [float(r["counter_build_sec"]) for r in block_rows], label="fine counter build")
        ax.plot(blocks, [float(r["block_sft_sec"]) for r in block_rows], label="SFT")
        ax.plot(blocks, [float(r["end_to_end_train_sec"]) for r in block_rows], label="end-to-end train path")
        ax.axhline(args.gpu_deadline_sec, color="red", linestyle="--", linewidth=1.0, label="60s")
        ax.set_xlabel("Train block")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncols=2)
        fig.tight_layout()
        fig.savefig(out_dir / "fullstack_timing_by_block.png", dpi=180)
        fig.savefig(out_dir / "fullstack_timing_by_block.pdf")
        plt.close(fig)


def summarize_fullstack(out_dir: Path, rows: list[dict[str, object]], block_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    summary: dict[str, object] = {
        "res": args.res,
        "trace": args.trace,
        "train_block_size": args.train_block_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mode": args.mode,
        "sample_rate": args.sample_rate,
        "start_seed": args.start_seed,
        "end_seed": args.end_seed,
        "seed_count": args.end_seed - args.start_seed,
        "prep_workers": args.prep_workers,
        "counter_threads": args.counter_threads,
        "n_blocks": len(block_rows),
        "n_windows": len(rows),
        "block_60s_miss": int(sum(bool(r["block_60s_miss"]) for r in block_rows)),
        "block_amortized_miss": int(sum(bool(r["block_amortized_miss"]) for r in block_rows)),
    }
    for source, keys in [
        (
            block_rows,
            [
                "prep_total_sec",
                "counter_build_sec",
                "block_sft_sec",
                "end_to_end_train_sec",
                "train_path_sec_per_input_minute",
                "parse_sec_sum",
                "sample_reconstruct_sec_sum",
                "synth_stream_write_sec_sum",
            ],
        ),
        (rows, ["infer_sec", "metric_eval_sec", "mrd", "wmrd", "gate_mrd", "gate_wmrd", "adaptive_mrd", "adaptive_wmrd"]),
    ]:
        if not source:
            continue
        for key in keys:
            values = np.array([float(r[key]) for r in source], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_p50"] = float(np.nanpercentile(values, 50))
            summary[f"{key}_p95"] = float(np.nanpercentile(values, 95))
            summary[f"{key}_max"] = float(np.nanmax(values))
    write_csv(out_dir / "summary.csv", [summary])


def main() -> None:
    args = parse_args()
    if args.train_block_size < 1:
        raise ValueError("--train-block-size must be >= 1")
    if args.sample_rate <= 0 or args.sample_rate > 1:
        raise ValueError("--sample-rate must be in (0, 1]")
    args.prep_workers = args.prep_workers or min(args.train_block_size, max(1, (os.cpu_count() or 4) // 4))
    args.counter_threads = args.counter_threads or max(1, min(32, os.cpu_count() or 1))
    root = Path(__file__).resolve().parents[1]
    device = configure_runtime(args)
    counter_len = counter_len_for_res(args.res)
    packet_size, key_offset, key_length = default_packet_format(args.trace)

    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    pretrained_root = args.run_root / "pretrained" / args.res
    ckpts = {
        head: sorted((pretrained_root / spec["model_dir"]).glob("best_model_*.pth"))[0]
        for head, spec in HEADS.items()
    }
    groups = load_groups(args.run_root, args.res, args.trace)
    selected_groups = groups[: args.max_blocks] if args.max_blocks else groups
    max_train_names = max(len(train) for train, _ in groups)
    if args.train_block_size > max_train_names:
        raise ValueError(f"--train-block-size={args.train_block_size} exceeds available train windows {max_train_names}")

    with (args.out_dir / "args.json").open("w") as f:
        json.dump(
            vars(args)
            | {
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "packet_size": packet_size,
                "key_offset": key_offset,
                "key_length": key_length,
                "counter_len": counter_len,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"using one visible GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"trace={args.trace} K={args.train_block_size} epochs={args.epochs} "
        f"seeds={args.start_seed}:{args.end_seed} prep_workers={args.prep_workers} "
        f"counter_threads={args.counter_threads}",
        flush=True,
    )
    print(f"pretrained ckpts: {ckpts}", flush=True)

    models = {head: make_model(HEADS[head]["out_dim"], ckpts[head], device) for head in HEADS}
    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    prep_rows: list[dict[str, object]] = []
    gate_state: dict[str, list[float]] = {}
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
        adaptive_gate=args.adaptive_gate,
    )

    for block_index, (train_names, test_names) in enumerate(selected_groups):
        train_minutes = [minute_from_name(name) for name in train_names[: args.train_block_size]]
        block_root = args.out_dir / "runtime_blocks" / f"block_{block_index:04d}"
        if block_root.exists() and not args.keep_runtime:
            shutil.rmtree(block_root)
        block_root.mkdir(parents=True, exist_ok=True)

        print("=" * 80, flush=True)
        print(f"block={block_index} train_minutes={train_minutes} test_count={len(test_names)}", flush=True)

        prep_t0 = time.perf_counter()
        futures = []
        with ProcessPoolExecutor(max_workers=args.prep_workers) as pool:
            for minute in train_minutes:
                raw_path = raw_dir / f"dataset_{minute:04d}.dat"
                if not raw_path.exists():
                    raise FileNotFoundError(raw_path)
                futures.append(
                    pool.submit(
                        prepare_one_window,
                        str(raw_path),
                        str(block_root),
                        minute,
                        args.trace,
                        args.sample_rate,
                        args.sample_seed,
                    )
                )
            for fut in as_completed(futures):
                row = fut.result()
                prep_rows.append(row)
                write_csv(args.out_dir / "prep_timing.csv", prep_rows)
                print(
                    f"  prepared minute={row['minute']} parse={row['parse_sec']:.3f}s "
                    f"sample={row['sample_reconstruct_sec']:.3f}s synth={row['synth_stream_write_sec']:.3f}s "
                    f"sampled_flows={row['sampled_flows']}",
                    flush=True,
                )
        prep_wall_sec = time.perf_counter() - prep_t0
        block_prep = [r for r in prep_rows if int(r["minute"]) in train_minutes]

        counter_sec = build_counter_store(
            root,
            block_root,
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
            sft_info.update(train_head(models[head], counter_root, block_root, block_root, train_minutes, head, train_args, device))
        block_sft_sec = time.perf_counter() - sft_t0

        prep_sum_keys = [
            "parse_sec",
            "sample_reconstruct_sec",
            "label_write_sec",
            "synth_stream_write_sec",
            "full_flows",
            "sampled_flows",
            "sampled_packets",
            "fine_flows",
            "fine_packets",
            "raw_packets",
            "fine_bytes",
        ]
        sums = {f"{key}_sum": float(sum(float(r[key]) for r in block_prep)) for key in prep_sum_keys}
        end_to_end_train_sec = prep_wall_sec + counter_sec + block_sft_sec
        amortized_budget = args.gpu_deadline_sec * len(train_minutes)
        block_row: dict[str, object] = {
            "block_index": block_index,
            "train_start_minute": min(train_minutes),
            "train_end_minute": max(train_minutes),
            "train_count": len(train_minutes),
            "test_count": len(test_names),
            "prep_wall_sec": prep_wall_sec,
            "prep_total_sec": prep_wall_sec,
            "counter_build_sec": counter_sec,
            "block_sft_sec": block_sft_sec,
            "end_to_end_train_sec": end_to_end_train_sec,
            "train_path_sec_per_input_minute": end_to_end_train_sec / max(1, len(train_minutes)),
            "block_60s_miss": end_to_end_train_sec > args.gpu_deadline_sec,
            "block_amortized_budget_sec": amortized_budget,
            "block_amortized_miss": end_to_end_train_sec > amortized_budget,
            **sums,
            **sft_info,
        }
        block_rows.append(block_row)
        write_csv(args.out_dir / "block_timing.csv", block_rows)
        print(
            f"  full train path: prep={prep_wall_sec:.3f}s counter={counter_sec:.3f}s "
            f"sft={block_sft_sec:.3f}s total={end_to_end_train_sec:.3f}s "
            f"miss60={block_row['block_60s_miss']} missKmin={block_row['block_amortized_miss']}",
            flush=True,
        )

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
                final_dir, minute, pred_1, pred_2, metric_args, gate_state,
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
                "snapshot_load_sec": snapshot_load_sec,
                "infer_sec": infer_sec,
                "metric_eval_sec": metric_sec,
                "sft_total_sec": block_sft_sec if first_test else 0.0,
                "fullstack_train_path_sec": end_to_end_train_sec if first_test else 0.0,
                "deadline_miss": bool(block_row["block_60s_miss"] and first_test),
                "model_lag_windows": minute - max(train_minutes),
                **metric_row,
                **sft_info,
            }
            rows.append(row)
            write_csv(args.out_dir / "window_metrics.csv", rows)
            print(
                f"  test minute={minute} infer={infer_sec:.3f}s "
                f"wmrd={metric_row['wmrd']:.4f} gate={metric_row['gate_wmrd']:.4f} "
                f"adaptive={metric_row['adaptive_wmrd']:.4f}",
                flush=True,
            )

    summarize_fullstack(args.out_dir, rows, block_rows, args)
    plot_outputs(args.out_dir, rows, block_rows, args)
    print(f"wrote full-stack online simulation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
