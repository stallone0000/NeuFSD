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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from simulate_fullstack_online import build_counter_store, counter_len_for_res
from simulate_realtime_online import (
    HEADS,
    cached_counter,
    infer_head,
    interp_extrap,
    setup_imports,
    train_head,
    write_csv,
)
from simulate_sliding_fullstack_online import build_training_root, prepare_cache_range


ANCHOR_SIZES = np.concatenate(
    (
        np.arange(1, 11, dtype=float),
        np.arange(11, 1001, dtype=float),
        np.arange(1001, 10001, 100, dtype=float),
    )
)


TRACE_CONFIG = {
    "caida_2016": "caida_2016",
    "caida_2018": "caida_2018",
    "imc": "caida_org",
    "mawi": "caida_org",
}


TRACE_SHORT = {
    "caida_2016": "caida2016",
    "caida_2018": "caida2018",
    "imc": "imc",
    "mawi": "mawi",
}


class ViTDecoder(nn.Module):
    def __init__(self, out_dim: int, in_chans: int = 3, img_size: int = 64):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1000),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1000, out_dim),
        )

    def forward(self, x):
        return self.head(self.vit(x))


class OneLayerMLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, in_chans: int, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_chans),
            nn.Conv2d(in_chans, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(192, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-50-window NeuFSD performance-eval runner.")
    parser.add_argument("--run-root", type=Path, default=Path("mainonly_runs_20260623"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", choices=["caida_2016", "caida_2018", "imc", "mawi"], required=True)
    parser.add_argument(
        "--variant",
        choices=["original", "origin_vit", "origin_mlp", "origin_cnn"],
        required=True,
    )
    parser.add_argument("--strategy", choices=["avg5", "window5", "last"], default="avg5")
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--start-minute", type=int, default=5)
    parser.add_argument("--end-minute", type=int, default=54)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=400)
    parser.add_argument("--last-replicas", type=int, default=5)
    parser.add_argument("--avg-replicas", type=int, default=5)
    parser.add_argument("--last-replica-stride", type=int, default=10000)
    parser.add_argument("--avg-replica-stride", type=int, default=10000)
    parser.add_argument("--prep-workers", type=int, default=32)
    parser.add_argument("--counter-threads", type=int, default=32)
    parser.add_argument("--gen-bin", type=Path, default=Path("run_tools/gen_counter_store"))
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--runtime-train-cache-dir", type=Path, default=None)
    parser.add_argument("--pretrained-root", type=Path, default=None)
    parser.add_argument("--force-cache-rebuild", action="store_true")
    parser.add_argument("--use-sample-counter-cache", action="store_true")
    parser.add_argument("--phi", type=int, default=1000)
    parser.add_argument("--input-mode", choices=["raw", "log1p", "log1p_total"], default="raw")
    parser.add_argument("--init-from", choices=["pretrained", "scratch"], default="pretrained")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def default_config_dir(root: Path, res: str, trace: str) -> Path:
    return root / "configs" / f"{res}_{TRACE_CONFIG[trace]}"


def default_sample_cache(run_root: Path, trace: str) -> Path:
    if trace == "caida_2016":
        return run_root / "sliding_fullstack_online" / "sample_cache_caida2016"
    if trace == "caida_2018":
        return run_root / "sliding_fullstack_online" / "sample_cache_caida2018"
    return run_root / "sliding_fullstack_online" / "fourdataset_s2000" / f"sample_cache_{trace}"


def default_train_cache(run_root: Path, trace: str, strategy: str) -> Path:
    short = TRACE_SHORT[trace]
    if trace in {"caida_2016", "caida_2018"}:
        return run_root / "sliding_fullstack_online" / "fair_s2000" / "train_cache" / f"{short}_{strategy}_rep5_s400"
    return (
        run_root
        / "sliding_fullstack_online"
        / "fourdataset_s2000"
        / "final_e5"
        / "train_cache"
        / f"{short}_{strategy}_rep5_s400"
    )


def configure_runtime(args: argparse.Namespace, root: Path) -> torch.device:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_imports(args.config_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"runner expects exactly one visible GPU, got {torch.cuda.device_count()}")
    os.environ["COUNTER_BACKEND"] = "memmap"
    os.environ["COUNTER_INPUT_MODE"] = args.input_mode
    os.environ["COUNTER_FEATURE_MODE"] = "origin" if args.variant.startswith("origin_") else "three"
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


def checkpoint_for(pretrained_root: Path, head: str) -> Path:
    ckpts = sorted((pretrained_root / HEADS[head]["model_dir"]).glob("best_model_*.pth"))
    if not ckpts:
        raise FileNotFoundError(pretrained_root / HEADS[head]["model_dir"])
    return ckpts[0]


def adapt_vit_state(state: dict[str, torch.Tensor], in_chans: int) -> dict[str, torch.Tensor]:
    if in_chans == 3:
        return state
    adapted = dict(state)
    key = "vit.patch_embed.proj.weight"
    if key in adapted and adapted[key].ndim == 4 and adapted[key].shape[1] == 3:
        adapted[key] = adapted[key].mean(dim=1, keepdim=True)
    return adapted


def make_model(args: argparse.Namespace, out_dim: int, ckpt: Path | None, device: torch.device) -> nn.Module:
    img_size = 64 if args.res == "64_64" else 128
    if args.variant == "original":
        model = ViTDecoder(out_dim=out_dim, in_chans=3, img_size=img_size)
        in_chans = 3
    elif args.variant == "origin_vit":
        model = ViTDecoder(out_dim=out_dim, in_chans=1, img_size=img_size)
        in_chans = 1
    elif args.variant == "origin_mlp":
        model = OneLayerMLP(img_size * img_size, out_dim)
        in_chans = 1
    elif args.variant == "origin_cnn":
        model = SmallCNN(1, out_dim)
        in_chans = 1
    else:
        raise ValueError(args.variant)
    model = model.to(device)
    if ckpt is not None and args.variant in {"original", "origin_vit"}:
        state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(adapt_vit_state(state, in_chans), strict=False)
        if unexpected:
            print(f"unexpected checkpoint keys for {args.variant}: {unexpected}", flush=True)
        if missing:
            print(f"missing checkpoint keys for {args.variant}: {missing}", flush=True)
    return model


def load_true(final_dir: Path, minute: int) -> tuple[np.ndarray, np.ndarray]:
    name = dataset_id(minute)
    one = np.load(final_dir / "tr_ts" / "1_10_real" / f"{name}.npy")
    ten = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{name}.npy")
    real = np.vstack((one.reshape(-1, 2), ten.reshape(-1, 2)))
    return real[:, 0].astype(int), real[:, 1].astype(float)


def load_heavy(heavy_root: Path, minute: int, phi: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = heavy_root / str(minute) / "heavy_0.csv"
    freq: dict[int, int] = {}
    mass = 0.0
    if not path.exists():
        return np.empty(0), np.empty(0), 0.0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            count = int(row["count"])
            if count <= phi:
                continue
            freq[count] = freq.get(count, 0) + 1
            mass += count
    if not freq:
        return np.empty(0), np.empty(0), 0.0
    values = np.array(sorted(freq), dtype=float)
    counts = np.array([freq[int(v)] for v in values], dtype=float)
    return values, counts, mass


def neural_counts(pred_1: np.ndarray, pred_2: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    preds = np.maximum(np.concatenate((pred_1, pred_2), axis=1), 0)
    out = np.empty((preds.shape[0], sizes.size), dtype=float)
    x = sizes.astype(float)
    for i in range(preds.shape[0]):
        out[i] = np.maximum(interp_extrap(ANCHOR_SIZES, preds[i], x), 0)
    return out


def metric_pair(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred = np.maximum(pred, 0)
    denom = (pred + true) / 2.0
    valid = denom > 0
    mrd = float(np.mean(np.abs(pred[valid] - true[valid]) / denom[valid]))
    wmrd = float(np.mean(np.abs(pred - true)) / np.mean(denom))
    return mrd, wmrd


def evaluate(
    final_dir: Path,
    heavy_root: Path,
    minute: int,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    phi: int,
    array_only: bool = False,
) -> dict[str, float]:
    sizes, true = load_true(final_dir, minute)
    neural = neural_counts(pred_1, pred_2, sizes)
    if array_only:
        base = np.zeros_like(true, dtype=float)
        light_mask = np.ones_like(sizes, dtype=bool)
        heavy_mass = 0.0
    else:
        light_mask = sizes <= phi
        heavy_values, heavy_freqs, heavy_mass = load_heavy(heavy_root, minute, phi)
        base = np.zeros_like(true, dtype=float)
        if np.any(~light_mask):
            base[~light_mask] = interp_extrap(heavy_values, heavy_freqs, sizes[~light_mask].astype(float))

    mrd_values: list[float] = []
    wmrd_values: list[float] = []
    for seed_idx in range(pred_1.shape[0]):
        pred = base.copy()
        pred[light_mask] = neural[seed_idx, light_mask]
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metric_pair(pred, true)
        mrd_values.append(mrd)
        wmrd_values.append(wmrd)
    return {
        "mrd": float(np.mean(mrd_values)),
        "wmrd": float(np.mean(wmrd_values)),
        "heavy_mass": float(heavy_mass),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.run_root = args.run_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_dir = args.out_root / "runs" / f"{TRACE_SHORT[args.trace]}_{args.variant}"
    args.config_dir = default_config_dir(root, args.res, args.trace)
    args.sample_cache_dir = args.sample_cache_dir or default_sample_cache(args.run_root, args.trace)
    args.runtime_train_cache_dir = args.runtime_train_cache_dir or default_train_cache(
        args.run_root, args.trace, args.strategy
    )
    args.pretrained_root = args.pretrained_root or (args.run_root / "pretrain_fourdatasets" / "pretrained" / args.res)
    if args.variant != "original" and args.variant not in {"origin_vit"}:
        args.init_from = "scratch"

    done = args.out_dir / "DONE"
    if done.exists() and not args.force:
        print(f"skip complete run: {args.out_dir}", flush=True)
        return

    device = configure_runtime(args, root)
    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    sample_minutes = list(range(args.start_minute - args.history_size, args.end_minute))
    prepare_cache_range(args, raw_dir, sample_minutes)
    counter_len = counter_len_for_res(args.res)
    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    heavy_root = final_dir / "EL"

    if args.init_from == "pretrained":
        ckpts = {head: checkpoint_for(args.pretrained_root, head) for head in HEADS}
    else:
        ckpts = {head: None for head in HEADS}

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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "args.json").open("w") as f:
        json.dump(
            vars(args)
            | {
                "config_dir": str(args.config_dir),
                "feature_mode": os.environ["COUNTER_FEATURE_MODE"],
                "visible_gpu": torch.cuda.get_device_name(0),
                "ckpts": {k: str(v) if v else None for k, v in ckpts.items()},
            },
            f,
            indent=2,
            default=str,
        )

    print(
        f"trace={args.trace} variant={args.variant} strategy={args.strategy} "
        f"minutes={args.start_minute}-{args.end_minute} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for minute in range(args.start_minute, args.end_minute + 1):
        history = list(range(minute - args.history_size, minute))
        train_root, train_minutes, root_info = build_training_root(args, minute, history)
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
        np.savez_compressed(pred_dir / f"minute_{minute:04d}.npz", pred_1=pred_1, pred_2=pred_2)

        metric_t0 = time.perf_counter()
        metric_row = evaluate(final_dir, heavy_root, minute, pred_1, pred_2, args.phi, array_only=False)
        metric_sec = time.perf_counter() - metric_t0
        row = {
            "trace": args.trace,
            "variant": args.variant,
            "minute": minute,
            "phi": args.phi,
            "history_start": history[0],
            "history_end": history[-1],
            "train_minutes": " ".join(str(v) for v in train_minutes),
            "train_count": len(train_minutes),
            "counter_build_sec": counter_sec,
            "sft_total_sec": sft_total_sec,
            "snapshot_load_sec": snapshot_load_sec,
            "infer_sec": infer_sec,
            "metric_eval_sec": metric_sec,
            "decode_sec": infer_sec + metric_sec,
            **root_info,
            **metric_row,
            **sft_info,
        }
        rows.append(row)
        write_csv(args.out_dir / "window_metrics.csv", rows)
        print(
            f"minute={minute} train={sft_total_sec:.3f}s infer={infer_sec:.3f}s "
            f"wmrd={metric_row['wmrd']:.4f} mrd={metric_row['mrd']:.4f}",
            flush=True,
        )

        if args.runtime_train_cache_dir is None:
            shutil.rmtree(train_root, ignore_errors=True)

    if rows:
        summary = {
            "trace": args.trace,
            "variant": args.variant,
            "n_windows": len(rows),
            "wmrd_mean": float(np.mean([float(r["wmrd"]) for r in rows])),
            "mrd_mean": float(np.mean([float(r["mrd"]) for r in rows])),
            "sft_total_sec_mean": float(np.mean([float(r["sft_total_sec"]) for r in rows])),
            "decode_sec_mean": float(np.mean([float(r["decode_sec"]) for r in rows])),
        }
        write_csv(args.out_dir / "summary.csv", [summary])
    done.write_text("done\n")


if __name__ == "__main__":
    main()
