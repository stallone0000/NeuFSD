#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset

from packet_counts import count_raw_packets, default_packet_format, validate_packet_count, window_packet_count


HEADS = {
    "1_10": {
        "label_dir": "1_10_chazhi",
        "model_dir": "ViT_1_10_results_1e-2",
        "out_dim": 10,
    },
    "10_1e4": {
        "label_dir": "10_1e4_chazhi",
        "model_dir": "ViT_10_1e4_results_1e-2",
        "out_dim": 1080,
    },
}
SPLICE_POINT = 100
LIGHT_SIZES = np.arange(1, SPLICE_POINT + 1, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict one-GPU realtime online simulation: infer each 1-minute window, "
            "then optionally fine-tune on a causal sliding history."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--res", default="64_64")
    parser.add_argument("--trace", default="caida_2016")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--gpu-deadline-sec", type=float, default=60.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--start-minute", type=int, default=0)
    parser.add_argument("--end-minute", type=int, default=303)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--packet-size", type=int, default=0, help="Raw packet size in bytes. 0 infers from trace.")
    parser.add_argument("--key-offset", type=int, default=-1, help="Flow key byte offset. -1 infers from trace.")
    parser.add_argument("--key-length", type=int, default=0, help="Flow key length in bytes. 0 infers from trace.")
    parser.add_argument("--mode", choices=["full", "head-only"], default="full")
    parser.add_argument(
        "--train-source",
        choices=["raw-sample", "fine-cache"],
        default="raw-sample",
        help=(
            "raw-sample trains original counters against labels built from sampled raw packets; "
            "fine-cache trains on cached fine_dataset_* counters/labels, matching the earlier online experiments."
        ),
    )
    parser.add_argument(
        "--fine-root",
        type=Path,
        default=None,
        help="Root containing cached tr_ts_finetuned_continue fine_dataset_* inputs and labels.",
    )
    parser.add_argument(
        "--policy",
        choices=["continuous", "fixed-train-once"],
        default="continuous",
        help="continuous trains after every measured window; fixed-train-once trains once on a fixed prefix.",
    )
    parser.add_argument("--fixed-train-start", type=int, default=0)
    parser.add_argument("--fixed-train-end", type=int, default=4)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--precompute-labels", action="store_true")
    parser.add_argument("--max-label-build", type=int, default=0)
    parser.add_argument("--adaptive-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=1.02)
    parser.add_argument("--gate-cap", type=float, default=1.20)
    parser.add_argument("--gate-down-threshold", type=float, default=0.0)
    parser.add_argument("--gate-floor", type=float, default=0.50)
    parser.add_argument("--sample-shape-weight", type=float, default=0.0)
    parser.add_argument("--sample-shape-max-freq", type=int, default=100)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
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
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def setup_imports(config_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    local_deps = root / "local_deps"
    for path in [root / "src", local_deps, config_dir]:
        if path.exists():
            sys.path.insert(0, str(path))


def dataset_id(minute: int) -> str:
    return f"dataset_{minute:04d}"


def fine_id(minute: int) -> str:
    return f"fine_dataset_{minute:04d}"


def numeric_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits or 0)


def resolve_packet_format(args: argparse.Namespace) -> None:
    if args.packet_size == 0 or args.key_offset < 0 or args.key_length == 0:
        packet_size, key_offset, key_length = default_packet_format(args.trace)
        args.packet_size = args.packet_size or packet_size
        args.key_offset = key_offset if args.key_offset < 0 else args.key_offset
        args.key_length = args.key_length or key_length
    if args.packet_size <= 0 or args.key_offset < 0 or args.key_length <= 0:
        raise ValueError("Invalid packet parser settings.")
    if args.key_offset + args.key_length > args.packet_size:
        raise ValueError(
            f"Flow key [{args.key_offset}, {args.key_offset + args.key_length}) exceeds packet size {args.packet_size}."
        )


def parse_packet_keys(path: Path, packet_size: int, key_offset: int, key_length: int) -> dict[bytes, int]:
    count_raw_packets(path, packet_size)
    buffer_size = 4096
    freq: dict[bytes, int] = defaultdict(int)
    packet_offset = 0
    buffer = bytearray(buffer_size + packet_size - 1)
    with path.open("rb") as f:
        while True:
            chunk = f.read(buffer_size - packet_offset)
            if not chunk:
                break
            buffer[packet_offset : packet_offset + len(chunk)] = chunk
            total_bytes = packet_offset + len(chunk)
            packets = total_bytes // packet_size
            for i in range(packets):
                start = i * packet_size
                key = bytes(buffer[start + key_offset : start + key_offset + key_length])
                freq[key] += 1
            packet_offset = total_bytes % packet_size
            if packet_offset > 0:
                buffer[:packet_offset] = buffer[total_bytes - packet_offset : total_bytes]
    return freq


def stable_sample(key: bytes, minute: int, sample_rate: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        key + minute.to_bytes(4, "little", signed=False) + seed.to_bytes(4, "little", signed=False),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "little") / float(1 << 64)
    return value < sample_rate


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


def sample_label_paths(label_root: Path, minute: int) -> tuple[Path, Path]:
    return (
        label_root / "1_10_chazhi" / f"{dataset_id(minute)}.npy",
        label_root / "10_1e4_chazhi" / f"{dataset_id(minute)}.npy",
    )


def build_sample_label(
    raw_path: Path,
    label_root: Path,
    minute: int,
    sample_rate: float,
    seed: int,
    packet_size: int,
    key_offset: int,
    key_length: int,
) -> dict[str, float]:
    t0 = time.perf_counter()
    label_root.joinpath("1_10_chazhi").mkdir(parents=True, exist_ok=True)
    label_root.joinpath("10_1e4_chazhi").mkdir(parents=True, exist_ok=True)
    flow_counts = parse_packet_keys(raw_path, packet_size, key_offset, key_length)
    sampled_fsd: dict[int, float] = defaultdict(float)
    multiplier = 1.0 / sample_rate
    sampled_flows = 0
    sampled_packets = 0
    for key, count in flow_counts.items():
        if stable_sample(key, minute, sample_rate, seed):
            sampled_fsd[int(count)] += multiplier
            sampled_flows += 1
            sampled_packets += int(count)
    sorted_fsd = sorted(sampled_fsd.items())
    one = interp_dense(sorted_fsd, np.arange(1, 11, dtype=float))
    ten = interp_dense(
        sorted_fsd,
        np.concatenate((np.arange(11, 1001, dtype=float), np.arange(1001, 10001, 100, dtype=float))),
    )
    one_path, ten_path = sample_label_paths(label_root, minute)
    np.save(one_path, one)
    np.save(ten_path, ten)
    return {
        "sample_label_build_sec": time.perf_counter() - t0,
        "sampled_flows": float(sampled_flows),
        "sampled_packets": float(sampled_packets),
        "full_flows": float(len(flow_counts)),
    }


def ensure_sample_label(
    raw_dir: Path,
    label_root: Path,
    minute: int,
    sample_rate: float,
    seed: int,
    packet_size: int,
    key_offset: int,
    key_length: int,
) -> dict[str, float]:
    one_path, ten_path = sample_label_paths(label_root, minute)
    if one_path.exists() and ten_path.exists():
        return {"sample_label_build_sec": 0.0, "sampled_flows": np.nan, "sampled_packets": np.nan, "full_flows": np.nan}
    raw_path = raw_dir / f"{dataset_id(minute)}.dat"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    return build_sample_label(raw_path, label_root, minute, sample_rate, seed, packet_size, key_offset, key_length)


def load_label(label_root: Path, minute: int, head: str) -> torch.Tensor:
    if head == "1_10":
        arr = np.load(label_root / "1_10_chazhi" / f"{dataset_id(minute)}.npy").reshape(1, -1)
    else:
        arr = np.load(label_root / "10_1e4_chazhi" / f"{dataset_id(minute)}.npy").reshape(1, -1)
    return torch.from_numpy(arr.astype(np.float32))


@lru_cache(maxsize=384)
def cached_counter(counter_root_text: str, minute: int) -> torch.Tensor:
    from mrac_data import read_counter_dataset

    data, _ = read_counter_dataset(Path(counter_root_text), dataset_id(minute))
    return data


def read_raw_sample_train_window(
    counter_root: Path,
    label_root: Path,
    minute: int,
    head: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = cached_counter(str(counter_root), minute)
    label = load_label(label_root, minute, head)
    labels = label.repeat(data.shape[0], 1)
    return data, labels


def read_fine_cache_train_window(fine_root: Path, minute: int, head: str) -> tuple[torch.Tensor, torch.Tensor]:
    from mrac_data import read_labeled_counter_dataset

    if (fine_root / "input_store" / "index.json").exists():
        return read_labeled_counter_dataset(fine_root, fine_id(minute), HEADS[head]["label_dir"])

    # Fine-cache roots are directory-based in the reproduced runs, while the
    # measured tr_ts roots often use memmap stores. Keep this local so measured
    # window reads can still use COUNTER_BACKEND=memmap.
    previous_backend = os.environ.get("COUNTER_BACKEND")
    os.environ["COUNTER_BACKEND"] = "file"
    try:
        return read_labeled_counter_dataset(fine_root, fine_id(minute), HEADS[head]["label_dir"])
    finally:
        if previous_backend is None:
            os.environ.pop("COUNTER_BACKEND", None)
        else:
            os.environ["COUNTER_BACKEND"] = previous_backend


def split_holdout(x: torch.Tensor, y: torch.Tensor, frac: float, seed: int) -> tuple[torch.Tensor, ...]:
    n = x.shape[0]
    holdout_n = max(1, int(round(n * frac)))
    holdout_n = min(holdout_n, n - 1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    holdout = order[:holdout_n]
    train = order[holdout_n:]
    return x[train], y[train], x[holdout], y[holdout]


def make_model(out_dim: int, ckpt: Path | None, device: torch.device):
    from model import CustomViT

    model = CustomViT(out_dim=out_dim).to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def set_trainable(model: torch.nn.Module, mode: str) -> list[torch.nn.Parameter]:
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return list(model.parameters())
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    return list(model.head.parameters())


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_head(model, data: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            preds.append(outputs.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_head(
    model,
    counter_root: Path,
    label_root: Path,
    fine_root: Path,
    minutes: list[int],
    head: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    load_t0 = time.perf_counter()
    xs, ys = [], []
    for minute in minutes:
        if args.train_source == "fine-cache":
            x, y = read_fine_cache_train_window(fine_root, minute, head)
        else:
            x, y = read_raw_sample_train_window(counter_root, label_root, minute, head)
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    train_x, train_y, holdout_x, holdout_y = split_holdout(
        x,
        y,
        args.holdout_frac,
        args.torch_seed + minutes[-1] * 17 + (0 if head == "1_10" else 100000),
    )
    data_load_sec = time.perf_counter() - load_t0
    params = set_trainable(model, args.mode)
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        TensorDataset(holdout_x, holdout_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    train_t0 = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        total = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
        final_loss = total / max(1, len(loader))
    cuda_sync()
    train_sec = time.perf_counter() - train_t0
    val_t0 = time.perf_counter()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, targets in holdout_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            val_losses.append(float(criterion(model(inputs), targets).detach().cpu()))
    cuda_sync()
    return {
        f"{head}_data_load_sec": data_load_sec,
        f"{head}_train_sec": train_sec,
        f"{head}_holdout_sec": time.perf_counter() - val_t0,
        f"{head}_train_loss": float(final_loss),
        f"{head}_holdout_loss": float(np.mean(val_losses)) if val_losses else np.nan,
        f"{head}_train_samples": float(train_x.shape[0]),
        f"{head}_holdout_samples": float(holdout_x.shape[0]),
    }


def load_true(final_dir: Path, minute: int) -> tuple[np.ndarray, np.ndarray]:
    name = dataset_id(minute)
    one = np.load(final_dir / "tr_ts" / "1_10_real" / f"{name}.npy")
    ten = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{name}.npy")
    real = np.vstack((one.reshape(-1, 2), ten.reshape(-1, 2)))
    return real[:, 0].astype(int), real[:, 1].astype(float)


def load_heavy(final_dir: Path, minute: int, heavy_count_min: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    path = final_dir / "EL" / str(minute) / "heavy_0.csv"
    freq: dict[int, int] = {}
    mass = 0.0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            count = int(row["count"])
            if count < heavy_count_min:
                continue
            freq[count] = freq.get(count, 0) + 1
            mass += count
    if not freq:
        return np.empty(0), np.empty(0), 0.0
    values = np.array(sorted(freq), dtype=float)
    counts = np.array([freq[int(v)] for v in values], dtype=float)
    return values, counts, mass


def interp_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=float)
    if xp.size == 0:
        return np.zeros(x.size, dtype=float)
    if xp.size == 1:
        return np.full(x.size, fp[0], dtype=float)
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    if np.any(lo):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[lo] = fp[0] + (x[lo] - xp[0]) * slope
    hi = x > xp[-1]
    if np.any(hi):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * slope
    return y


def small_predictions(pred_1: np.ndarray, pred_2: np.ndarray, small_sizes: np.ndarray) -> np.ndarray:
    out = np.empty((pred_1.shape[0], small_sizes.size), dtype=float)
    le10 = small_sizes <= 10
    if np.any(le10):
        out[:, le10] = pred_1[:, small_sizes[le10] - 1]
    if np.any(~le10):
        out[:, ~le10] = pred_2[:, small_sizes[~le10] - 11]
    return out


def light_mass_predictions(pred_1: np.ndarray, pred_2: np.ndarray) -> np.ndarray:
    light_preds = np.concatenate((pred_1[:, :10], pred_2[:, :90]), axis=1)
    return np.maximum(light_preds, 0).dot(LIGHT_SIZES)


def load_sample_shape(args: argparse.Namespace, minute: int, small_sizes: np.ndarray) -> tuple[np.ndarray | None, bool]:
    weight = float(getattr(args, "sample_shape_weight", 0.0))
    cache_dir = getattr(args, "sample_cache_dir", None)
    if weight <= 0 or cache_dir is None:
        return None, False
    path = Path(cache_dir) / "sampled_fsd" / f"{fine_id(minute)}.npz"
    if not path.exists():
        return None, False
    z = np.load(path)
    values = {int(freq): float(count) for freq, count in zip(z["freq"], z["count"])}
    sample = np.array([values.get(int(size), 0.0) for size in small_sizes], dtype=float)
    return sample, True


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred = np.maximum(pred, 0)
    denom = (pred + true) / 2.0
    valid = denom > 0
    return (
        float(np.mean(np.abs(pred[valid] - true[valid]) / denom[valid])),
        float(np.mean(np.abs(pred - true)) / np.mean(denom)),
    )


def final_metrics(
    final_dir: Path,
    minute: int,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    args: argparse.Namespace,
    state: dict[str, list[float]],
    *,
    packet_count: int,
) -> dict[str, float]:
    packet_count = validate_packet_count(packet_count)
    flow_sizes, true_counts = load_true(final_dir, minute)
    small_mask = flow_sizes <= SPLICE_POINT
    large_mask = ~small_mask
    small_sizes = flow_sizes[small_mask]
    sample_shape, sample_shape_available = load_sample_shape(args, minute, small_sizes)
    heavy_values, heavy_freqs, heavy_mass = load_heavy(final_dir, minute)
    base_large = np.zeros(flow_sizes.shape, dtype=float)
    base_large[large_mask] = interp_extrap(heavy_values, heavy_freqs, flow_sizes[large_mask].astype(float))
    small_pred = np.maximum(small_predictions(pred_1, pred_2, small_sizes), 0)
    residual = max(packet_count - heavy_mass, 0.0)
    raw_scale = np.divide(
        residual,
        light_mass_predictions(pred_1, pred_2),
        out=np.ones(pred_1.shape[0], dtype=float),
        where=light_mass_predictions(pred_1, pred_2) > 0,
    )
    applied = np.where(raw_scale > args.gate_threshold, np.minimum(raw_scale, args.gate_cap), 1.0)
    gate_down_threshold = float(getattr(args, "gate_down_threshold", 0.0))
    gate_floor = float(getattr(args, "gate_floor", 0.50))
    twoway_applied = np.where(
        raw_scale < gate_down_threshold,
        np.maximum(raw_scale, gate_floor),
        applied,
    )
    heavy_frac = heavy_mass / packet_count if packet_count > 0 else 0.0
    light_frac = residual / packet_count if packet_count > 0 else 0.0
    state.setdefault("heavy_frac_history", []).append(heavy_frac)
    state.setdefault("light_frac_history", []).append(light_frac)

    use_adaptive = False
    if args.adaptive_gate and np.mean(applied > 1.0) > 0:
        past_heavy = state["heavy_frac_history"][:-1]
        past_light = state["light_frac_history"][:-1]
        normal_heavy = len(past_heavy) >= 3 and float(np.quantile(past_heavy, 0.75)) >= 0.52
        light_spike = False
        if len(past_light) >= 3:
            base = float(np.quantile(past_light[-30:], 0.65))
            light_spike = light_frac >= base * 1.015 and light_frac >= base + 0.015
        use_adaptive = normal_heavy or light_spike

    raw_mrd, raw_wmrd = [], []
    gate_mrd, gate_wmrd = [], []
    twoway_mrd, twoway_wmrd = [], []
    sample_shape_mrd, sample_shape_wmrd = [], []
    adaptive_mrd, adaptive_wmrd = [], []
    for seed_idx in range(pred_1.shape[0]):
        pred = base_large.copy()
        pred[small_mask] = small_pred[seed_idx]
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metrics(pred, true_counts)
        raw_mrd.append(mrd)
        raw_wmrd.append(wmrd)
        gated = pred.copy()
        if applied[seed_idx] > 1.0:
            gated[small_mask] = np.around(gated[small_mask] * applied[seed_idx], 0)
            gated[gated < 0] = 0
        mrd, wmrd = metrics(gated, true_counts)
        gate_mrd.append(mrd)
        gate_wmrd.append(wmrd)
        twoway = pred.copy()
        if twoway_applied[seed_idx] != 1.0:
            twoway[small_mask] = np.around(twoway[small_mask] * twoway_applied[seed_idx], 0)
            twoway[twoway < 0] = 0
        mrd, wmrd = metrics(twoway, true_counts)
        twoway_mrd.append(mrd)
        twoway_wmrd.append(wmrd)
        sample_blend = twoway.copy()
        sample_weight = float(getattr(args, "sample_shape_weight", 0.0))
        sample_max_freq = int(getattr(args, "sample_shape_max_freq", SPLICE_POINT))
        if sample_shape_available and sample_weight > 0:
            shape_mask = small_sizes <= sample_max_freq
            if np.any(shape_mask):
                sample_blend[small_mask] = sample_blend[small_mask].copy()
                blended_small = sample_blend[small_mask]
                blended_small[shape_mask] = np.around(
                    (1.0 - sample_weight) * blended_small[shape_mask] + sample_weight * sample_shape[shape_mask],
                    0,
                )
                sample_blend[small_mask] = blended_small
                sample_blend[sample_blend < 0] = 0
        mrd, wmrd = metrics(sample_blend, true_counts)
        sample_shape_mrd.append(mrd)
        sample_shape_wmrd.append(wmrd)
        chosen = gated if use_adaptive else pred
        mrd, wmrd = metrics(chosen, true_counts)
        adaptive_mrd.append(mrd)
        adaptive_wmrd.append(wmrd)
    return {
        "mrd": float(np.mean(raw_mrd)),
        "wmrd": float(np.mean(raw_wmrd)),
        "gate_mrd": float(np.mean(gate_mrd)),
        "gate_wmrd": float(np.mean(gate_wmrd)),
        "twoway_gate_mrd": float(np.mean(twoway_mrd)),
        "twoway_gate_wmrd": float(np.mean(twoway_wmrd)),
        "sample_shape_mrd": float(np.mean(sample_shape_mrd)),
        "sample_shape_wmrd": float(np.mean(sample_shape_wmrd)),
        "adaptive_mrd": float(np.mean(adaptive_mrd)),
        "adaptive_wmrd": float(np.mean(adaptive_wmrd)),
        "packet_count": packet_count,
        "heavy_mass": float(heavy_mass),
        "residual_light_mass": float(residual),
        "heavy_frac": float(heavy_frac),
        "light_frac": float(light_frac),
        "raw_scale_mean": float(np.mean(raw_scale)),
        "gated_seed_fraction": float(np.mean(applied > 1.0)),
        "downscaled_seed_fraction": float(np.mean(twoway_applied < 1.0)),
        "twoway_scale_mean": float(np.mean(twoway_applied)),
        "sample_shape_available": bool(sample_shape_available),
        "adaptive_use_gate": bool(use_adaptive),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], out_dir: Path, args: argparse.Namespace) -> None:
    summary: dict[str, object] = {
        "res": args.res,
        "trace": args.trace,
        "policy": args.policy,
        "gpu_deadline_sec": args.gpu_deadline_sec,
        "window_size": args.window_size,
        "fixed_train_start": args.fixed_train_start,
        "fixed_train_end": args.fixed_train_end,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mode": args.mode,
        "train_source": args.train_source,
        "fine_root": str(args.fine_root),
        "packet_size": args.packet_size,
        "key_offset": args.key_offset,
        "key_length": args.key_length,
        "n_windows": len(rows),
        "deadline_miss": int(sum(bool(r["deadline_miss"]) for r in rows)),
    }
    for key in [
        "infer_sec",
        "sft_total_sec",
        "bootstrap_sft_sec",
        "bootstrap_deadline_budget_sec",
        "gpu_control_total_sec",
        "sample_label_build_sec",
        "model_lag_windows",
        "mrd",
        "wmrd",
        "gate_mrd",
        "gate_wmrd",
        "adaptive_mrd",
        "adaptive_wmrd",
    ]:
        values = np.array([float(r[key]) for r in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_p50"] = float(np.nanpercentile(values, 50))
        summary[f"{key}_p95"] = float(np.nanpercentile(values, 95))
        summary[f"{key}_max"] = float(np.nanmax(values))
    write_csv(out_dir / "summary.csv", [summary])

    minutes = np.array([int(r["minute"]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True)
    axes[0].plot(minutes, [float(r["gpu_control_total_sec"]) for r in rows], label="GPU control total")
    axes[0].plot(minutes, [float(r["infer_sec"]) for r in rows], label="Inference")
    axes[0].plot(minutes, [float(r["sft_total_sec"]) for r in rows], label="SFT")
    axes[0].axhline(args.gpu_deadline_sec, color="red", linestyle="--", linewidth=1.0, label="60s deadline")
    axes[0].set_ylabel("Seconds")
    axes[0].legend(loc="upper left", ncols=4)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(minutes, [float(r["wmrd"]) for r in rows], label="raw")
    axes[1].plot(minutes, [float(r["adaptive_wmrd"]) for r in rows], label="adaptive gate")
    axes[1].set_ylabel("WMRD")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(minutes, [float(r["model_lag_windows"]) for r in rows], label="model lag")
    axes[2].set_ylabel("Windows")
    axes[2].set_xlabel("Time (minute)")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "realtime_pipeline_timeseries.png", dpi=180)
    fig.savefig(out_dir / "realtime_pipeline_timeseries.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolve_packet_format(args)
    setup_imports(args.config_dir)
    if args.policy == "fixed-train-once" and args.fixed_train_end < args.fixed_train_start:
        raise ValueError("--fixed-train-end must be >= --fixed-train-start")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Strict one-GPU simulation requires exactly one visible GPU; got {torch.cuda.device_count()}. "
            "Run with CUDA_VISIBLE_DEVICES=<one gpu>."
        )
    device = torch.device("cuda:0")
    if bool(int(os.environ.get("ENABLE_TF32", "1"))):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    random.seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    torch.manual_seed(args.torch_seed)

    counter_root = args.run_root / "counter_store" / f"{args.res}_{args.trace}" / "tr_ts"
    raw_dir = args.data_full_root / args.trace / "caida_1min_split"
    label_root = args.out_dir / "sample_labels"
    final_dir = args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_final"
    fine_root = args.fine_root or (args.run_root / "run_full_matrix" / f"{args.res}_{args.trace}_exp" / "tr_ts_finetuned_continue")
    args.fine_root = fine_root

    with (args.out_dir / "args.json").open("w") as f:
        json.dump(vars(args) | {"visible_gpu_name": torch.cuda.get_device_name(0)}, f, indent=2, default=str)

    pretrained_root = args.pretrained_root or (args.run_root / "pretrained" / args.res)
    if args.init_from == "pretrained":
        ckpts = {
            head: sorted((pretrained_root / spec["model_dir"]).glob("best_model_*.pth"))[0]
            for head, spec in HEADS.items()
        }
    else:
        ckpts = {head: None for head in HEADS}
    print(f"using one visible GPU: {torch.cuda.get_device_name(0)}")
    print(f"init_from={args.init_from} ckpts: {ckpts}")
    print(
        f"packet parser: packet_size={args.packet_size} key_offset={args.key_offset} "
        f"key_length={args.key_length}"
    )
    print(f"train source: {args.train_source}")
    if args.train_source == "fine-cache":
        print(f"fine cache root: {fine_root}")
        if not fine_root.exists():
            raise FileNotFoundError(fine_root)

    if args.precompute_labels:
        if args.train_source != "raw-sample":
            raise ValueError("--precompute-labels only applies to --train-source raw-sample")
        count = 0
        for minute in range(args.start_minute, args.end_minute + 1):
            info = ensure_sample_label(
                raw_dir,
                label_root,
                minute,
                args.sample_rate,
                args.sample_seed,
                args.packet_size,
                args.key_offset,
                args.key_length,
            )
            if info["sample_label_build_sec"]:
                print(f"built sample label minute={minute} sec={info['sample_label_build_sec']:.3f}", flush=True)
            count += 1
            if args.max_label_build and count >= args.max_label_build:
                break

    models = {
        head: make_model(HEADS[head]["out_dim"], ckpts[head], device)
        for head in HEADS
    }

    bootstrap_sft_sec = 0.0
    bootstrap_deadline_budget_sec = 0.0
    bootstrap_info: dict[str, float] = {}
    active_train_end_minute = -1
    if args.policy == "fixed-train-once":
        train_minutes = list(range(args.fixed_train_start, args.fixed_train_end + 1))
        bootstrap_deadline_budget_sec = len(train_minutes) * args.gpu_deadline_sec
        print(f"fixed policy: bootstrap SFT on minutes {train_minutes}", flush=True)
        if args.train_source == "raw-sample":
            for train_minute in train_minutes:
                ensure_sample_label(
                    raw_dir,
                    label_root,
                    train_minute,
                    args.sample_rate,
                    args.sample_seed,
                    args.packet_size,
                    args.key_offset,
                    args.key_length,
                )
        sft_t0 = time.perf_counter()
        for head in ["1_10", "10_1e4"]:
            bootstrap_info.update(train_head(models[head], counter_root, label_root, fine_root, train_minutes, head, args, device))
        bootstrap_sft_sec = time.perf_counter() - sft_t0
        active_train_end_minute = args.fixed_train_end
        print(
            f"fixed policy bootstrap_sft={bootstrap_sft_sec:.3f}s "
            f"budget={bootstrap_deadline_budget_sec:.3f}s",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    gate_state: dict[str, list[float]] = {}
    simulated_time_sec = 0.0
    for minute in range(args.start_minute, args.end_minute + 1):
        print(f"minute {minute}/{args.end_minute}", flush=True)
        window_start_time = minute * args.gpu_deadline_sec
        if simulated_time_sec < window_start_time:
            simulated_time_sec = window_start_time

        if args.train_source == "raw-sample" and (
            args.policy == "continuous" or args.fixed_train_start <= minute <= args.fixed_train_end
        ):
            label_info = ensure_sample_label(
                raw_dir,
                label_root,
                minute,
                args.sample_rate,
                args.sample_seed,
                args.packet_size,
                args.key_offset,
                args.key_length,
            )
        else:
            label_info = {
                "sample_label_build_sec": 0.0,
                "sampled_flows": np.nan,
                "sampled_packets": np.nan,
                "full_flows": np.nan,
            }
        data_t0 = time.perf_counter()
        test_x = cached_counter(str(counter_root), minute)
        snapshot_load_sec = time.perf_counter() - data_t0

        cuda_sync()
        infer_t0 = time.perf_counter()
        pred_1 = infer_head(models["1_10"], test_x, args.batch_size * 2, device)
        pred_2 = infer_head(models["10_1e4"], test_x, args.batch_size * 2, device)
        cuda_sync()
        infer_sec = time.perf_counter() - infer_t0
        metric_t0 = time.perf_counter()
        metric_row = final_metrics(
            final_dir, minute, pred_1, pred_2, args, gate_state,
            packet_count=window_packet_count(raw_dir, minute, args.trace, args.packet_size),
        )
        metric_sec = time.perf_counter() - metric_t0

        if args.policy == "continuous":
            history_start = max(args.start_minute, minute - args.window_size + 1)
            history_end = minute
            train_minutes = list(range(history_start, minute + 1))
            sft_info: dict[str, float] = {}
            sft_t0 = time.perf_counter()
            for head in ["1_10", "10_1e4"]:
                sft_info.update(train_head(models[head], counter_root, label_root, fine_root, train_minutes, head, args, device))
            sft_total_sec = time.perf_counter() - sft_t0
        else:
            history_start = args.fixed_train_start
            history_end = args.fixed_train_end
            train_minutes = list(range(history_start, history_end + 1))
            sft_info = dict(bootstrap_info)
            sft_total_sec = 0.0
        gpu_control_total = infer_sec + sft_total_sec
        deadline_miss = gpu_control_total > args.gpu_deadline_sec
        if args.policy == "continuous" and not deadline_miss:
            active_train_end_minute = minute
        simulated_time_sec += gpu_control_total
        model_lag = minute - active_train_end_minute

        row: dict[str, object] = {
            "minute": minute,
            "history_start": history_start,
            "history_end": history_end,
            "history_count": len(train_minutes),
            "sample_label_build_sec": label_info["sample_label_build_sec"],
            "sampled_flows": label_info["sampled_flows"],
            "sampled_packets": label_info["sampled_packets"],
            "full_flows": label_info["full_flows"],
            "snapshot_load_sec": snapshot_load_sec,
            "infer_sec": infer_sec,
            "metric_eval_sec": metric_sec,
            "sft_total_sec": sft_total_sec,
            "bootstrap_sft_sec": bootstrap_sft_sec,
            "bootstrap_deadline_budget_sec": bootstrap_deadline_budget_sec,
            "gpu_control_total_sec": gpu_control_total,
            "deadline_miss": bool(deadline_miss),
            "model_lag_windows": model_lag,
            **metric_row,
            **sft_info,
        }
        rows.append(row)
        write_csv(args.out_dir / "window_metrics.csv", rows)
        print(
            f"  infer={infer_sec:.3f}s sft={sft_total_sec:.3f}s total={gpu_control_total:.3f}s "
            f"wmrd={metric_row['wmrd']:.4f} adaptive={metric_row['adaptive_wmrd']:.4f} "
            f"miss={deadline_miss}",
            flush=True,
        )

    summarize(rows, args.out_dir, args)
    print(f"wrote realtime simulation to {args.out_dir}")


if __name__ == "__main__":
    main()
