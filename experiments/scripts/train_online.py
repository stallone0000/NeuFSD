#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from packet_counts import validate_packet_count, window_packet_count


HEAD_LABEL = {
    "1_10": ("1_10_chazhi", "ViT_1_10_results_1e-2"),
    "10_1e4": ("10_1e4_chazhi", "ViT_10_1e4_results_1e-2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online MRAC fine-tuning in independent or continuous mode.")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--head", choices=sorted(HEAD_LABEL), required=True)
    parser.add_argument("--mode", choices=["independent", "continuous"], required=True)
    parser.add_argument("--initial-ckpt", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("REPRO_EPOCHS", 20)))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", 5)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ONLINE_BATCH_SIZE", 64)))
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-train-blocks", type=int, default=int(os.environ.get("MAX_TRAIN_BLOCKS", 0)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=["test-val", "sample-holdout", "train-loss"],
        default=os.environ.get("ONLINE_SELECTION", "test-val"),
        help=(
            "Checkpoint selection policy. test-val preserves the original offline protocol; "
            "sample-holdout and train-loss avoid using test-window FSD labels for model selection."
        ),
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=float(os.environ.get("ONLINE_HOLDOUT_FRAC", 0.1)),
        help="Fraction of sampled training counter snapshots held out when --selection sample-holdout.",
    )
    parser.add_argument(
        "--final-active-loss",
        action="store_true",
        help="Train/evaluate only bins used by final heavy+counter splice: all 1_10 bins, and 11..100 for 10_1e4.",
    )
    parser.add_argument(
        "--label-light-mass-correction",
        choices=["none", "up-only"],
        default="none",
        help="Online-safe training-label correction: scale sampled 1..100 FSD labels using packet residual after heavy part.",
    )
    parser.add_argument("--label-light-mass-threshold", type=float, default=1.05)
    parser.add_argument("--label-light-mass-cap", type=float, default=1.25)
    parser.add_argument("--heavy-dir", type=Path, default=None, help="Directory containing EL/<minute>/heavy_0.csv.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing original dataset_*.dat windows; required for label light-mass correction.")
    parser.add_argument("--trace", default=None, help="Raw format name, required unless --packet-size is supplied.")
    parser.add_argument("--packet-size", type=int, default=0, help="Raw record size in bytes; 0 infers from --trace.")
    parser.add_argument("--packet-count", type=float, default=None, help="Optional expected packet count, checked against every corrected raw window.")
    return parser.parse_args()


def setup_imports(config_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    local_deps = root / "local_deps"
    for path in [root / "src", local_deps, config_dir]:
        if path.exists():
            sys.path.insert(0, str(path))


def task_groups(tasks: dict[str, bool]) -> list[tuple[list[str], list[str]]]:
    items = list(tasks.items())
    groups = []
    i = 0
    while i < len(items):
        train = []
        test = []
        while i < len(items) and items[i][1]:
            train.append(Path(items[i][0]).stem)
            i += 1
        while i < len(items) and not items[i][1]:
            test.append(Path(items[i][0]).stem)
            i += 1
        if train or test:
            groups.append((train, test))
    return groups


def run_evaluation(model, dataloader, device, metric_slice=None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_targets)
    metric_preds = preds[:, metric_slice] if metric_slice is not None else preds
    metric_trues = trues[:, metric_slice] if metric_slice is not None else trues
    mse = np.mean((metric_preds - metric_trues) ** 2)
    mae = np.mean(np.abs(metric_preds - metric_trues))
    r2 = r2_score(metric_trues.ravel(), metric_preds.ravel())
    metric_preds = np.maximum(metric_preds, 0)
    val_wmrd = np.mean(np.abs(metric_preds - metric_trues)) / np.mean((metric_preds + metric_trues) / 2) * 100
    return preds, trues, mse, mae, r2, val_wmrd


def split_sample_holdout(train_x: torch.Tensor, train_y: torch.Tensor, frac: float, seed: int) -> tuple[torch.Tensor, ...]:
    if not 0 < frac < 1:
        raise ValueError("--holdout-frac must be in (0, 1)")
    n = train_x.shape[0]
    holdout_n = max(1, int(round(n * frac)))
    if holdout_n >= n:
        holdout_n = n - 1
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    holdout_idx = order[:holdout_n]
    train_idx = order[holdout_n:]
    return train_x[train_idx], train_y[train_idx], train_x[holdout_idx], train_y[holdout_idx]


def original_dataset_id(dataset_id: str) -> str:
    if dataset_id.startswith("fine_"):
        return dataset_id[len("fine_") :]
    return dataset_id


def minute_from_dataset_id(dataset_id: str) -> int:
    return int(original_dataset_id(dataset_id).split("_")[-1])


def heavy_packet_mass(heavy_dir: Path | None, dataset_id: str) -> float:
    if heavy_dir is None:
        return 0.0
    path = heavy_dir / str(minute_from_dataset_id(dataset_id)) / "heavy_0.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            count = float(row["count"])
            if count > 99:
                total += count
    return total


def light_label_mass(root_dir: Path, dataset_id: str) -> float:
    one = np.load(root_dir / "1_10_chazhi" / f"{dataset_id}.npy").reshape(-1).astype(float)
    ten = np.load(root_dir / "10_1e4_chazhi" / f"{dataset_id}.npy").reshape(-1).astype(float)
    sizes = np.arange(1, 101, dtype=float)
    light = np.concatenate((one[:10], ten[:90]))
    return float(np.sum(sizes * np.maximum(light, 0)))


def maybe_correct_label(
    labels: torch.Tensor,
    root_dir: Path,
    dataset_id: str,
    head: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.label_light_mass_correction == "none":
        return labels
    if args.raw_dir is None:
        raise ValueError("--raw-dir is required for --label-light-mass-correction")
    packet_count = window_packet_count(args.raw_dir, minute_from_dataset_id(dataset_id), args.trace, args.packet_size)
    if args.packet_count is not None and validate_packet_count(args.packet_count) != packet_count:
        raise ValueError(f"--packet-count={args.packet_count} does not match {dataset_id} raw count {packet_count}")
    residual = max(packet_count - heavy_packet_mass(args.heavy_dir, dataset_id), 0.0)
    sample_mass = light_label_mass(root_dir, dataset_id)
    if sample_mass <= 0:
        return labels
    raw_scale = residual / sample_mass
    if args.label_light_mass_correction == "up-only" and raw_scale <= args.label_light_mass_threshold:
        return labels
    scale = min(raw_scale, args.label_light_mass_cap)
    labels = labels.clone()
    if head == "1_10":
        labels *= scale
    else:
        labels[:, :90] *= scale
    return labels


def main() -> None:
    args = parse_args()
    if args.label_light_mass_correction != "none":
        if args.raw_dir is None:
            raise ValueError("--raw-dir is required for --label-light-mass-correction")
        if args.packet_size == 0 and args.trace is None:
            raise ValueError("Pass --trace or --packet-size for --label-light-mass-correction")
    setup_imports(args.config_dir)
    from model import CustomViT
    from mrac_data import read_labeled_counter_dataset

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for online fine-tuning")
    device = torch.device("cuda:0")
    if bool(int(os.environ.get("ENABLE_TF32", "0"))):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    pin_memory = bool(int(os.environ.get("PIN_MEMORY", "1")))
    gpu_cache_data = bool(int(os.environ.get("GPU_CACHE_DATA", "0")))
    if gpu_cache_data:
        pin_memory = False
    label_dir, model_dir_name = HEAD_LABEL[args.head]
    active_slice = None
    if args.final_active_loss and args.head == "10_1e4":
        # Final evaluation splices neural estimates only for 11..100; 101+ is
        # supplied by the heavy part. The first 90 dims are 11..100.
        active_slice = slice(0, 90)
    out_root = args.out_root or (args.exp_dir / "0_finetuned_results")
    model_out_root = out_root / model_dir_name
    model_out_root.mkdir(parents=True, exist_ok=True)

    with (args.exp_dir / "train_test_name_key.json").open() as f:
        tasks = json.load(f)
    groups = task_groups(tasks)
    first_train = next(train[0] for train, _ in groups if train)
    _, dummy_label = read_labeled_counter_dataset(args.exp_dir / "tr_ts_finetuned_continue", first_train, label_dir)
    out_dim = dummy_label.shape[-1]

    def make_model():
        model = CustomViT(out_dim=out_dim).to(device)
        if args.initial_ckpt:
            model.load_state_dict(torch.load(args.initial_ckpt, map_location=device))
            print(f"loaded initial ckpt: {args.initial_ckpt}")
        return model

    model = None
    if args.mode == "continuous":
        model = make_model()

    for block_index, (train_names, test_names) in enumerate(groups):
        if not train_names:
            continue
        if args.max_train_blocks and block_index >= args.max_train_blocks:
            print(f"stop at MAX_TRAIN_BLOCKS={args.max_train_blocks}")
            break
        block_name = train_names[0]
        print("=" * 80)
        print(f"block={block_index} mode={args.mode} train={train_names} test={test_names}")
        if args.mode == "independent" or model is None:
            model = make_model()

        train_data, train_labels = [], []
        for dataset_id in train_names:
            train_root = args.exp_dir / "tr_ts_finetuned_continue"
            data, labels = read_labeled_counter_dataset(train_root, dataset_id, label_dir)
            labels = maybe_correct_label(labels, train_root, dataset_id, args.head, args)
            train_data.append(data)
            train_labels.append(labels)
        train_x = torch.cat(train_data, 0)
        train_y = torch.cat(train_labels, 0)
        selection_loaders = []
        if args.selection == "sample-holdout":
            train_x, train_y, holdout_x, holdout_y = split_sample_holdout(
                train_x,
                train_y,
                args.holdout_frac,
                args.seed + block_index,
            )
            if gpu_cache_data:
                train_x = train_x.to(device)
                train_y = train_y.to(device)
                holdout_x = holdout_x.to(device)
                holdout_y = holdout_y.to(device)
            selection_loaders.append(
                (
                    "sample_holdout",
                    DataLoader(
                        TensorDataset(holdout_x, holdout_y),
                        batch_size=args.batch_size * 2,
                        shuffle=False,
                        num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)),
                        pin_memory=pin_memory,
                    ),
                )
            )
        elif gpu_cache_data:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)),
            pin_memory=pin_memory,
        )

        def make_test_loaders():
            loaders = []
            eval_names = test_names or [train_names[-1]]
            for dataset_id in eval_names:
                data, labels = read_labeled_counter_dataset(args.exp_dir / "tr_ts", dataset_id, label_dir)
                if gpu_cache_data:
                    data = data.to(device)
                    labels = labels.to(device)
                loaders.append(
                    (
                        dataset_id,
                        DataLoader(
                            TensorDataset(data, labels),
                            batch_size=args.batch_size * 2,
                            shuffle=False,
                            num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)),
                            pin_memory=pin_memory,
                        ),
                    )
                )
            return loaders

        if args.selection == "test-val":
            selection_loaders = make_test_loaders()

        current_dir = model_out_root / f"finetuned_block_{block_name}"
        current_dir.mkdir(parents=True, exist_ok=True)
        csv_file = current_dir / "finetune_results.csv"
        with csv_file.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "selection_score", "selection_source"])

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.SmoothL1Loss()
        best_score = float("inf")
        best_model_path = None
        patience_counter = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"{args.head} block {block_index} epoch {epoch+1}", leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                if active_slice is not None:
                    loss = criterion(outputs[:, active_slice], targets[:, active_slice])
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            selection_preds, selection_trues = [], []
            if args.selection == "train-loss":
                selection_score = float(train_loss)
            else:
                scores = []
                for dataset_id, loader in selection_loaders:
                    preds, trues, _, _, _, val_wmrd = run_evaluation(model, loader, device, active_slice)
                    scores.append(val_wmrd)
                    selection_preds.append(preds)
                    selection_trues.append(trues)
                selection_score = float(np.mean(scores))
            with csv_file.open("a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, train_loss, selection_score, args.selection])
            print(f"epoch={epoch+1} loss={train_loss:.6f} selection={args.selection} score={selection_score:.6f}")
            if selection_score < best_score:
                if best_model_path and best_model_path.exists():
                    best_model_path.unlink()
                best_score = selection_score
                best_model_path = current_dir / f"best_model_{best_score:.6f}.pth"
                torch.save(model.state_dict(), best_model_path)
                if selection_preds:
                    np.save(current_dir / "selection_preds.npy", np.concatenate(selection_preds))
                    np.save(current_dir / "selection_trues.npy", np.concatenate(selection_trues))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"early stop after {args.patience} stale epochs")
                    break
        if best_model_path is None:
            raise RuntimeError(f"no best model saved for block {block_name}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"best block selection={args.selection} score={best_score:.6f} ckpt={best_model_path}")

        for dataset_id, loader in make_test_loaders():
            test_dir = current_dir / "test_results" / dataset_id
            test_dir.mkdir(parents=True, exist_ok=True)
            preds, trues, mse, mae, r2, val_wmrd = run_evaluation(model, loader, device, active_slice)
            np.save(test_dir / "preds.npy", preds)
            np.save(test_dir / "trues.npy", trues)
            with (test_dir / "metrics.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["val_mse", "val_mae", "val_r2", "val_wmrd"])
                writer.writerow([mse, mae, r2, val_wmrd])
            print(f"test {dataset_id}: wmrd={val_wmrd:.6f}")
    print("online fine-tuning complete")


if __name__ == "__main__":
    main()
