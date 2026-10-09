#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


HEAD_LABEL = {
    "1_10": "1_10_chazhi",
    "10_1e4": "10_1e4_chazhi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MRAC ViT on exact pretraining counter stores.")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", type=Path, required=True, help="Roots containing input_store/ and label dirs")
    parser.add_argument("--head", choices=sorted(HEAD_LABEL), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("PRETRAIN_EPOCHS", 10)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PRETRAIN_BATCH_SIZE", 64)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("PRETRAIN_LR", 1e-2)))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--val-tail-per-root",
        type=int,
        default=0,
        help=(
            "If >0, reserve the last N labeled datasets from each root for validation. "
            "This keeps every trace represented in both train and validation splits."
        ),
    )
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_imports(config_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    local_deps = root / "local_deps"
    for path in [root / "src", local_deps, config_dir]:
        if path.exists():
            sys.path.insert(0, str(path))


def list_dataset_ids(root: Path) -> list[str]:
    index_path = root / "input_store" / "index.json"
    if index_path.exists():
        with index_path.open() as f:
            return sorted(json.load(f)["datasets"].keys())
    input_dir = root / "input"
    return sorted(p.name for p in input_dir.iterdir() if p.is_dir())


def wmrd(preds: np.ndarray, trues: np.ndarray) -> float:
    preds = np.maximum(preds, 0)
    return float(np.mean(np.abs(preds - trues)) / np.mean((preds + trues) / 2) * 100)


def evaluate(model, loader, device) -> float:
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(targets.numpy())
    return wmrd(np.concatenate(all_preds), np.concatenate(all_trues))


def main() -> None:
    args = parse_args()
    setup_imports(args.config_dir)
    from model import CustomViT
    from mrac_data import read_labeled_counter_dataset

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pretraining")
    device = torch.device("cuda:0")

    label_dir = HEAD_LABEL[args.head]
    dataset_refs = []
    per_root_refs: list[list[tuple[Path, str]]] = []
    for root in args.roots:
        root_refs = []
        for dataset_id in list_dataset_ids(root):
            if (root / label_dir / f"{dataset_id}.npy").exists():
                root_refs.append((root, dataset_id))
                dataset_refs.append((root, dataset_id))
        per_root_refs.append(root_refs)

    if args.val_tail_per_root > 0:
        train_refs = []
        val_refs = []
        for root_refs in per_root_refs:
            if not root_refs:
                continue
            val_count = min(args.val_tail_per_root, max(1, len(root_refs) - 1))
            train_refs.extend(root_refs[:-val_count])
            val_refs.extend(root_refs[-val_count:])
    elif args.max_datasets:
        dataset_refs = dataset_refs[: args.max_datasets]
        if len(dataset_refs) < 2:
            raise RuntimeError("Need at least two labeled datasets for pretraining")
        val_count = max(1, int(round(len(dataset_refs) * args.val_frac)))
        val_refs = dataset_refs[-val_count:]
        train_refs = dataset_refs[:-val_count]
    else:
        if len(dataset_refs) < 2:
            raise RuntimeError("Need at least two labeled datasets for pretraining")
        val_count = max(1, int(round(len(dataset_refs) * args.val_frac)))
        val_refs = dataset_refs[-val_count:]
        train_refs = dataset_refs[:-val_count]
    if not train_refs or not val_refs:
        raise RuntimeError("Need non-empty train and validation splits for pretraining")
    print(f"pretrain refs: train={len(train_refs)} val={len(val_refs)} head={args.head}")

    def load_refs(refs):
        data_list, label_list = [], []
        for root, dataset_id in tqdm(refs, desc="loading pretrain data"):
            data, labels = read_labeled_counter_dataset(root, dataset_id, label_dir)
            data_list.append(data)
            label_list.append(labels)
        return torch.cat(data_list, dim=0), torch.cat(label_list, dim=0)

    train_x, train_y = load_refs(train_refs)
    val_x, val_y = load_refs(val_refs)
    out_dim = train_y.shape[-1]
    model = CustomViT(out_dim=out_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)),
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = None
    best_wmrd = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"pretrain epoch {epoch + 1}/{args.epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_wmrd = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} train_loss={total_loss / len(train_loader):.6f} val_wmrd={val_wmrd:.6f}")
        if val_wmrd < best_wmrd:
            if best_path and best_path.exists():
                best_path.unlink()
            best_wmrd = val_wmrd
            best_path = args.out_dir / f"best_model_{best_wmrd:.6f}.pth"
            torch.save(model.state_dict(), best_path)
    print(f"best_wmrd={best_wmrd:.6f} best_path={best_path}")


if __name__ == "__main__":
    main()
