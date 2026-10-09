from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


def counter_to_image(counter: np.ndarray) -> np.ndarray:
    arr = np.asarray(counter, dtype=np.float32)
    feature_mode = os.environ.get("COUNTER_FEATURE_MODE", "three").lower()
    if feature_mode in {"three", "sort_origin_sort", "sorted_origin_reverse"}:
        ascending = np.sort(arr)
        descending = ascending[::-1]
        image = np.concatenate((ascending, arr, descending)).astype(np.float32, copy=False)
    elif feature_mode in {"origin", "raw_origin"}:
        image = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown COUNTER_FEATURE_MODE={feature_mode}")

    mode = os.environ.get("COUNTER_INPUT_MODE", "raw").lower()
    if mode == "raw":
        return image
    if mode == "log1p":
        return np.log1p(image).astype(np.float32, copy=False)
    if mode == "log1p_total":
        image = np.log1p(image).astype(np.float32, copy=False)
        denom = max(float(np.log1p(np.sum(arr, dtype=np.float64))), 1.0)
        return image / denom
    raise ValueError(f"unknown COUNTER_INPUT_MODE={mode}")


def _image_size(counter_len: int) -> int:
    size = int(round(counter_len ** 0.5))
    if size * size != counter_len:
        raise ValueError(f"counter length is not square: {counter_len}")
    return size


def _read_file_backend(root_dir: Path, dataset_id: str) -> tuple[np.ndarray, int, int]:
    folder = root_dir / "input" / dataset_id
    if not folder.exists():
        raise FileNotFoundError(f"missing counter folder: {folder}")
    files = sorted(folder.iterdir(), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"empty counter folder: {folder}")

    rows = []
    counter_len = None
    for path in files:
        counter = np.fromfile(path, dtype="<u4")
        counter_len = counter.shape[0]
        rows.append(counter_to_image(counter))
    size = _image_size(counter_len or 0)
    channels = rows[0].shape[0] // (counter_len or 1)
    if channels * (counter_len or 0) != rows[0].shape[0]:
        raise ValueError(f"counter image length {rows[0].shape[0]} is not divisible by counter length {counter_len}")
    return np.stack(rows, axis=0), size, channels


@lru_cache(maxsize=32)
def _load_store(index_path: str):
    with open(index_path) as f:
        index = json.load(f)
    data_path = Path(index_path).with_name(index["data_file"])
    shape = tuple(index["shape"])
    mmap = np.memmap(data_path, dtype=np.dtype(index.get("dtype", "<u4")), mode="r", shape=shape)
    return index, mmap


def _read_store_backend(root_dir: Path, dataset_id: str) -> tuple[np.ndarray, int, int]:
    index_path = root_dir / "input_store" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing counter store index: {index_path}")
    index, mmap = _load_store(str(index_path))
    entry = index["datasets"].get(dataset_id)
    if entry is None:
        raise KeyError(f"{dataset_id} not in counter store {index_path}")
    start = int(entry["start"])
    count = int(entry["count"])
    counters = np.asarray(mmap[start : start + count])
    rows = [counter_to_image(row) for row in counters]
    counter_len = int(index["counter_len"])
    size = _image_size(counter_len)
    channels = rows[0].shape[0] // counter_len
    if channels * counter_len != rows[0].shape[0]:
        raise ValueError(f"counter image length {rows[0].shape[0]} is not divisible by counter length {counter_len}")
    return np.stack(rows, axis=0), size, channels


def read_counter_dataset(root_dir_name: str | os.PathLike[str], dataset_id: str) -> tuple[torch.Tensor, int]:
    root_dir = Path(root_dir_name)
    backend = os.environ.get("COUNTER_BACKEND", "auto").lower()
    if backend == "memmap" or (backend == "auto" and (root_dir / "input_store" / "index.json").exists()):
        data, size, channels = _read_store_backend(root_dir, dataset_id)
    else:
        data, size, channels = _read_file_backend(root_dir, dataset_id)
    data = data.reshape(data.shape[0], channels, size, size)
    return torch.from_numpy(data).float(), size


def read_labeled_counter_dataset(
    root_dir_name: str | os.PathLike[str],
    dataset_id: str,
    label_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_tensor, _ = read_counter_dataset(root_dir_name, dataset_id)
    label_path = Path(root_dir_name) / label_dir / f"{dataset_id}.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"missing label: {label_path}")
    label = np.load(label_path).reshape(1, -1)
    label = np.repeat(label, data_tensor.shape[0], axis=0)
    return data_tensor, torch.from_numpy(label).float()
