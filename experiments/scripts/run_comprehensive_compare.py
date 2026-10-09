#!/usr/bin/env python3
"""Run the comprehensive NeuFSD-vs-baseline comparison.

The runner is intentionally resumable. Each baseline command writes a per-task
JSON result under the run root; rerunning the script skips completed tasks
unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from packet_counts import count_raw_packets


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
COMPARE = ROOT / "compare" / "caida_org"
RUN_ROOT_DEFAULT = ROOT / "mainonly_runs_20260623" / "comprehensive_compare_20260630"

MEM_BYTES = [2**k * 1024 for k in range(4, 9)]
MEM_KB = [m // 1024 for m in MEM_BYTES]
TIME_BASELINE_KB = 16
BASELINE_ALGS = ["elastic", "mrac", "array", "hash", "davinci"]
PLOT_ALGS = ["NeuFSD", "Elastic", "MRAC", "Array Sample", "Hash Sample", "DaVinci"]
ALG_LABEL = {
    "elastic": "Elastic",
    "mrac": "MRAC",
    "array": "Array Sample",
    "hash": "Hash Sample",
    "davinci": "DaVinci",
}

DATASETS = {
    "caida2016": {
        "label": "CAIDA2016",
        "data": ROOT / "data_full" / "caida_2016" / "caida_1min_split",
        "record_size": 16,
        "key_offset": 8,
        "key_len": 8,
    },
    "caida2018": {
        "label": "CAIDA2018",
        "data": ROOT / "data_full" / "caida_2018" / "caida_1min_split",
        "record_size": 21,
        "key_offset": 0,
        "key_len": 13,
    },
    "imc": {
        "label": "IMC",
        "data": ROOT / "data_full" / "imc" / "caida_1min_split",
        "record_size": 13,
        "key_offset": 0,
        "key_len": 13,
    },
    "mawi": {
        "label": "MAWI",
        "data": ROOT / "data_full" / "mawi" / "caida_1min_split",
        "record_size": 13,
        "key_offset": 0,
        "key_len": 13,
    },
}

FINAL_E5 = ROOT / "mainonly_runs_20260623" / "sliding_fullstack_online" / "fourdataset_s2000" / "final_e5"
OURS_CANDIDATES = {
    "caida2016": [
        FINAL_E5 / "selective_gate_caida2016_hmed0p50_scale1p08" / "caida_2016_avg5_selective_window_metrics.csv",
        FINAL_E5 / "caida2016_avg5_e5_s2000_newpre" / "window_metrics.csv",
    ],
    "caida2018": [
        FINAL_E5 / "caida2018_avg5_e5_s2000_newpre" / "window_metrics.csv",
    ],
    "imc": [
        FINAL_E5 / "imc_avg5_e5_s2000_newpre_imcdefault" / "window_metrics.csv",
        FINAL_E5 / "imc_avg5_e5_s2000_newpre" / "window_metrics.csv",
    ],
    "mawi": [
        FINAL_E5 / "mawi_gate_w0p30_20260628" / "mawi_avg5_w0p30" / "window_metrics.csv",
        FINAL_E5 / "mawi_avg5_e5_s2000_newpre" / "window_metrics.csv",
    ],
}

METRIC_PAIRS = [
    ("selective_mrd", "selective_wmrd", "selective"),
    ("sample_shape_mrd", "sample_shape_wmrd", "sample_shape"),
    ("twoway_gate_mrd", "twoway_gate_wmrd", "twoway"),
    ("gate_mrd", "gate_wmrd", "residual"),
    ("adaptive_mrd", "adaptive_wmrd", "adaptive"),
    ("mrd", "wmrd", "raw"),
]


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=True)


def style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def dataset_max_window(trace: str) -> int:
    files = sorted(DATASETS[trace]["data"].glob("dataset_*.dat"))
    if not files:
        raise FileNotFoundError(DATASETS[trace]["data"])
    return max(int(p.stem.split("_")[-1]) for p in files)


def window_file(trace: str, minute: int) -> Path:
    return DATASETS[trace]["data"] / f"dataset_{minute:04d}.dat"


def packet_count_for_window(trace: str, minute: int) -> int:
    return count_raw_packets(window_file(trace, minute), DATASETS[trace]["record_size"])


def compile_tools(run_root: Path) -> dict[str, str]:
    bin_dir = run_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    run(["make", "-j8"], cwd=COMPARE)
    run(
        [
            "g++",
            "-std=c++11",
            "-O3",
            "-mavx2",
            "-mbmi",
            "-mbmi2",
            "-Wno-psabi",
            "-o",
            str(COMPARE / "traditional_sample"),
            "traditional_sample.cpp",
            "src/common/BOBHash32.cpp",
            "-I.",
        ],
        cwd=COMPARE,
    )
    run(
        [
            "g++",
            "-std=c++17",
            "-O3",
            "-mavx2",
            "-mbmi",
            "-mbmi2",
            "-Wno-psabi",
            "-Wno-sign-compare",
            "-Wno-unused-variable",
            "-DMAX_PRIME32=1229",
            "-I.",
            "-Isrc",
            "-Isrc/common",
            "-Isrc/Sketchs",
            "-o",
            str(COMPARE / "bin" / "davinci_runner"),
            "davinci_runner.cpp",
            "src/Sketchs/DaVinci/HeavyPart.cpp",
            "src/common/BOBHash32.cpp",
        ],
        cwd=COMPARE,
    )
    canonicalizer = bin_dir / "canonicalize_trace"
    run(
        [
            "g++",
            "-std=c++17",
            "-O3",
            "-o",
            str(canonicalizer),
            str(ROOT / "scripts" / "canonicalize_trace.cpp"),
        ]
    )
    return {
        "sketch_test": str(COMPARE / "bin" / "sketch_test"),
        "traditional_sample": str(COMPARE / "traditional_sample"),
        "davinci": str(COMPARE / "bin" / "davinci_runner"),
        "canonicalizer": str(canonicalizer),
    }


def prepare_one_canonical(args: tuple[str, int, str, str]) -> tuple[str, int, str]:
    trace, minute, run_root_str, canonicalizer = args
    run_root = Path(run_root_str)
    meta = DATASETS[trace]
    src = window_file(trace, minute)
    dst = run_root / "canonical" / trace / f"{minute}.dat"
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected = 13 * packet_count_for_window(trace, minute)
    if dst.exists() and dst.stat().st_size == expected and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return trace, minute, "cached"
    tmp = dst.with_suffix(".dat.tmp")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        canonicalizer,
        str(src),
        str(tmp),
        str(meta["record_size"]),
        str(meta["key_offset"]),
        str(meta["key_len"]),
    ]
    subprocess.run(cmd, text=True, capture_output=True, check=True)
    if tmp.stat().st_size != expected:
        raise ValueError(f"Canonical window {tmp} must contain {expected} bytes")
    tmp.replace(dst)
    return trace, minute, "built"


def prepare_canonical_windows(run_root: Path, binaries: dict[str, str], windows_by_trace: dict[str, list[int]], workers: int) -> None:
    tasks = [
        (trace, minute, str(run_root), binaries["canonicalizer"])
        for trace, minutes in windows_by_trace.items()
        for minute in minutes
    ]
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as ex:
        futures = [ex.submit(prepare_one_canonical, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            trace, minute, status = fut.result()
            if i % 25 == 0 or status == "built":
                print(f"[canonical] {i}/{len(tasks)} {trace} minute={minute} {status}", flush=True)

    for trace, minutes in windows_by_trace.items():
        data_dir = run_root / "baseline_data" / trace
        data_dir.mkdir(parents=True, exist_ok=True)
        for minute in minutes:
            link = data_dir / f"{minute}.dat"
            target = run_root / "canonical" / trace / f"{minute}.dat"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)


def result_path(run_root: Path, trace: str, minute: int, alg: str, mem: int) -> Path:
    return run_root / "tasks" / trace / f"m{minute:04d}_{alg}_{mem}" / "result.json"


def cached_baseline_result(path: Path, trace: str, minute: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("packets") != packet_count_for_window(trace, minute):
        return None
    if path.stat().st_mtime_ns < window_file(trace, minute).stat().st_mtime_ns:
        return None
    return result


def parse_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_stdout(alg: str, stdout: str, csv_path: Path | None = None) -> dict[str, float | None]:
    mrd = parse_float(r"^MRD:\s*([0-9.eE+-]+)", stdout)
    wmrd = parse_float(r"^WMRD:\s*([0-9.eE+-]+)", stdout)
    insert = parse_float(r"(?:Insertion time|Insert time):\s*([0-9.eE+-]+)\s*ms", stdout)
    decode = parse_float(r"Distribution calculation(?: \(decode\))? time:\s*([0-9.eE+-]+)\s*ms", stdout)
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if wmrd is None and "WMRD" in df.columns:
                wmrd = float(df["WMRD"].iloc[-1])
            if mrd is None:
                if "MRD" in df.columns:
                    mrd = float(df["MRD"].iloc[-1])
                elif "ARE" in df.columns:
                    mrd = float(df["ARE"].iloc[-1])
            if insert is None and "Insert time" in df.columns:
                insert = float(df["Insert time"].iloc[-1])
            if decode is None and "Decode time" in df.columns:
                decode = float(df["Decode time"].iloc[-1])
        except Exception:
            pass
    return {"mrd": mrd, "wmrd": wmrd, "insert_ms": insert, "decode_ms": decode}


def run_baseline_task(task: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(task["run_root"])
    trace = task["trace"]
    minute = int(task["minute"])
    alg = task["algorithm"]
    mem = int(task["memory_bytes"])
    out_json = result_path(run_root, trace, minute, alg, mem)
    packet_count = packet_count_for_window(trace, minute)
    if not task.get("force"):
        cached = cached_baseline_result(out_json, trace, minute)
        if cached is not None:
            return cached

    workdir = out_json.parent
    workdir.mkdir(parents=True, exist_ok=True)
    data_dir = run_root / "baseline_data" / trace
    canonical_count = count_raw_packets(data_dir / f"{minute}.dat", 13)
    if canonical_count != packet_count:
        raise ValueError(f"Canonical window count {canonical_count} differs from raw count {packet_count}")
    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"
    csv_path = workdir / "out.csv"
    binaries = task["binaries"]

    if alg in {"elastic", "mrac"}:
        cmd = [
            binaries["sketch_test"],
            "-t",
            alg,
            "-d",
            str(data_dir) + "/",
            "-s",
            str(minute + 1),
            "-e",
            str(minute + 1),
            "-m",
            str(mem),
            "-r",
            "1",
        ]
    elif alg in {"array", "hash"}:
        cmd = [
            binaries["traditional_sample"],
            "-t",
            alg,
            "-d",
            str(data_dir) + "/",
            "-s",
            str(minute),
            "-e",
            str(minute),
            "-m",
            str(mem),
            "-o",
            str(csv_path),
        ]
    elif alg == "davinci":
        cmd = [
            binaries["davinci"],
            "-d",
            str(data_dir) + "/",
            "-s",
            str(minute + 1),
            "-e",
            str(minute + 1),
            "-m",
            str(mem),
            "-o",
            str(csv_path),
        ]
    else:
        raise ValueError(alg)

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, timeout=task["timeout_sec"])
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        parsed = parse_stdout(alg, proc.stdout, csv_path)
        status = "ok" if proc.returncode == 0 else "failed"
        err = "" if proc.returncode == 0 else f"returncode={proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        parsed = {"mrd": None, "wmrd": None, "insert_ms": None, "decode_ms": None}
        status = "timeout"
        err = f"timeout after {task['timeout_sec']}s"

    result = {
        "trace": trace,
        "minute": minute,
        "algorithm": alg,
        "algorithm_label": ALG_LABEL[alg],
        "memory_bytes": mem,
        "memory_kb": mem // 1024,
        "packets": packet_count,
        "status": status,
        "error": err,
        "elapsed_sec": time.time() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        **parsed,
    }
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out_json)
    return result


def collect_baseline_results(run_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((run_root / "tasks").glob("*/*/result.json")):
        try:
            row = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if cached_baseline_result(p, row["trace"], int(row["minute"])) is None:
            print(f"[baseline] ignoring stale result {p}; rerun the corresponding window", flush=True)
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def write_baseline_csv(run_root: Path) -> pd.DataFrame:
    df = collect_baseline_results(run_root)
    out = run_root / "baseline_detail.csv"
    if not df.empty:
        df.sort_values(["trace", "minute", "algorithm", "memory_kb"]).to_csv(out, index=False)
    return df


def load_ours(trace: str) -> tuple[pd.DataFrame, str, Path]:
    best: tuple[float, pd.DataFrame, str, Path, str, str] | None = None
    for path in OURS_CANDIDATES[trace]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for mrd_col, wmrd_col, name in METRIC_PAIRS:
            if mrd_col in df.columns and wmrd_col in df.columns:
                mean_w = float(pd.to_numeric(df[wmrd_col], errors="coerce").mean())
                if best is None or mean_w < best[0]:
                    best = (mean_w, df.copy(), name, path, mrd_col, wmrd_col)
    if best is None:
        raise FileNotFoundError(f"No NeuFSD metrics found for {trace}")
    _, df, name, path, mrd_col, wmrd_col = best
    out = df[["minute", mrd_col, wmrd_col]].copy()
    out.columns = ["minute", "mrd", "wmrd"]
    out["algorithm"] = "NeuFSD"
    out["memory_kb"] = 16
    return out, name, path


def measure_forward_time(run_root: Path, force: bool = False) -> dict[str, Any]:
    out = run_root / "forward_timing.json"
    if out.exists() and not force:
        return json.loads(out.read_text())

    script = r'''
import importlib.util, json, math, os, sys, time
from pathlib import Path
import torch
import torch.nn as nn
root = Path(sys.argv[1])
cfg = root / "configs" / "64_64_caida_2018" / "model.py"
backend = "repo_custom_vit"
try:
    spec = importlib.util.spec_from_file_location("mrac_model", cfg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Model = mod.CustomViT
except Exception as exc:
    backend = "native_vit_fallback:" + exc.__class__.__name__
    class NativeViT(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.patch = nn.Conv2d(3, 768, kernel_size=16, stride=16)
            self.cls = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos = nn.Parameter(torch.zeros(1, 17, 768))
            layer = nn.TransformerEncoderLayer(
                d_model=768, nhead=12, dim_feedforward=3072, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=12)
            self.head = nn.Sequential(
                nn.LayerNorm(768), nn.Linear(768, 1000), nn.GELU(),
                nn.Dropout(0.0), nn.Linear(1000, out_dim))
        def forward(self, x):
            x = self.patch(x).flatten(2).transpose(1, 2)
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos
            x = self.enc(x)[:, 0]
            return self.head(x)
    Model = NativeViT
def bench(device_name, reps, warmup):
    device = torch.device(device_name)
    models = [Model(out_dim=10).to(device).eval(),
              Model(out_dim=9990).to(device).eval()]
    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            for m in models:
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            for m in models:
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / reps
res = {"cpu_ms": bench("cpu", 20, 5), "cpu_threads": torch.get_num_threads(), "backend": backend}
if torch.cuda.is_available():
    res["h800_ms"] = bench("cuda", 120, 20)
    res["cuda_name"] = torch.cuda.get_device_name(0)
print(json.dumps(res))
'''
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(ROOT)],
        text=True,
        capture_output=True,
        env=env,
        timeout=600,
        check=True,
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    return res


def schedule_tasks(
    run_root: Path,
    binaries: dict[str, str],
    sweep_windows: dict[str, list[int]],
    time_windows: dict[str, list[int]],
    force: bool,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    tasks = []
    seen: set[tuple[str, int, str, int]] = set()
    for trace, minutes in sweep_windows.items():
        for minute in minutes:
            for mem in MEM_BYTES:
                for alg in BASELINE_ALGS:
                    seen.add((trace, minute, alg, mem))
    for trace, minutes in time_windows.items():
        for minute in minutes:
            for alg in BASELINE_ALGS:
                seen.add((trace, minute, alg, TIME_BASELINE_KB * 1024))
    for trace, minute, alg, mem in sorted(seen):
        out_json = result_path(run_root, trace, minute, alg, mem)
        if not force and cached_baseline_result(out_json, trace, minute) is not None:
            continue
        tasks.append(
            {
                "run_root": str(run_root),
                "trace": trace,
                "minute": minute,
                "algorithm": alg,
                "memory_bytes": mem,
                "binaries": binaries,
                "force": force,
                "timeout_sec": timeout_sec,
            }
        )
    return tasks


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> None:
    if not tasks:
        print("[baseline] no pending tasks", flush=True)
        return
    print(f"[baseline] running {len(tasks)} pending tasks with {workers} workers", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_baseline_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if done % 20 == 0 or res["status"] != "ok":
                print(
                    f"[baseline] {done}/{len(tasks)} {res['trace']} m={res['minute']} "
                    f"{res['algorithm']} {res['memory_kb']}KB {res['status']}",
                    flush=True,
                )


def summarize_with_range(values: pd.Series) -> tuple[float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan, math.nan
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def insert_speed_mpps(df: pd.DataFrame) -> pd.Series:
    packets = pd.to_numeric(df["packets"], errors="coerce")
    milliseconds = pd.to_numeric(df["insert_ms"], errors="coerce")
    return (packets / milliseconds.where(milliseconds > 0) / 1000.0).where(packets >= 0)


def ours_insert_speed_from_elastic(df: pd.DataFrame, trace: str, minutes: list[int]) -> np.ndarray:
    rows = df[
        (df.trace == trace)
        & (df.algorithm == "elastic")
        & (df.memory_kb == 16)
        & (df.minute.isin(minutes))
        & (df.status == "ok")
    ]
    return insert_speed_mpps(rows).dropna().to_numpy(dtype=float)


def plot_memory_sweep(
    run_root: Path,
    df: pd.DataFrame,
    sweep_windows: dict[str, list[int]],
    ours_cache: dict[str, pd.DataFrame],
    timing: dict[str, Any],
) -> None:
    style()
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "NeuFSD": "#d62728",
        "Elastic": "#1f77b4",
        "MRAC": "#ff7f0e",
        "Array Sample": "#2ca02c",
        "Hash Sample": "#9467bd",
        "DaVinci": "#8c564b",
    }
    markers = {"Elastic": "o", "MRAC": "s", "Array Sample": "^", "Hash Sample": "D", "DaVinci": "v"}

    specs = [
        ("wmrd", "WMRD", "WMRD", True),
        ("mrd", "MRD", "MRD", True),
        ("decode_ms", "Decode Time (s)", "Decode Time", True),
        ("insert_speed", "Insert Speed (Mpps)", "Insert Speed", False),
    ]

    for metric, ylabel, stem, logy in specs:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.3), sharex=True)
        for ax, trace in zip(axes, DATASETS):
            minutes = sweep_windows[trace]
            ax.set_title(DATASETS[trace]["label"], fontweight="bold", fontsize=22)
            ax.set_xscale("log", base=2)
            ax.set_xlim(MEM_KB[0], MEM_KB[-1])
            ax.set_xticks(MEM_KB)
            ax.set_xticklabels([str(k) for k in MEM_KB])
            ax.tick_params(labelsize=17)
            ax.grid(True, linestyle="--", axis="both", alpha=0.55)
            if logy:
                ax.set_yscale("log")

            for alg in BASELINE_ALGS:
                if metric == "decode_ms" and alg in {"array", "hash"}:
                    continue
                label = ALG_LABEL[alg]
                means, lows, highs = [], [], []
                for kb in MEM_KB:
                    sub = df[
                        (df.trace == trace)
                        & (df.algorithm == alg)
                        & (df.memory_kb == kb)
                        & (df.minute.isin(minutes))
                        & (df.status == "ok")
                    ]
                    if metric == "insert_speed":
                        vals = insert_speed_mpps(sub)
                    elif metric == "decode_ms":
                        vals = pd.to_numeric(sub["decode_ms"], errors="coerce") / 1000.0
                        vals = vals.mask(vals <= 0, 1e-6)
                    else:
                        vals = pd.to_numeric(sub[metric], errors="coerce")
                    mean, lo, hi = summarize_with_range(vals)
                    means.append(mean)
                    lows.append(lo)
                    highs.append(hi)
                y = np.array(means, dtype=float)
                low_arr = np.array(lows, dtype=float)
                high_arr = np.array(highs, dtype=float)
                yerr = np.vstack([
                    np.maximum(y - low_arr, 0.0),
                    np.maximum(high_arr - y, 0.0),
                ])
                ax.errorbar(
                    MEM_KB,
                    y,
                    yerr=yerr,
                    label=label,
                    color=colors[label],
                    marker=markers[label],
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.6,
                    linewidth=2.0,
                    capsize=4,
                    elinewidth=1.4,
                )

            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(minutes)]
            if metric in {"wmrd", "mrd"}:
                vals = pd.to_numeric(ours[metric], errors="coerce").dropna().to_numpy(dtype=float)
            elif metric == "decode_ms":
                vals = np.array([])
            else:
                vals = ours_insert_speed_from_elastic(df, trace, minutes)
            if len(vals):
                ax.hlines(np.mean(vals), MEM_KB[0], MEM_KB[-1], colors=colors["NeuFSD"], linewidth=3.0, label="NeuFSD")
                ax.fill_between(MEM_KB, np.min(vals), np.max(vals), color=colors["NeuFSD"], alpha=0.12)
            if metric == "decode_ms":
                if "h800_ms" in timing:
                    ax.hlines(
                        timing["h800_ms"] / 1000.0,
                        MEM_KB[0],
                        MEM_KB[-1],
                        colors=colors["NeuFSD"],
                        linewidth=3.0,
                        label="NeuFSD-H800",
                    )
                ax.hlines(
                    timing["cpu_ms"] / 1000.0,
                    MEM_KB[0],
                    MEM_KB[-1],
                    colors=colors["NeuFSD"],
                    linestyles="--",
                    linewidth=3.0,
                    label="NeuFSD-CPU",
                )

            if ax is axes[0]:
                ax.set_ylabel(ylabel, fontweight="bold", fontsize=22)
            ax.set_xlabel("Memory (KB)", fontweight="bold", fontsize=22)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(plot_dir / f"memory_sweep_{stem.lower().replace(' ', '_')}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_time_series(
    run_root: Path,
    df: pd.DataFrame,
    time_windows: dict[str, list[int]],
    ours_cache: dict[str, pd.DataFrame],
) -> None:
    style()
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "NeuFSD": "#d62728",
        "Elastic": "#1f77b4",
        "MRAC": "#ff7f0e",
        "Array Sample": "#2ca02c",
        "Hash Sample": "#9467bd",
        "DaVinci": "#8c564b",
    }
    for trace, minutes in time_windows.items():
        for metric, ylabel in [("wmrd", "WMRD"), ("mrd", "MRD")]:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))
            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(minutes)].sort_values("minute")
            ax.plot(
                ours["minute"],
                ours[metric],
                color=colors["NeuFSD"],
                linewidth=3.0,
                label="NeuFSD",
            )
            for alg in BASELINE_ALGS:
                sub = df[
                    (df.trace == trace)
                    & (df.algorithm == alg)
                    & (df.memory_kb == TIME_BASELINE_KB)
                    & (df.minute.isin(minutes))
                    & (df.status == "ok")
                ].sort_values("minute")
                if sub.empty:
                    continue
                ax.plot(
                    sub["minute"],
                    pd.to_numeric(sub[metric], errors="coerce"),
                    linewidth=2.2,
                    label=ALG_LABEL[alg],
                    color=colors[ALG_LABEL[alg]],
                )
            ax.tick_params(labelsize=19)
            ax.set_xlabel("Time (min)", fontweight="bold", fontsize=24)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=24)
            ax.grid(True, linestyle="--", axis="both", alpha=0.55)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False, fontsize=17)
            fig.tight_layout()
            fig.savefig(plot_dir / f"timeseries_{trace}_{metric}.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_system_timeline(run_root: Path) -> None:
    style()
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.2))
    rows = [
        ("NeuFSD", [(0.0, 0.08, "Decode"), (0.2, 4.0, "Build SFT data"), (4.4, 8.0, "SFT"), (12.0, 48.0, "Idle / next window")], "#d62728"),
        ("Sketch+EM", [(0.0, 18.0, "EM decode"), (18.2, 30.0, "Output")], "C0"),
        ("Sampling", [(0.0, 1.0, "Scale sample FSD"), (1.2, 10.0, "Output")], "C2"),
    ]
    y_positions = [2, 1, 0]
    for (name, spans, color), y in zip(rows, y_positions):
        ax.text(-2.0, y, name, ha="right", va="center", fontweight="bold", fontsize=18)
        for start, dur, label in spans:
            ax.broken_barh([(start, dur)], (y - 0.28, 0.56), facecolors=color, alpha=0.82)
            ax.text(start + dur / 2, y, label, ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.axvline(60, color="black", linewidth=1.8)
    ax.text(60, 2.55, "1-min window boundary", ha="right", va="bottom", fontsize=15, fontweight="bold")
    ax.set_xlim(-8, 62)
    ax.set_ylim(-0.7, 2.8)
    ax.set_yticks([])
    ax.set_xlabel("Elapsed Time Since Window Ends (s)", fontweight="bold", fontsize=22)
    ax.tick_params(labelsize=17)
    ax.grid(True, linestyle="--", axis="x", alpha=0.55)
    fig.tight_layout()
    fig.savefig(plot_dir / "online_window_timing_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


def write_ours_detail(run_root: Path, ours_cache: dict[str, pd.DataFrame], chosen: dict[str, tuple[str, Path]]) -> None:
    rows = []
    for trace, df in ours_cache.items():
        d = df.copy()
        d["trace"] = trace
        d["dataset"] = DATASETS[trace]["label"]
        d["memory_kb"] = 16
        d["optimization"] = chosen[trace][0]
        d["source_csv"] = str(chosen[trace][1])
        rows.append(d)
    pd.concat(rows, ignore_index=True).to_csv(run_root / "neufsd_detail.csv", index=False)


def write_summary(run_root: Path, df: pd.DataFrame, ours_cache: dict[str, pd.DataFrame], sweep_windows: dict[str, list[int]], time_windows: dict[str, list[int]], timing: dict[str, Any]) -> None:
    rows = []
    for trace in DATASETS:
        for phase, windows in [("memory10", sweep_windows[trace]), ("timeseries", time_windows[trace])]:
            ours = ours_cache[trace]
            ours = ours[ours.minute.isin(windows)]
            ours_insert_vals = ours_insert_speed_from_elastic(df, trace, list(windows))
            rows.append({
                "phase": phase,
                "trace": trace,
                "dataset": DATASETS[trace]["label"],
                "algorithm": "NeuFSD",
                "memory_kb": 16,
                "n_windows": len(ours),
                "mean_mrd": ours["mrd"].mean(),
                "mean_wmrd": ours["wmrd"].mean(),
                "mean_decode_ms_h800": timing.get("h800_ms", math.nan),
                "mean_decode_ms_cpu": timing.get("cpu_ms", math.nan),
                "mean_insert_mpps": float(np.mean(ours_insert_vals)) if len(ours_insert_vals) else math.nan,
            })
            for alg in BASELINE_ALGS:
                mems = MEM_KB if phase == "memory10" else [TIME_BASELINE_KB]
                for kb in mems:
                    sub = df[
                        (df.trace == trace)
                        & (df.algorithm == alg)
                        & (df.memory_kb == kb)
                        & (df.minute.isin(windows))
                        & (df.status == "ok")
                    ]
                    if sub.empty:
                        continue
                    rows.append({
                        "phase": phase,
                        "trace": trace,
                        "dataset": DATASETS[trace]["label"],
                        "algorithm": ALG_LABEL[alg],
                        "memory_kb": kb,
                        "n_windows": len(sub),
                        "mean_mrd": pd.to_numeric(sub["mrd"], errors="coerce").mean(),
                        "mean_wmrd": pd.to_numeric(sub["wmrd"], errors="coerce").mean(),
                        "mean_decode_ms_h800": math.nan,
                        "mean_decode_ms_cpu": pd.to_numeric(sub["decode_ms"], errors="coerce").mean(),
                        "mean_insert_mpps": insert_speed_mpps(sub).mean(),
                    })
    pd.DataFrame(rows).to_csv(run_root / "summary_compare.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--start-minute", type=int, default=5)
    ap.add_argument("--sweep-windows", type=int, default=10)
    ap.add_argument("--time-windows", type=int, default=100)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--timeout-sec", type=int, default=1200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="Only collect and plot existing result JSON files.")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    args.run_root = args.run_root.resolve()

    args.run_root.mkdir(parents=True, exist_ok=True)
    traces = list(DATASETS)
    sweep_windows: dict[str, list[int]] = {}
    time_windows: dict[str, list[int]] = {}
    all_windows: dict[str, list[int]] = {}
    for trace in traces:
        max_w = dataset_max_window(trace)
        sweep_end = min(max_w, args.start_minute + args.sweep_windows - 1)
        time_end = min(max_w, args.start_minute + args.time_windows - 1)
        sweep_windows[trace] = list(range(args.start_minute, sweep_end + 1))
        time_windows[trace] = list(range(args.start_minute, time_end + 1))
        all_windows[trace] = sorted(set(sweep_windows[trace]) | set(time_windows[trace]))
        if len(time_windows[trace]) < args.time_windows:
            print(f"[warn] {trace} has only {len(time_windows[trace])} measurable windows from minute {args.start_minute}")

    binaries = compile_tools(args.run_root)
    if not args.skip_run:
        prepare_canonical_windows(args.run_root, binaries, all_windows, args.workers)
        tasks = schedule_tasks(args.run_root, binaries, sweep_windows, time_windows, args.force, args.timeout_sec)
        run_tasks(tasks, args.workers)

    df = write_baseline_csv(args.run_root)
    if df.empty:
        raise RuntimeError("No baseline results collected")
    timing = measure_forward_time(args.run_root, force=args.force)

    ours_cache: dict[str, pd.DataFrame] = {}
    chosen: dict[str, tuple[str, Path]] = {}
    for trace in traces:
        ours, opt_name, src = load_ours(trace)
        ours_cache[trace] = ours
        chosen[trace] = (opt_name, src)
    write_ours_detail(args.run_root, ours_cache, chosen)
    write_summary(args.run_root, df, ours_cache, sweep_windows, time_windows, timing)

    if not args.skip_plots:
        plot_memory_sweep(args.run_root, df, sweep_windows, ours_cache, timing)
        plot_time_series(args.run_root, df, time_windows, ours_cache)
        plot_system_timeline(args.run_root)

    metadata = {
        "run_root": str(args.run_root),
        "memory_kb": MEM_KB,
        "sweep_windows": sweep_windows,
        "time_windows": time_windows,
        "canonicalization": "13B records; first 4B are FNV-1a hash of the full original flow key, remaining bytes are zero",
        "neufsd_sources": {k: {"optimization": v[0], "csv": str(v[1])} for k, v in chosen.items()},
        "forward_timing": timing,
    }
    (args.run_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[done] wrote results under {args.run_root}")


if __name__ == "__main__":
    main()
