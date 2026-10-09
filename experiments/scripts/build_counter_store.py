#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


TRACE_FORMAT = {
    "caida_2016": (16, 8, 8),
    "caida_2018": (21, 0, 13),
    "caida_2018_new": (21, 0, 13),
    "imc": (13, 0, 13),
    "mawi": (13, 0, 13),
    "caida2016": (16, 8, 8),
    "caida2018": (21, 0, 13),
    "key13": (13, 0, 13),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact memmap counter store from .dat slices.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True, help="Root that will receive input_store/")
    parser.add_argument("--trace-format", choices=sorted(TRACE_FORMAT), required=True)
    parser.add_argument("--counter-len", type=int, required=True, choices=[4096, 16384])
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=400)
    parser.add_argument("--gen-bin", type=Path, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-blocks", action="store_true")
    return parser.parse_args()


def compile_gen(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    src = root / "src" / "gen_counter_store.c"
    subprocess.run(["gcc", "-O3", "-std=c11", "-DNDEBUG", "-fopenmp", "-o", str(out), str(src)], check=True)
    return out


def dataset_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits or 0)


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parents[1]
    packet_size, key_offset, key_size = TRACE_FORMAT[args.trace_format]
    gen_bin = args.gen_bin or compile_gen(script_root, script_root / "run_tools" / "gen_counter_store")
    if not gen_bin.exists():
        gen_bin = compile_gen(script_root, gen_bin)

    dat_files = sorted(args.input_dir.glob("*.dat"), key=dataset_key)
    if args.limit:
        dat_files = dat_files[: args.limit]
    if not dat_files:
        raise SystemExit(f"no .dat files in {args.input_dir}")

    store_dir = args.out_root / "input_store"
    block_dir = args.tmp_dir or (store_dir / "_blocks")
    if store_dir.exists():
        shutil.rmtree(store_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    datasets = {}
    seed_count = args.end_seed - args.start_seed
    data_path = store_dir / "counters.u4"
    with data_path.open("wb") as merged:
        for idx, dat in enumerate(dat_files, 1):
            dataset_id = dat.stem
            block_path = block_dir / f"{dataset_id}.u4"
            cmd = [
                str(gen_bin),
                "--input", str(dat),
                "--output", str(block_path),
                "--start-seed", str(args.start_seed),
                "--end-seed", str(args.end_seed),
                "--counters", str(args.counter_len),
                "--packet-size", str(packet_size),
                "--key-offset", str(key_offset),
                "--key-size", str(key_size),
            ]
            print(f"[{idx}/{len(dat_files)}] {dataset_id}")
            subprocess.run(cmd, check=True)
            expected_bytes = seed_count * args.counter_len * 4
            actual_bytes = block_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(f"{block_path} size {actual_bytes}, expected {expected_bytes}")
            datasets[dataset_id] = {"start": rows, "count": seed_count}
            with block_path.open("rb") as f:
                shutil.copyfileobj(f, merged, length=16 * 1024 * 1024)
            rows += seed_count
            if not args.keep_blocks:
                block_path.unlink()

    index = {
        "data_file": "counters.u4",
        "dtype": "<u4",
        "shape": [rows, args.counter_len],
        "counter_len": args.counter_len,
        "start_seed": args.start_seed,
        "end_seed": args.end_seed,
        "packet_size": packet_size,
        "key_offset": key_offset,
        "key_size": key_size,
        "datasets": datasets,
    }
    with (store_dir / "index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    if not args.keep_blocks:
        try:
            block_dir.rmdir()
        except OSError:
            pass
    print(f"wrote {data_path}")
    print(f"wrote {store_dir / 'index.json'}")


if __name__ == "__main__":
    main()
