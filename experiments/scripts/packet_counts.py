"""Packet totals from complete fixed-width raw windows, independent of FSD bins."""
from __future__ import annotations

import math
from pathlib import Path


def default_packet_format(trace: str) -> tuple[int, int, int]:
    if trace == "caida_2016":
        return 16, 8, 8
    if trace in {"caida_2018", "caida_2018_new"}:
        return 21, 0, 13
    if trace in {"caida_org", "imc", "mawi", "key13"}:
        return 13, 0, 13
    raise ValueError(f"Unknown raw packet format for trace={trace}; pass explicit packet parser settings.")


def count_raw_packets(path: Path, packet_size: int) -> int:
    if isinstance(packet_size, bool) or not isinstance(packet_size, int) or packet_size <= 0:
        raise ValueError("packet_size must be a positive integer")
    size = path.stat().st_size
    if not path.is_file():
        raise ValueError(f"Raw window is not a file: {path}")
    packet_count, remainder = divmod(size, packet_size)
    if remainder:
        raise ValueError(f"Raw window {path} has {size} bytes, not divisible by packet size {packet_size}")
    return packet_count


def window_packet_count(raw_dir: Path, minute: int, trace: str, packet_size: int = 0) -> int:
    record_size = packet_size if packet_size != 0 else default_packet_format(trace)[0]
    return count_raw_packets(raw_dir / f"dataset_{minute:04d}.dat", record_size)


def validate_packet_count(packet_count: int | float) -> int:
    if isinstance(packet_count, bool) or not math.isfinite(packet_count) or packet_count < 0 or int(packet_count) != packet_count:
        raise ValueError("packet_count must be a nonnegative integer from the raw window")
    return int(packet_count)
