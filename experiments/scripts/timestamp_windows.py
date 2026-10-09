"""Stream timestamped captures into reproducible, contiguous observation windows."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile


MANIFEST = "window_manifest.json"
SCHEMA_VERSION = 1
PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1000), b"\xa1\xb2\xc3\xd4": (">", 1000),
    b"\x4d\x3c\xb2\xa1": ("<", 1), b"\xa1\xb2\x3c\x4d": (">", 1),
}
TIMESTAMP_FORMATS = tuple(
    f"{kind}-{endian}-{unit}"
    for kind, units in (("float64", ("seconds",)),
                        ("uint64", ("seconds", "microseconds", "nanoseconds")))
    for endian in ("le", "be") for unit in units
)


def numeric_key(path):
    name = Path(path).name
    return tuple(int(x) for x in re.findall(r"\d+", name)), name


def source_files(source: Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"source does not exist: {source}")
    files = sorted((p for p in source.iterdir() if p.is_file() and
                    (p.suffix.lower() in (".dat", ".bin", ".pcap", ".cap", ".gz") or not p.suffix)),
                   key=numeric_key)
    if not files:
        raise ValueError(f"no capture or record files in {source}")
    return files


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-mode", choices=("time", "records"), default="time",
                        help="time uses actual timestamps; records is legacy fixed-count chunking")
    parser.add_argument("--window-seconds", type=Decimal, default=Decimal(60))
    parser.add_argument("--window-records", type=int, default=1_000_000)
    parser.add_argument("--last-window", choices=("keep", "drop"), default="keep",
                        help="keep or drop the final observed partial interval (default: keep)")
    parser.add_argument("--input-format", choices=("auto", "pcap", "records"), default="auto")
    parser.add_argument("--timestamp-format", choices=TIMESTAMP_FORMATS)
    parser.add_argument("--timestamp-offset", type=int)
    parser.add_argument("--pcap-policy", choices=("ipv4", "tcp-udp"))


@contextmanager
def open_capture(path: Path):
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        if compressed:
            with gzip.GzipFile(fileobj=raw) as stream:
                yield stream
        else:
            yield raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_key13(packet: bytes, linktype: int, policy: str) -> tuple[bytes | None, str]:
    offset = 0
    if linktype == 1:
        if len(packet) < 14:
            return None, "short_link"
        ethertype = int.from_bytes(packet[12:14], "big")
        offset = 14
        while ethertype in (0x8100, 0x88A8):
            if len(packet) < offset + 4:
                return None, "short_vlan"
            ethertype = int.from_bytes(packet[offset + 2:offset + 4], "big")
            offset += 4
        if ethertype != 0x0800:
            return None, "non_ipv4"
    ip = packet[offset:]
    if len(ip) < 20:
        return None, "short_ipv4"
    if ip[0] >> 4 != 4:
        return None, "non_ipv4"
    ihl = (ip[0] & 15) * 4
    if ihl < 20:
        return None, "invalid_ihl"
    if len(ip) < ihl:
        return None, "short_ipv4"
    protocol = ip[9]
    if policy == "tcp-udp" and protocol not in (6, 17):
        return None, "non_tcp_udp"
    transport_min = {6: 20, 17: 8}.get(protocol, 0)
    if int.from_bytes(ip[2:4], "big") < ihl + transport_min:
        return None, "invalid_ip_length"
    if int.from_bytes(ip[6:8], "big") & 0x1FFF:
        return None, "noninitial_fragment"
    if len(ip) < ihl + transport_min:
        return None, "short_transport"
    ports = ip[ihl:ihl + 4] if transport_min else bytes(4)
    return ip[12:16] + ports[:2] + ip[16:20] + ports[2:] + bytes((protocol,)), "kept"


def iter_pcap(path: Path, policy: str, stats: Counter):
    with open_capture(path) as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in PCAP_MAGIC:
            raise ValueError(f"{path}: expected classic PCAP global header (PCAPNG unsupported)")
        endian, multiplier = PCAP_MAGIC[header[:4]]
        major, minor, _, _, snaplen, network = struct.unpack(endian + "HHiIII", header[4:])
        linktype = network & 0xFFFF
        if (major, minor) != (2, 4) or linktype not in (1, 101):
            raise ValueError(f"{path}: requires PCAP v2.4 Ethernet(1) or raw IPv4(101); got {major}.{minor}/{linktype}")
        unpack = struct.Struct(endian + "IIII").unpack
        while True:
            record = stream.read(16)
            if not record:
                break
            if len(record) != 16:
                raise ValueError(f"{path}: truncated PCAP record header")
            seconds, fraction, captured, wire = unpack(record)
            if fraction >= 1_000_000_000 // multiplier:
                raise ValueError(f"{path}: invalid PCAP timestamp fraction {fraction}")
            if captured > 16 * 1024 * 1024 or captured > wire:
                raise ValueError(f"{path}: invalid captured packet length {captured}/{wire}")
            packet = stream.read(captured)
            if len(packet) != captured:
                raise ValueError(f"{path}: truncated PCAP captured bytes")
            if captured > snaplen:
                stats["caplen_above_snaplen"] += 1
            key, reason = extract_key13(packet, linktype, policy)
            stats["captured_records"] += 1
            stats[reason] += 1
            yield seconds * 1_000_000_000 + fraction * multiplier, key


def timestamp_ns(record: bytes, offset: int, timestamp_format: str) -> int:
    kind, endian, unit = timestamp_format.split("-")
    value = struct.unpack_from(("<" if endian == "le" else ">") + ("d" if kind == "float64" else "Q"),
                               record, offset)[0]
    if kind == "float64":
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {timestamp_format} timestamp: {value}")
        # Split seconds before scaling to retain submicrosecond epoch precision.
        whole = math.floor(value)
        return whole * 1_000_000_000 + round((value - whole) * 1_000_000_000)
    return value * {"seconds": 1_000_000_000, "microseconds": 1000, "nanoseconds": 1}[unit]


def iter_records(path: Path, record_size: int, timestamp_format: str | None,
                 timestamp_offset: int | None, output_key: bool, key_offset: int, stats: Counter):
    with open_capture(path) as stream:
        block_size = max(1, (4 * 1024 * 1024) // record_size) * record_size
        while block := stream.read(block_size):
            if len(block) % record_size:
                raise ValueError(f"{path}: trailing {len(block) % record_size} bytes; record size is {record_size}")
            for offset in range(0, len(block), record_size):
                record = block[offset:offset + record_size]
                stamp = timestamp_ns(record, timestamp_offset, timestamp_format) if timestamp_format else None
                stats["captured_records"] += 1
                stats["kept"] += 1
                yield stamp, record[key_offset:key_offset + 13] if output_key else record


def prepare_windows(input_files, out_dir, *, window_mode="time", window_seconds=60,
                    window_records=1_000_000, last_window="keep", input_format="auto",
                    record_size=13, timestamp_format=None, timestamp_offset=None,
                    output_key=False, key_offset=0, pcap_policy="tcp-udp", max_windows=0,
                    expected_record_size=None):
    """Publish all windows atomically, or validate and reuse an identical existing set.

    Intervals are [first_capture_timestamp + k*T, first_capture_timestamp + (k+1)*T).
    Missing/filtered-empty intervals and timestamp reversals fail without publishing.
    EOF cannot prove a final interval complete, so it is marked partial or dropped.
    """
    files = [Path(p) for p in input_files]
    out_dir = Path(out_dir)
    width = Decimal(str(window_seconds)) * 1_000_000_000
    if not files or window_mode not in ("time", "records") or last_window not in ("keep", "drop"):
        raise ValueError("invalid input files or window policy")
    if not width.is_finite() or width <= 0 or width != width.to_integral_value():
        raise ValueError("--window-seconds must be positive and representable in nanoseconds")
    width_ns = int(width)
    if window_records <= 0 or record_size <= 0 or max_windows < 0:
        raise ValueError("record size/window count must be positive; max-windows cannot be negative")
    formats = []
    for path in files:
        with open_capture(path) as stream:
            magic = stream.read(4)
        formats.append(("pcap" if magic in PCAP_MAGIC else "records") if input_format == "auto" else input_format)
    if len(set(formats)) != 1:
        raise ValueError("mixed PCAP and fixed-record inputs; process them separately")
    detected = formats[0]
    if detected not in ("pcap", "records") or pcap_policy not in ("ipv4", "tcp-udp"):
        raise ValueError("unsupported input format or PCAP policy")
    if detected == "records":
        if window_mode == "time" and not timestamp_format:
            raise ValueError("time windows require timestamps: use upstream PCAP, or explicitly specify "
                             "--timestamp-format and --timestamp-offset for timestamped records. "
                             "13-byte key-only input cannot recover time; --window-mode records is legacy chunking.")
        if timestamp_format and (timestamp_format not in TIMESTAMP_FORMATS or timestamp_offset is None or
                                 timestamp_offset < 0 or timestamp_offset + 8 > record_size):
            raise ValueError("timestamp format/offset must identify an 8-byte field inside each record")
        if output_key and (key_offset < 0 or key_offset + 13 > record_size):
            raise ValueError("13-byte key does not fit the input record")
    output_size = 13 if detected == "pcap" or output_key else record_size
    if expected_record_size is not None and output_size != expected_record_size:
        raise ValueError(f"input conversion emits {output_size}-byte records but this trace requires "
                         f"{expected_record_size}-byte records; use the matching timestamped fixed-record source "
                         "or prepare_key13_dataset.py for IMC/MAWI PCAP")
    config = dict(window_mode=window_mode, window_seconds_ns=width_ns if window_mode == "time" else None,
                  window_records=window_records if window_mode == "records" else None, last_window=last_window,
                  input_format=detected, record_size=record_size if detected == "records" else None,
                  timestamp_format=timestamp_format if detected == "records" else "pcap-header",
                  timestamp_offset=timestamp_offset if detected == "records" else None,
                  output_record_size=output_size, output_key=output_key, key_offset=key_offset,
                  pcap_policy=pcap_policy if detected == "pcap" else None, max_windows=max_windows)
    source_stats = [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files]
    sources = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha256_file(p)) for p in files]
    if out_dir.exists() and any(out_dir.iterdir()):
        manifest_path = out_dir / MANIFEST
        if not manifest_path.is_file():
            raise ValueError(f"{out_dir}: existing output lacks timestamp provenance; use a fresh output directory")
        previous = json.loads(manifest_path.read_text())
        if (previous.get("schema_version") != SCHEMA_VERSION or previous.get("config") != config or
                previous.get("sources") != sources):
            raise ValueError(f"{out_dir}: source or window configuration changed; use a fresh output directory")
        expected = {w["file"] for w in previous["windows"]}
        if {p.name for p in out_dir.glob("*.dat")} != expected:
            raise ValueError(f"{out_dir}: window files differ from manifest; use a fresh output directory")
        for window in previous["windows"]:
            path = out_dir / window["file"]
            if path.stat().st_size != window["records"] * output_size or sha256_file(path) != window["sha256"]:
                raise ValueError(f"{path}: cached window failed integrity validation; use a fresh output directory")
        return previous

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    stats = Counter()
    windows = []
    origin = previous_stamp = current_index = None
    output = None
    digest = None
    count = captured = 0
    first_kept = last_kept = None
    discarded = 0
    limit_reached = False
    lookahead_records = lookahead_kept = 0

    def finish(partial):
        nonlocal output, count, captured, first_kept, last_kept, discarded
        if count == 0:
            raise ValueError(f"empty time interval dataset_{current_index:04d}; timestamps cannot be compressed. "
                             "Process contiguous capture intervals separately, or inspect the PCAP filtering policy.")
        output.close()
        output = None
        path = staging / f"dataset_{current_index:04d}.dat"
        if partial and last_window == "drop":
            discarded += count
            path.unlink()
        else:
            windows.append(dict(index=current_index, file=path.name, records=count, captured_records=captured,
                                start_ns=origin + current_index * width_ns if window_mode == "time" else None,
                                end_ns=origin + (current_index + 1) * width_ns if window_mode == "time" else None,
                                first_packet_ns=first_kept, last_packet_ns=last_kept, partial=partial,
                                sha256=digest.hexdigest()))
        count = captured = 0
        first_kept = last_kept = None

    try:
        for path in files:
            iterator = (iter_pcap(path, pcap_policy, stats) if detected == "pcap" else
                        iter_records(path, record_size, timestamp_format, timestamp_offset, output_key, key_offset, stats))
            try:
                for stamp, record in iterator:
                    if stamp is not None:
                        if window_mode == "time" and previous_stamp is not None and stamp < previous_stamp:
                            raise ValueError(f"{path}: timestamp moved backwards ({stamp} < {previous_stamp}); "
                                             "input files must be in chronological order, with no overlapping captures")
                        if origin is None:
                            origin = stamp
                        previous_stamp = stamp
                    index = ((stamp - origin) // width_ns if window_mode == "time" else
                             len(windows))
                    if current_index is None:
                        current_index = index
                    if window_mode == "time" and index != current_index:
                        finish(False)
                        if max_windows and len(windows) >= max_windows:
                            limit_reached = True
                            lookahead_records = 1
                            lookahead_kept = int(record is not None)
                            break
                        if index != current_index + 1:
                            raise ValueError(f"empty time interval dataset_{current_index + 1:04d} before index {index}; "
                                             "timestamps cannot be compressed. Process contiguous capture intervals separately.")
                        current_index = index
                    if output is None:
                        output = (staging / f"dataset_{current_index:04d}.dat").open("wb")
                        digest = hashlib.sha256()
                    captured += 1
                    if record is None:
                        continue
                    output.write(record)
                    digest.update(record)
                    count += 1
                    if first_kept is None:
                        first_kept = stamp
                    last_kept = stamp
                    if window_mode == "records" and count == window_records:
                        finish(False)
                        current_index += 1
                        if max_windows and len(windows) >= max_windows:
                            limit_reached = True
                            break
            finally:
                iterator.close()
            if limit_reached:
                break
        if output is not None:
            finish(True)
        if not windows:
            raise ValueError("no nonempty output windows; keep the final partial window or supply a longer capture")
        if source_stats != [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files]:
            raise ValueError("source changed while preparing windows; retry from a stable capture")
        manifest = dict(schema_version=SCHEMA_VERSION, config=config, sources=sources, origin_ns=origin,
                        last_observed_timestamp_ns=previous_stamp, windows=windows, stats=dict(stats),
                        discarded_records=discarded, limit_reached=limit_reached,
                        boundary_lookahead_records=lookahead_records, boundary_lookahead_kept=lookahead_kept,
                        empty_interval_policy="reject", source_order="provided order; CLI uses numeric filenames",
                        interval_convention="[origin + index * width, origin + (index + 1) * width)",
                        final_interval_policy="EOF interval is partial; capture coverage beyond last packet is unknown")
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if out_dir.exists():
            out_dir.rmdir()  # An initially empty directory must still be empty.
        staging.rename(out_dir)
        return manifest
    finally:
        if output is not None:
            output.close()
        if staging.exists():
            shutil.rmtree(staging)
