#!/usr/bin/env python3
"""Split captures by actual timestamps; fixed-record chunking is explicitly optional."""
import argparse
from pathlib import Path

from timestamp_windows import add_window_arguments, numeric_key, prepare_windows, source_files


RECSIZE = {"caida_2018": 21, "caida_2018_new": 21, "caida_2016": 16,
           "caida_org": 13, "imc": 13, "mawi": 13}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", choices=RECSIZE)
    parser.add_argument("in_dir", type=Path, help="source file or directory, processed in numeric filename order")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--packet-size", type=int)
    parser.add_argument("--max-windows", type=int, default=0)
    add_window_arguments(parser)
    args = parser.parse_args()
    timestamp_format = args.timestamp_format
    timestamp_offset = args.timestamp_offset
    if args.trace in ("caida_2018", "caida_2018_new"):
        if args.window_mode == "time":
            timestamp_format = timestamp_format or "float64-le-seconds"
        timestamp_offset = 13 if timestamp_offset is None else timestamp_offset
    elif args.trace == "caida_2016":
        timestamp_offset = 0 if timestamp_offset is None else timestamp_offset
    try:
        manifest = prepare_windows(
            source_files(args.in_dir), args.out_dir, window_mode=args.window_mode,
            window_seconds=args.window_seconds, window_records=args.window_records,
            last_window=args.last_window, input_format=args.input_format,
            record_size=args.packet_size or RECSIZE[args.trace], timestamp_format=timestamp_format,
            timestamp_offset=timestamp_offset, pcap_policy=args.pcap_policy or
            ("ipv4" if args.trace == "imc" else "tcp-udp"), max_windows=args.max_windows,
            output_key=args.trace in ("imc", "mawi"), expected_record_size=RECSIZE[args.trace],
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    unit = f"{args.window_seconds}s timestamp" if args.window_mode == "time" else f"{args.window_records}-record"
    print(f"[split_raw] {len(manifest['windows'])} {unit} windows -> {args.out_dir}; "
          f"records={sum(w['records'] for w in manifest['windows'])}; "
          f"provenance={args.out_dir / 'window_manifest.json'}")


if __name__ == "__main__":
    main()
