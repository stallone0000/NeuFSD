"""CPU-only checks for timestamp boundaries, PCAP conversion, and cache provenance."""
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from timestamp_windows import extract_key13, prepare_windows, source_files, timestamp_ns


def ipv4(protocol=17, *, options=False, fragment=0, truncated=False):
    ihl = 24 if options else 20
    ip = bytearray(ihl)
    ip[0] = 0x40 + ihl // 4
    ip[2:4] = (1500).to_bytes(2, "big")
    ip[6:8] = fragment.to_bytes(2, "big")
    ip[9] = protocol
    ip[12:16] = bytes((1, 2, 3, 4))
    ip[16:20] = bytes((5, 6, 7, 8))
    transport = struct.pack("!HH", 1234, 4321) + bytes(16 if protocol == 6 else 4)
    return bytes(ip) + (transport[:3] if truncated else transport)


KEY = bytes((1, 2, 3, 4)) + struct.pack("!H", 1234) + bytes((5, 6, 7, 8)) + struct.pack("!H", 4321) + b"\x11"


def pcap(records, endian="<", nano=False, linktype=1, vlan=False):
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    out = bytearray(struct.pack(endian + "IHHiIII", magic, 2, 4, 0, 0, 65535, linktype))
    for stamp, packet in records:
        if linktype == 1:
            packet = bytes(12) + (b"\x81\x00\x00\x01\x08\x00" if vlan else b"\x08\x00") + packet
        seconds, nanoseconds = divmod(stamp, 1_000_000_000)
        fraction = nanoseconds if nano else nanoseconds // 1000
        out.extend(struct.pack(endian + "IIII", seconds, fraction, len(packet), 1518))
        out.extend(packet)
    return bytes(out)


class TimestampWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def records(self, name="source.dat", stamps=(100.25, 101.0, 160.25, 179.0, 220.25)):
        path = self.root / name
        path.write_bytes(b"".join(KEY + struct.pack("<d", value) for value in stamps))
        return path

    def split(self, paths, out="windows", **kwargs):
        options = dict(record_size=21, timestamp_format="float64-le-seconds", timestamp_offset=13)
        options.update(kwargs)
        return prepare_windows(paths, self.root / out, **options)

    def test_actual_boundaries_variable_counts_and_partial(self):
        source = self.records()
        manifest = self.split([source])
        self.assertEqual(manifest["origin_ns"], 100_250_000_000)
        self.assertEqual([w["records"] for w in manifest["windows"]], [2, 2, 1])
        self.assertEqual([w["partial"] for w in manifest["windows"]], [False, False, True])
        self.assertEqual([w["start_ns"] for w in manifest["windows"]], [100_250_000_000, 160_250_000_000, 220_250_000_000])
        joined = b"".join((self.root / "windows" / w["file"]).read_bytes() for w in manifest["windows"])
        self.assertEqual(joined, source.read_bytes())
        dropped = self.split([source], out="dropped", last_window="drop")
        self.assertEqual([w["records"] for w in dropped["windows"]], [2, 2])
        self.assertEqual(dropped["discarded_records"], 1)

    def test_file_boundaries_do_not_reset_origin(self):
        source = self.root / "parts"
        source.mkdir()
        self.records("parts/part10.dat", stamps=(160.25, 161.0))
        self.records("parts/part2.dat", stamps=(100.25, 150.0))
        paths = source_files(source)
        self.assertEqual([p.name for p in paths], ["part2.dat", "part10.dat"])
        manifest = self.split(paths)
        self.assertEqual([w["records"] for w in manifest["windows"]], [2, 2])

    def test_all_classic_pcap_formats_linktypes_and_snaplen(self):
        for endian in ("<", ">"):
            for nano in (False, True):
                for linktype in (1, 101):
                    with self.subTest(endian=endian, nano=nano, linktype=linktype):
                        source = self.root / "input.pcap"
                        origin = 1_000_000_001 if nano else 1_000_001_000
                        source.write_bytes(pcap([(origin, ipv4()), (origin + 60_000_000_000, ipv4())],
                                                endian, nano, linktype))
                        manifest = prepare_windows([source], self.root / f"out{endian}{nano}{linktype}")
                        self.assertEqual(manifest["origin_ns"], origin)
                        self.assertEqual([w["records"] for w in manifest["windows"]], [1, 1])
                        self.assertEqual((self.root / f"out{endian}{nano}{linktype}" / "dataset_0000.dat").read_bytes(), KEY)

    def test_gzip_vlan_ip_options_and_equal_timestamps(self):
        source = self.root / "input.pcap.gz"
        source.write_bytes(gzip.compress(pcap([(1_000_000_000, ipv4(options=True))] * 2, vlan=True)))
        manifest = prepare_windows([source], self.root / "out")
        self.assertEqual(manifest["windows"][0]["records"], 2)
        self.assertEqual((self.root / "out/dataset_0000.dat").read_bytes(), KEY * 2)

    def test_filter_policy_and_first_captured_origin(self):
        source = self.root / "input.pcap"
        source.write_bytes(pcap([(5_000_000_000, ipv4(protocol=1)), (6_000_000_000, ipv4()),
                                (65_000_000_000, ipv4(fragment=0x2000))]))
        manifest = prepare_windows([source], self.root / "udp", pcap_policy="tcp-udp")
        self.assertEqual(manifest["origin_ns"], 5_000_000_000)
        self.assertEqual([w["records"] for w in manifest["windows"]], [1, 1])
        self.assertEqual(manifest["stats"]["non_tcp_udp"], 1)
        self.assertEqual(manifest["windows"][0]["captured_records"], 2)
        all_ip = prepare_windows([source], self.root / "all", pcap_policy="ipv4")
        self.assertEqual([w["records"] for w in all_ip["windows"]], [2, 1])
        self.assertEqual((self.root / "all/dataset_0000.dat").read_bytes()[:13],
                         bytes((1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 1)))
        self.assertEqual(extract_key13(ipv4(fragment=1), 101, "ipv4")[1], "noninitial_fragment")
        self.assertEqual(extract_key13(ipv4(truncated=True), 101, "ipv4")[1], "short_transport")

    def test_time_requires_timestamp_and_caida2016_format_is_explicit(self):
        path = self.root / "keys.dat"
        path.write_bytes(KEY * 3)
        with self.assertRaisesRegex(ValueError, "key-only"):
            prepare_windows([path], self.root / "out")
        source = self.root / "caida2016"
        source.mkdir()
        (source / "trace.dat").write_bytes(struct.pack("<d", 100) + b"12345678")
        proc = subprocess.run([sys.executable, str(SCRIPTS / "split_raw.py"), "caida_2016", str(source),
                               str(self.root / "missing")], text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("timestamp-format", proc.stderr)
        subprocess.run([sys.executable, str(SCRIPTS / "split_raw.py"), "caida_2016", str(source),
                        str(self.root / "explicit"), "--timestamp-format", "float64-le-seconds"],
                       check=True, stdout=subprocess.DEVNULL)

    def test_gap_reversal_and_truncated_input_are_transactional(self):
        for stamps, error in (((100.0, 220.0), "empty time interval"), ((100.0, 99.0), "backwards")):
            with self.subTest(stamps=stamps):
                source = self.records(stamps=stamps)
                with self.assertRaisesRegex(ValueError, error):
                    self.split([source])
                self.assertFalse((self.root / "windows").exists())
                self.assertFalse(list(self.root.glob(".windows.staging-*")))
        source = self.records()
        source.write_bytes(source.read_bytes() + b"bad")
        with self.assertRaisesRegex(ValueError, "trailing"):
            self.split([source])
        self.assertFalse((self.root / "windows").exists())

    def test_filtered_empty_interval_is_not_compressed(self):
        path = self.root / "capture.pcap"
        path.write_bytes(pcap([(0, ipv4(protocol=1)), (60_000_000_000, ipv4())]))
        with self.assertRaisesRegex(ValueError, "empty time interval dataset_0000"):
            prepare_windows([path], self.root / "windows")
        self.assertFalse((self.root / "windows").exists())

    def test_bad_pcap_headers_and_timestamp_rejected(self):
        valid = pcap([(0, ipv4())])
        variants = (valid[:-1], valid[:24] + b"short", pcap([(0, ipv4())], linktype=113),
                    valid[:28] + struct.pack("<I", 1_000_000) + valid[32:])
        for data in variants:
            path = self.root / "bad.pcap"
            path.write_bytes(data)
            with self.subTest(data=data[:40]), self.assertRaises(ValueError):
                prepare_windows([path], self.root / "out")
            self.assertFalse((self.root / "out").exists())

    def test_cache_reuse_rejects_changed_source_config_or_output(self):
        source = self.records()
        original = self.split([source])
        self.assertEqual(original, self.split([source]))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.split([source], last_window="drop")
        data = source.read_bytes()
        source.write_bytes(b"X" + data[1:])
        with self.assertRaisesRegex(ValueError, "source or window configuration changed"):
            self.split([source])
        source.write_bytes(data)
        output = self.root / "windows/dataset_0000.dat"
        content = output.read_bytes()
        output.write_bytes(b"Y" + content[1:])
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.split([source])
        self.assertEqual(output.read_bytes(), b"Y" + content[1:])

    def test_existing_output_without_manifest_is_preserved(self):
        source = self.records()
        out = self.root / "windows"
        out.mkdir()
        (out / "unrelated.txt").write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "lacks timestamp provenance"):
            self.split([source])
        self.assertEqual((out / "unrelated.txt").read_text(), "preserve me")

    def test_explicit_record_count_mode_is_not_named_time(self):
        source = self.root / "keys.dat"
        source.write_bytes(KEY * 5)
        manifest = prepare_windows([source], self.root / "out", window_mode="records", window_records=2)
        self.assertEqual(manifest["config"]["window_mode"], "records")
        self.assertIsNone(manifest["origin_ns"])
        self.assertEqual([w["records"] for w in manifest["windows"]], [2, 2, 1])

    def test_max_windows_preserves_complete_intervals(self):
        source = self.records()
        manifest = self.split([source], max_windows=1)
        self.assertEqual(len(manifest["windows"]), 1)
        self.assertEqual(manifest["windows"][0]["records"], 2)
        self.assertFalse(manifest["windows"][0]["partial"])
        self.assertTrue(manifest["limit_reached"])
        self.assertEqual(manifest["stats"]["kept"], 2 + manifest["boundary_lookahead_kept"])

    def test_split_raw_enforces_trace_output_layout(self):
        source = self.root / "capture.pcap"
        source.write_bytes(pcap([(0, ipv4())]))
        for trace in ("caida_2016", "caida_2018"):
            proc = subprocess.run([sys.executable, str(SCRIPTS / "split_raw.py"), trace, str(source),
                                   str(self.root / trace)], text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("this trace requires", proc.stderr)
            self.assertFalse((self.root / trace).exists())
        records = self.records()
        subprocess.run([sys.executable, str(SCRIPTS / "split_raw.py"), "imc", str(records),
                        str(self.root / "imc"), "--packet-size", "21", "--timestamp-format", "float64-le-seconds",
                        "--timestamp-offset", "13"], check=True, stdout=subprocess.DEVNULL)
        self.assertEqual((self.root / "imc/dataset_0000.dat").read_bytes(), KEY * 2)

    def test_caida2018_default_timestamps_and_opaque_legacy_records(self):
        source = self.records()
        command = [sys.executable, str(SCRIPTS / "split_raw.py"), "caida_2018", str(source)]
        subprocess.run(command + [str(self.root / "time")], check=True, stdout=subprocess.DEVNULL)
        manifest = json.loads((self.root / "time/window_manifest.json").read_text())
        self.assertEqual([w["records"] for w in manifest["windows"]], [2, 2, 1])
        source.write_bytes((KEY + b"\xff" * 8) * 5)
        subprocess.run(command + [str(self.root / "legacy"), "--window-mode", "records", "--window-records", "2"],
                       check=True, stdout=subprocess.DEVNULL)
        self.assertEqual((self.root / "legacy/dataset_0000.dat").stat().st_size, 42)

    def test_epoch_precision_and_explicit_timestamp_encodings(self):
        value = 1_520_000_000.123456
        stamp = timestamp_ns(struct.pack("<d", value), 0, "float64-le-seconds")
        self.assertEqual(stamp // 1_000_000_000, 1_520_000_000)
        for endian, symbol in (("le", "<"), ("be", ">")):
            self.assertEqual(timestamp_ns(struct.pack(symbol + "d", 2.5), 0, f"float64-{endian}-seconds"), 2_500_000_000)
            for unit, multiplier in (("seconds", 1_000_000_000), ("microseconds", 1000), ("nanoseconds", 1)):
                self.assertEqual(timestamp_ns(struct.pack(symbol + "Q", 7), 0, f"uint64-{endian}-{unit}"), 7 * multiplier)
        for invalid in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                timestamp_ns(struct.pack("<d", invalid), 0, "float64-le-seconds")

    def test_key13_cli_labels_use_output_layout_and_variable_packet_mass(self):
        source = self.records()
        cmd = [sys.executable, str(SCRIPTS / "prepare_key13_dataset.py"), "--trace", "imc",
               "--source", str(source), "--packet-size", "21", "--timestamp-format", "float64-le-seconds",
               "--timestamp-offset", "13", "--workers", "1", "--data-full-root", str(self.root / "data"),
               "--run-root", str(self.root / "runs")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        trace = self.root / "data/imc"
        for index, expected in enumerate((2, 2, 1)):
            labels = np.load(trace / "tr_ts/full_real" / f"dataset_{index:04d}.npy")
            self.assertEqual(int((labels[:, 0] * labels[:, 1]).sum()), expected)
            self.assertEqual((trace / "caida_1min_split" / f"dataset_{index:04d}.dat").stat().st_size, expected * 13)
        manifest = json.loads((trace / "prepare_key13_manifest.json").read_text())
        self.assertEqual(manifest["records_per_window"], [2, 2, 1])
        self.assertEqual(manifest["packet_size"], 13)


if __name__ == "__main__":
    unittest.main()
