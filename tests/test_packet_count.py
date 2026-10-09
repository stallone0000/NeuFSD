"""CPU-only checks for per-window packet mass and comparison throughput."""
from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pandas as pd
import torch

import packet_counts
import perf_eval_sweeps_and_plots as perf
import run_comprehensive_compare as compare
import simulate_realtime_online as realtime
import train_online


class PacketCountTests(unittest.TestCase):
    def test_formats_count_each_window_and_respect_override(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            for trace, size in [("caida_2016", 16), ("caida_2018", 21), ("caida_2018_new", 21),
                                ("caida_org", 13), ("imc", 13), ("mawi", 13), ("key13", 13)]:
                with self.subTest(trace=trace):
                    for minute, count in enumerate([3, 7, 0]):
                        (raw / f"dataset_{minute:04d}.dat").write_bytes(bytes(size * count))
                        self.assertEqual(packet_counts.window_packet_count(raw, minute, trace), count)
            (raw / "dataset_0000.dat").write_bytes(bytes(9 * 5))
            self.assertEqual(packet_counts.window_packet_count(raw, 0, "custom", 9), 5)
            with self.assertRaises(ValueError):
                packet_counts.window_packet_count(raw, 0, "custom")

    def test_missing_misaligned_and_invalid_record_sizes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.dat"
            with self.assertRaises(FileNotFoundError):
                packet_counts.count_raw_packets(path, 13)
            path.write_bytes(bytes(14))
            with self.assertRaisesRegex(ValueError, "not divisible"):
                packet_counts.count_raw_packets(path, 13)
            for invalid in [0, -1, 1.5, True]:
                with self.subTest(size=invalid), self.assertRaises(ValueError):
                    packet_counts.count_raw_packets(path, invalid)
            with self.assertRaisesRegex(ValueError, "not divisible"):
                realtime.parse_packet_keys(path, 13, 0, 13)

    def test_parser_custom_format_and_existing_defaults(self):
        defaults = Namespace(trace="caida_2016", packet_size=0, key_offset=-1, key_length=0)
        realtime.resolve_packet_format(defaults)
        self.assertEqual((defaults.packet_size, defaults.key_offset, defaults.key_length), (16, 8, 8))
        explicit = Namespace(trace="custom", packet_size=9, key_offset=2, key_length=7)
        realtime.resolve_packet_format(explicit)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom.dat"
            path.write_bytes(b"xx1234567" * 3)
            self.assertEqual(realtime.parse_packet_keys(path, 9, 2, 7), {b"1234567": 3})

    def make_windows(self, root):
        raw = root / "raw"
        raw.mkdir()
        for minute, light_count in enumerate([30, 60]):
            name = f"dataset_{minute:04d}"
            (raw / f"{name}.dat").write_bytes(b"H" * (12000 * 13) + b"L" * (light_count * 13))
            for folder, value in [("1_10_real", np.array([[1, 10], [2, 10]])),
                                  ("10_1e4_real", np.empty((0, 2)))]:
                dst = root / "tr_ts" / folder
                dst.mkdir(parents=True, exist_ok=True)
                np.save(dst / f"{name}.npy", value)
            heavy = root / "EL" / str(minute)
            heavy.mkdir(parents=True)
            (heavy / "heavy_0.csv").write_text("count\n12000\n")
        pred_1 = np.zeros((1, 10))
        pred_1[0, :2] = 10
        return raw, pred_1, np.zeros((1, 1080))

    def test_final_metrics_use_raw_total_despite_truncated_fsd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, pred_1, pred_2 = self.make_windows(root)
            args = Namespace(gate_threshold=1.02, gate_cap=1.2, adaptive_gate=False)
            history = {}
            for minute, residual in enumerate([30, 60]):
                count = packet_counts.window_packet_count(raw, minute, "imc")
                row = realtime.final_metrics(root, minute, pred_1, pred_2, args, history, packet_count=count)
                self.assertEqual(row["packet_count"], 12000 + residual)
                self.assertEqual(row["residual_light_mass"], residual)
                self.assertAlmostEqual(row["heavy_frac"], 12000 / count)
                self.assertAlmostEqual(row["light_frac"], residual / count)
                self.assertAlmostEqual(row["raw_scale_mean"], residual / 30)
            self.assertEqual(len(history["heavy_frac_history"]), 2)
            with self.assertRaises(TypeError):
                realtime.final_metrics(root, 0, pred_1, pred_2, args, {})
            for invalid in [-1, float("nan"), float("inf"), 12.5]:
                with self.subTest(count=invalid), self.assertRaises(ValueError):
                    realtime.final_metrics(root, 0, pred_1, pred_2, args, {}, packet_count=invalid)

    def test_perf_gate_uses_each_raw_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, pred_1, pred_2 = self.make_windows(root)
            for minute, residual in enumerate([30, 60]):
                count = packet_counts.window_packet_count(raw, minute, "imc")
                row = perf.eval_final_candidates(root, root / "EL", minute, pred_1, pred_2, 100, "imc", packet_count=count)
                self.assertEqual(row["packet_count"], count)
                self.assertEqual(row["residual_light_mass"], residual)
                self.assertAlmostEqual(row["heavy_frac"], 12000 / count)
                self.assertAlmostEqual(row["raw_scale_mean"], residual / 30)

    def test_training_correction_counts_original_not_sampled_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, pred_1, pred_2 = self.make_windows(root)
            for minute in [0, 1]:
                for folder, values in [("1_10_chazhi", pred_1), ("10_1e4_chazhi", pred_2)]:
                    dst = root / folder
                    dst.mkdir(exist_ok=True)
                    np.save(dst / f"fine_dataset_{minute:04d}.npy", values)
            args = Namespace(label_light_mass_correction="up-only", raw_dir=raw, trace="imc", packet_size=0,
                             packet_count=None, heavy_dir=root / "EL", label_light_mass_threshold=1.05,
                             label_light_mass_cap=1.25)
            labels = torch.ones((1, 10))
            unchanged = train_online.maybe_correct_label(labels, root, "fine_dataset_0000", "1_10", args)
            corrected = train_online.maybe_correct_label(labels, root, "fine_dataset_0001", "1_10", args)
            torch.testing.assert_close(unchanged, labels)
            torch.testing.assert_close(corrected, labels * 1.25)
            args.packet_count = 1_000_000
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_online.maybe_correct_label(labels, root, "fine_dataset_0001", "1_10", args)
            args.raw_dir = None
            with self.assertRaisesRegex(ValueError, "raw-dir"):
                train_online.maybe_correct_label(labels, root, "fine_dataset_0001", "1_10", args)
            args.label_light_mass_correction = "none"
            self.assertIs(train_online.maybe_correct_label(labels, root, "fine_dataset_0001", "1_10", args), labels)

    def test_perf_explicit_raw_root_has_no_fallback(self):
        args = Namespace(data_full_root=Path("/missing/explicit/raw"), run_root=Path("unused"))
        with self.assertRaises(FileNotFoundError):
            perf.raw_window_path(args, "imc", 0)

    def test_canonical_cache_tracks_variable_counts_and_validates_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "dataset_0000.dat"
            source.write_bytes(bytes(16 * 7))
            dst = root / "canonical" / "test" / "0.dat"
            dst.parent.mkdir(parents=True)
            dst.write_bytes(bytes(13 * 7))
            meta = {"data": raw, "record_size": 16, "key_offset": 8, "key_len": 8}
            with patch.dict(compare.DATASETS, {"test": meta}), patch.object(compare.subprocess, "run") as run:
                task = ("test", 0, str(root), "canonicalizer")
                self.assertEqual(compare.prepare_one_canonical(task)[2], "cached")
                run.assert_not_called()
                source.write_bytes(bytes(16 * 3))
                run.side_effect = lambda cmd, **kwargs: Path(cmd[2]).write_bytes(bytes(13 * 3))
                self.assertEqual(compare.prepare_one_canonical(task)[2], "built")
                self.assertEqual(dst.stat().st_size, 13 * 3)
                source.write_bytes(bytes(16 * 4))
                with self.assertRaisesRegex(ValueError, "must contain"):
                    compare.prepare_one_canonical(task)

    def test_stale_result_packet_total_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset_0000.dat").write_bytes(bytes(13 * 7))
            result = root / "result.json"
            result.write_text(json.dumps({"packets": 1_000_000}))
            with patch.dict(compare.DATASETS, {"test": {"data": root, "record_size": 13}}):
                self.assertIsNone(compare.cached_baseline_result(result, "test", 0))
                result.write_text(json.dumps({"packets": 7}))
                self.assertEqual(compare.cached_baseline_result(result, "test", 0)["packets"], 7)

    def test_throughput_uses_packets_and_milliseconds_per_row(self):
        rows = pd.DataFrame({"packets": [1000, 6000, 2000, 0], "insert_ms": [2, 3, 0, 5],
                             "trace": ["imc"] * 4, "algorithm": ["elastic"] * 4,
                             "memory_kb": [16] * 4, "minute": [0, 1, 2, 3], "status": ["ok"] * 4})
        np.testing.assert_allclose(compare.insert_speed_mpps(rows), [0.5, 2.0, np.nan, 0], equal_nan=True)
        np.testing.assert_allclose(compare.ours_insert_speed_from_elastic(rows, "imc", [0, 1]), [0.5, 2.0])

    def test_training_cli_default_has_no_million_packet_fallback(self):
        with patch.object(sys, "argv", ["train_online.py", "--config-dir", ".", "--exp-dir", ".",
                                        "--head", "1_10", "--mode", "continuous"]):
            args = train_online.parse_args()
        self.assertIsNone(args.packet_count)
        self.assertEqual(args.label_light_mass_correction, "none")


if __name__ == "__main__":
    unittest.main()
