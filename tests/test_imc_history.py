import argparse
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_imc_current_oracle as imc


class IMCHistoryTests(unittest.TestCase):
    def args(self, root, replicas=1):
        with patch.object(sys, "argv", ["imc", "--out-dir", str(root / "out")]):
            args = imc.parse_args()
        args.run_root = root / "run"
        args.data_full_root = root / "data"
        args.sample_cache_dir = root / "sample"
        args.runtime_train_cache_dir = root / "training"
        args.pretrained_root = root / "prior"
        args.train_replicas = replicas
        args.end_seed = 2
        args.end_minute = 6
        return args

    def prepare_sample(self, args, minute, count=3):
        imc.write_artifacts_from_fsd(
            args.sample_cache_dir, imc.fine_id(minute), minute, [(1, count)], "imc", save_fsd=True,
        )

    def test_source_and_cli_validation(self):
        self.assertEqual(imc.training_source(1), 0)
        self.assertEqual(imc.training_source(9), 8)
        for minute in (-1, 0):
            with self.assertRaises(ValueError):
                imc.training_source(minute)
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            self.assertEqual(args.sample_shape_weight, 0)
            imc.validate_protocol(args)
            args.sample_shape_weight = 0.5
            with self.assertRaises(ValueError):
                imc.validate_protocol(args)
        with patch.object(sys, "argv", ["imc", "--out-dir", ".", "--current-replicas", "3"]):
            self.assertEqual(imc.parse_args().train_replicas, 3)
        with patch.object(sys, "argv", ["imc", "--out-dir", ".", "--sample-shape-weight", "0.5"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                imc.parse_args()

    def test_training_artifacts_use_preceding_window_for_every_replica(self):
        for replicas in (1, 3):
            with self.subTest(replicas=replicas), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self.args(root, replicas)
                self.prepare_sample(args, 4, 3)
                self.prepare_sample(args, 5, 99)
                with patch.object(imc, "load_cached_fsd", wraps=imc.load_cached_fsd) as load:
                    train_root, _ = imc.build_previous_train_root(args, 5, root)
                if replicas > 1:
                    load.assert_called_once_with(args.sample_cache_dir, 4)
                metadata = json.loads((train_root / "training.json").read_text())
                self.assertEqual(metadata["source_minute"], 4)
                for unit in metadata["train_unit_ids"]:
                    label = np.load(train_root / "1_10_chazhi" / f"{imc.fine_id(unit)}.npy")
                    np.testing.assert_array_equal(label, np.full(10, 3))
                    self.assertEqual((train_root / "fine_dat" / f"{imc.fine_id(unit)}.dat").stat().st_size, 39)

    def test_training_cache_tracks_source_and_rejects_old_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            self.prepare_sample(args, 4)
            old = args.runtime_train_cache_dir / "minute_0005_current"
            old.mkdir(parents=True)
            (old / "sentinel").write_text("unchanged")
            train_root, info = imc.build_previous_train_root(args, 5, root)
            self.assertFalse(info["used_train_cache"])
            with patch.object(imc, "train_root_complete", return_value=True):
                _, info = imc.build_previous_train_root(args, 5, root)
                self.assertTrue(info["used_train_cache"])
                self.prepare_sample(args, 4, 7)
                _, info = imc.build_previous_train_root(args, 5, root)
                self.assertFalse(info["used_train_cache"])
                self.assertEqual(np.load(train_root / "1_10_chazhi/fine_dataset_0004.npy")[0], 7)
                args.force_cache_rebuild = True
                _, info = imc.build_previous_train_root(args, 5, root)
                self.assertFalse(info["used_train_cache"])
            self.assertEqual((old / "sentinel").read_text(), "unchanged")

    def test_training_precedes_current_inference_without_current_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            self.prepare_sample(args, 4)
            raw = args.data_full_root / "imc" / "caida_1min_split"
            raw.mkdir(parents=True)
            (raw / "dataset_0005.dat").write_bytes(bytes(13 * 10))
            final = args.run_root / "run_full_matrix/64_64_imc_final"
            for head, truth in (("1_10", [[1, 10]]), ("10_1e4", [])):
                directory = final / "tr_ts" / f"{head}_real"
                directory.mkdir(parents=True)
                np.save(directory / "dataset_0005.npy", np.asarray(truth).reshape(-1, 2))
            heavy = final / "EL/5"
            heavy.mkdir(parents=True)
            (heavy / "heavy_0.csv").write_text("count\n")
            events = []

            def train(model, counter_root, train_root, label_root, minutes, head, train_args, device):
                self.assertEqual(minutes, [4])
                events.append(("train", head))
                return {}

            def counter(counter_root, minute):
                self.assertEqual(minute, 5)
                self.assertEqual(len(events), 2)
                events.append(("counter", minute))
                return torch.zeros((1, 3, 64, 64))

            with ExitStack() as stack:
                stack.enter_context(patch.object(imc, "checkpoint_for", return_value=root / "unused.pth"))
                stack.enter_context(patch.object(imc, "make_model", return_value=object()))
                stack.enter_context(patch.object(imc, "build_counter_store", return_value=0.0))
                stack.enter_context(patch.object(imc, "train_head", side_effect=train))
                stack.enter_context(patch.object(imc, "cached_counter", side_effect=counter))
                stack.enter_context(patch.object(imc, "infer_head", side_effect=[np.ones((1, 10)), np.ones((1, 1080))]))
                stack.enter_context(patch.object(imc.torch.cuda, "is_available", return_value=False))
                stack.enter_context(patch.object(imc.torch.cuda, "empty_cache"))
                row = imc.train_eval_minute(args, root, torch.device("cpu"), 5)
            self.assertEqual(events, [("train", "1_10"), ("train", "10_1e4"), ("counter", 5)])
            self.assertEqual(row["train_source_minute"], 4)
            self.assertEqual(row["minute"], 5)
            self.assertEqual(row["packet_count"], 10)
            self.assertFalse(row["sample_shape_available"])
            self.assertEqual(row["protocol"], imc.PROTOCOL)
            self.assertFalse(imc.cache_complete(args.sample_cache_dir, 5))

    def test_prepare_cache_only_includes_only_preceding_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            args.prepare_cache_only = True
            with patch.object(imc, "parse_args", return_value=args), \
                    patch.object(imc, "prepare_cache_range") as prepare, \
                    patch.object(imc, "configure") as configure:
                imc.main()
            self.assertEqual(prepare.call_args.args[2], [4, 5])
            configure.assert_not_called()

    def test_old_results_cannot_resume_as_history(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.out_dir.mkdir()
            (args.out_dir / "DONE").write_text("done\n")
            with self.assertRaises(ValueError):
                imc.load_completed_rows(args)
            (args.out_dir / "args.json").write_text(json.dumps({"protocol": "current_oracle"}))
            with self.assertRaises(ValueError):
                imc.load_completed_rows(args)
            (args.out_dir / "args.json").write_text(json.dumps({"protocol": imc.PROTOCOL}))
            path = args.out_dir / "window_metrics.csv"
            prefix = "minute,train_source_minute,strategy,protocol,sample_shape_weight\n"
            path.write_text(prefix + f"5,5,previous_window,{imc.PROTOCOL},0\n")
            with self.assertRaises(ValueError):
                imc.load_completed_rows(args)
            path.write_text(prefix + f"5,4,previous_window,{imc.PROTOCOL},0\n")
            self.assertEqual(len(imc.load_completed_rows(args)), 1)


if __name__ == "__main__":
    unittest.main()
