import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_comprehensive_compare as compare
import plot_comprehensive_single_latest_imc as plots


class ExplicitResultTests(unittest.TestCase):
    def write_results(self, path, raw=0.8, residual=0.1):
        frame = pd.DataFrame({
            "minute": [5, 6, 99], "mrd": [raw] * 3, "wmrd": [raw] * 3,
            "gate_mrd": [residual] * 3, "gate_wmrd": [residual] * 3,
        })
        frame.to_csv(path, index=False)
        return frame

    def test_only_declared_file_and_column_are_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected, other = root / "selected.csv", root / "better.csv"
            self.write_results(selected)
            self.write_results(other, 0.001, 0.001)
            with patch.object(compare.pd, "read_csv", wraps=pd.read_csv) as read:
                result, metric, path = compare.load_ours("imc", selected, "raw")
            read.assert_called_once_with(selected.resolve())
            self.assertEqual(metric, "raw")
            self.assertEqual(path, selected.resolve())
            np.testing.assert_allclose(result["wmrd"], [0.8] * 3)
            self.write_results(selected, 0.8, 0.00001)
            unchanged, _, _ = compare.load_ours("imc", selected, "raw")
            pd.testing.assert_frame_equal(result, unchanged)
            gated, metric, _ = compare.load_ours("imc", selected, "residual")
            self.assertEqual(metric, "residual")
            np.testing.assert_allclose(gated["wmrd"], [0.00001] * 3)

    def test_missing_or_invalid_selected_data_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.csv"
            frame = self.write_results(path)
            with self.assertRaises(ValueError):
                compare.load_ours("imc", path, "unknown")
            with self.assertRaises(ValueError):
                compare.load_ours("imc", path, "twoway")
            with self.assertRaises(FileNotFoundError):
                compare.load_ours("imc", path.with_name("missing.csv"), "raw")
            for field, values in (("wmrd", [np.nan, 0.2, 0.3]),
                                  ("wmrd", [np.inf, 0.2, 0.3]),
                                  ("minute", [5, 5, 6]),
                                  ("minute", [5, 6.5, 7])):
                invalid = frame.copy()
                invalid[field] = values
                invalid.to_csv(path, index=False)
                with self.subTest(field=field, values=values), self.assertRaises(ValueError):
                    compare.load_ours("imc", path, "raw")

    def test_trace_mapping_requires_one_explicit_file_per_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = []
            for trace in compare.DATASETS:
                path = root / f"{trace}.csv"
                self.write_results(path)
                items.append(f"{trace}={path}")
            mapping = compare.parse_ours_results(items)
            self.assertEqual(set(mapping), set(compare.DATASETS))
            alias_items = [item.replace("caida2016=", "caida_2016=") for item in items]
            self.assertEqual(compare.parse_ours_results(alias_items), mapping)
            for bad in (items[:-1], items + [items[0]], items + ["wrong=x"], ["imc"]):
                with self.assertRaises(ValueError):
                    compare.parse_ours_results(bad)

    def test_both_comparison_entrypoints_require_explicit_metric_and_paths(self):
        parser = argparse.ArgumentParser()
        compare.add_ours_arguments(parser)
        for arguments in ([], ["--ours-results", "imc=x"], ["--ours-metric", "raw"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)
        parsed = parser.parse_args(["--ours-results", "imc=x", "--ours-metric", "raw"])
        self.assertEqual(parsed.ours_metric, "raw")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected.csv"
            self.write_results(path)
            paths = {trace: path for trace in compare.DATASETS}
            with patch.object(compare, "load_ours", wraps=compare.load_ours) as load:
                results, chosen = plots.load_ours_results(paths, "raw")
            self.assertEqual(load.call_count, len(paths))
            for trace in paths:
                np.testing.assert_allclose(results[trace]["wmrd"], [0.8] * 3)
                self.assertEqual(chosen[trace], ("raw", path.resolve()))


if __name__ == "__main__":
    unittest.main()
