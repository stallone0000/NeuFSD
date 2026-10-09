import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))


class ResidualPlotTests(unittest.TestCase):
    def test_packet_mass_includes_flows_outside_truth_support(self):
        with tempfile.TemporaryDirectory() as temp:
            final = Path(temp)
            dataset = "dataset_0005"
            block = "dataset_0000"
            for head, dim in (("1_10", 10), ("10_1e4", 1080)):
                directory = (final / "0_finetuned_results" / f"ViT_{head}_results_1e-2"
                             / f"finetuned_block_{block}" / "test_results" / dataset)
                directory.mkdir(parents=True)
                prediction = np.zeros((1, dim))
                if head == "1_10":
                    prediction[0, 0] = 10
                np.save(directory / "preds.npy", prediction)
                truth_dir = final / "tr_ts" / f"{head}_real"
                truth_dir.mkdir(parents=True)
                truth = np.array([[1.0, 20.0]]) if head == "1_10" else np.empty((0, 2))
                np.save(truth_dir / f"{dataset}.npy", truth)
            heavy = final / "EL" / "5"
            heavy.mkdir(parents=True)
            (heavy / "heavy_0.csv").write_text("count\n11000\n")

            summary = importlib.import_module("summarize_pipeline_eval")
            with self.assertRaises(ValueError):
                summary.compute_dataset(final, dataset, block, light_mass_correction="up-only")
            scores = summary.compute_dataset(
                final, dataset, block, light_mass_correction="up-only",
                light_mass_cap=10, packet_count=11020,
            )
            self.assertEqual(scores, (0.0, 0.0))

            optimization = importlib.import_module("plot_optimization_curves")
            result = optimization.compute_dataset(final, dataset, block, 1.02, 10, packet_count=11020)
            self.assertEqual(result["scale"], 2.0)
            self.assertEqual(result["gated_wmrd"], 0.0)

            full = importlib.import_module("plot_lightmass_gate_full_curves")
            result = full.compute_gate_dataset(final, dataset, block, 1.02, 10, 100, packet_count=11020)
            self.assertEqual(result["raw_scale_mean"], 2.0)
            self.assertEqual(result["gate_wmrd"], 0.0)

            sweep = importlib.import_module("sweep_lightmass_gate")
            rows = sweep.dataset_rows(final, dataset, block, packet_count=11020)
            gated = next(row for row in rows if row["variant"] == "uniform_t1.02_c1.2")
            self.assertEqual(gated["wmrd"], 0.5)


if __name__ == "__main__":
    unittest.main()
