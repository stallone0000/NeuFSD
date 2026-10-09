"""Generated-input checks; no downloaded traces, checkpoints, or GPU required."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


def load_module(path):
    spec = importlib.util.spec_from_file_location('test_target', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    def test_record_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            for res in ('64_64', '128_128'):
                for trace in ('caida_2016', 'caida_2018', 'caida_2018_new'):
                    with self.subTest(res=res, trace=trace):
                        old_format = trace == 'caida_2016'
                        keys = [(i % 251).to_bytes(8 if old_format else 13, 'big')
                                for i in range(1000)]
                        records = [(b'prefix00' + key if old_format else key + b'suffix00')
                                   for key in keys]
                        path = Path(temp) / 'packets.dat'
                        path.write_bytes(b''.join(records))
                        parser_name = 'parse_file_2016.py' if old_format else 'parse_file_2018.py'
                        module = load_module(EXPERIMENTS / 'configs' / f'{res}_{trace}' / parser_name)
                        self.assertEqual(module.parse_data_file_c_style(str(path)), Counter(keys))

    def test_synthetic_counter_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            subprocess.run([
                sys.executable, str(EXPERIMENTS / 'scripts/generate_zipf_pretrain.py'),
                '--out-root', str(work / 'data'), '--formats', 'caida2016', 'caida2018',
                '--flows', '64', '--alphas', '0', '1', '--packets', '1000',
            ], check=True, stdout=subprocess.DEVNULL)
            for fmt in ('caida2016', 'caida2018'):
                labels = work / 'data' / fmt / 'tr_ts'
                subprocess.run([
                    sys.executable, str(EXPERIMENTS / 'scripts/build_counter_store.py'),
                    '--input-dir', str(work / 'data' / fmt / 'caida_1min_split'),
                    '--out-root', str(labels), '--trace-format', fmt,
                    '--counter-len', '4096', '--start-seed', '0', '--end-seed', '2',
                    '--gen-bin', str(work / 'gen'),
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**os.environ, 'OMP_NUM_THREADS': '2'})
                index = json.loads((labels / 'input_store/index.json').read_text())
                self.assertEqual(index['shape'], [4, 4096])
                counters = np.fromfile(labels / 'input_store/counters.u4', dtype='<u4').reshape(4, 4096)
                np.testing.assert_array_equal(counters.sum(axis=1), [1000] * 4)
                for dataset in index['datasets']:
                    self.assertEqual(np.load(labels / '1_10_chazhi' / f'{dataset}.npy').shape, (10,))
                    self.assertEqual(np.load(labels / '10_1e4_chazhi' / f'{dataset}.npy').shape, (1080,))

    def test_model_shape_and_parameters(self):
        import torch
        sys.path.insert(0, str(EXPERIMENTS / 'local_deps'))
        for res, side, counts in (
            ('64_64', 64, (86440962, 87512032)),
            ('128_128', 128, (86477826, 87548896)),
        ):
            module = load_module(EXPERIMENTS / 'configs' / f'{res}_caida_org/model.py')
            for dim, expected in zip((10, 1080), counts):
                with self.subTest(res=res, output=dim), torch.device('meta'):
                    model = module.CustomViT(dim)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                    self.assertEqual(tuple(model(torch.zeros(1, 3, side, side)).shape), (1, dim))

    def test_collector_reads_generated_results(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            run = work / 'run'
            summary = run / '64_64_caida_2018_final/plots/pipeline_eval/summary_metrics.csv'
            summary.parent.mkdir(parents=True)
            summary.write_text('dataset_id,mrd_avg,wmrd_avg\nOVERALL_AVG,0.25,0.125\n')
            timing = run / '64_64_caida_2018_infer_time.csv'
            timing.write_text('head,ms_per_sample\n1_10,1.0\n10_1e4,2.0\ntotal_two_heads,3.0\n')
            subprocess.run([
                sys.executable, str(EXPERIMENTS / 'compare/collect_compare.py'), 'caida_2018',
            ], cwd=work, env={**os.environ, 'RUN_ROOT': str(run)},
                check=True, stdout=subprocess.DEVNULL)
            for filename, expected in (
                ('com_error_mrd.csv', '0.25'), ('com_error_wmrd.csv', '0.125'),
                ('decode_time.csv', '3.0'), ('insert_time.csv', ''),
            ):
                with (work / filename).open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]['Ours'], expected)
                self.assertTrue(all(row['Ours'] == '' for row in rows[1:]))


if __name__ == '__main__':
    unittest.main()
