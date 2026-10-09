import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


class RuntimePathTests(unittest.TestCase):
    def test_comparison_uses_observed_packet_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            for method, packets, duration in [('elastic', 250000, 5), ('mrac', 2400000, 20)]:
                (work / f'{method}_16384.txt').write_text(
                    f'Total packets: {packets}, Total flows: 10\n'
                    f'MRD: 0.1\nWMRD: 0.2\nInsertion time: {duration} ms\n'
                    'Distribution calculation (decode) time: 2 ms\n'
                )
            for method, packets, duration in [('array', 250000, 10), ('hash', 2400000, 40)]:
                (work / f'16_{method}_results.csv').write_text(
                    'packets number,memory,method,WMRD,ARE,Insert time\n'
                    f'{packets},16384,{method},0.2,0.1,{duration}\n'
                )
            (work / 'elastic_32768.txt').write_text('Insertion time: 5 ms\n')
            (work / 'mrac_32768.txt').write_text('Total packets: 200\nInsertion time: 0 ms\n')
            subprocess.run(
                [sys.executable, str(EXPERIMENTS / 'compare/collect_compare.py'), 'caida_2018'],
                cwd=work, env={**os.environ, 'RUN_ROOT': str(work / 'run')},
                check=True, stdout=subprocess.DEVNULL,
            )
            with (work / 'throughput_mpps.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            for method, expected in [('elastic', 50), ('mrac', 120), ('array', 25), ('hash', 60)]:
                self.assertEqual(float(rows[0][method]), expected)
            self.assertEqual(rows[0]['Ours'], '')
            self.assertTrue(all(row[method] == '' for row in rows[1:]
                                for method in ('elastic', 'mrac', 'array', 'hash', 'Ours')))

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
            frame = pd.read_csv(work / 'throughput_mpps.csv')
            for trace in ('caida_2016', 'caida_2018', 'caida_2018_new', 'caida_org'):
                with self.subTest(trace=trace):
                    spec = importlib.util.spec_from_file_location(
                        f'plot_{trace}', EXPERIMENTS / 'compare' / trace / 'plot_compare.py')
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    try:
                        with mock.patch.object(module.pd, 'read_csv', return_value=frame) as reader, \
                                mock.patch.object(plt, 'savefig'), mock.patch.object(plt, 'close'):
                            module.paint_MPPS()
                            reader.assert_called_once_with('./throughput_mpps.csv')
                            np.testing.assert_array_equal(
                                [line.get_ydata()[0] for line in plt.gca().lines[:4]],
                                [50, 120, 25, 60],
                            )
                    finally:
                        plt.close('all')

    def test_four_mode_pretrain_recipe_and_online_raw_inputs(self):
        script = (EXPERIMENTS / 'scripts/run_four_mode_experiments.sh').read_text()
        prefix = script.split('echo "RUN_ROOT=$RUN_ROOT', 1)[0]
        cases = [
            ({}, 20, 3, 2),
            ({'SMOKE': '1'}, 1, 0, 2),
            ({'PRETRAIN_EPOCHS': '7', 'PRETRAIN_VAL_TAIL_PER_ROOT': '1',
              'INCLUDE_CAIDA_PRETRAIN': '1', 'TRACES': 'caida_2016'}, 7, 1, 3),
        ]
        for overrides, epochs, val_tail, roots in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(('PRETRAIN_', 'ZIPF_'))
                       and key not in {'SMOKE', 'INCLUDE_CAIDA_PRETRAIN', 'TRACES', 'RES_LIST'}}
                env.update({
                    'RUN_ROOT': str(work / 'run'), 'DATA_FULL_ROOT': str(work / 'data'),
                    'PYTHON': 'capture_python', 'NEUFSD_TEST_ROOT': str(EXPERIMENTS),
                    'CUDA_VISIBLE_DEVICES': '', **overrides,
                })
                commands = prefix + '''
ROOT="$NEUFSD_TEST_ROOT"
capture_python() { printf '%s\\n' "$@"; }
prepare_caida_pretrain_root() { printf '%s\\n' "$RUN_ROOT/caida_pretrain/${1}_${2}"; }
run_pretrain_one 64_64 1_10 ''
run_online_head 64_64 caida_2018 scratch_independent independent 0 1_10 ''
'''
                subprocess.run(['bash', '/dev/stdin'], input=commands, text=True,
                               env=env, check=True, stdout=subprocess.DEVNULL)
                args = (work / 'run/logs/pretrain_64_64_1_10.log').read_text().splitlines()
                self.assertEqual(args[args.index('--epochs') + 1], str(epochs))
                self.assertEqual(args[args.index('--val-tail-per-root') + 1], str(val_tail))
                self.assertEqual(len(args[args.index('--roots') + 1:args.index('--head')]), roots)
                online = (work / 'run/logs/online_scratch_independent_64_64_caida_2018_1_10.log').read_text().splitlines()
                self.assertEqual(online[online.index('--trace') + 1], 'caida_2018')
                self.assertEqual(online[online.index('--raw-dir') + 1],
                                 str(work / 'data/caida_2018/caida_1min_split'))


if __name__ == '__main__':
    unittest.main()
