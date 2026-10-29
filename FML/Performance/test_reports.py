"""Functional/provenance checks only: never gate CI on noisy runtime or RSS thresholds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--exe', required=True, type=Path)
options, remaining = parser.parse_known_args()
EXE = options.exe.resolve()
ENV = dict(os.environ, OMP_NUM_THREADS='2', OMP_DYNAMIC='false', OMP_WAIT_POLICY='passive')


class Reports(unittest.TestCase):
    def run_case(self, case, warmup=1, message_bytes=1048576):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'row.json'
            result = subprocess.run([str(EXE), '--case', case, '--mesh', '8', '--repeat', '2', '--warmup', str(warmup),
                                     '--message-bytes', str(message_bytes), '--output', str(path)],
                                    env=ENV, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            row = json.loads(path.read_text())
            self.assertEqual(row['schema_version'], 1)
            self.assertTrue(row['validation_passed'])
            self.assertEqual(row['ranks'], 1)
            self.assertEqual(len(row['rank_hosts']), 1)
            self.assertEqual(len(row['source_sha256']), 64)
            self.assertEqual(len(row['samples_max_rank_seconds']), 2)
            self.assertGreaterEqual(row['maximum_rank_seconds'], row['median_max_rank_seconds'])
            self.assertAlmostEqual(row['median_max_rank_seconds'], statistics.median(row['samples_max_rank_seconds']))
            self.assertIn('FFTW', row['counter_scope']['payload'])
            self.assertTrue(all(x >= 0 for x in row['samples_max_rank_seconds']))
            self.assertIn('FML_ENABLE_PERFORMANCE_COUNTERS', row['build_options'])
            self.assertTrue(row['compiler'] and row['fftw_library'])
            self.assertIn(row['precision_bytes'], [4, 8, 16])
            return row

    def test_all_cases(self):
        for case in ['fft_roundtrip', 'grid_copy', 'grid_clone', 'resample', 'migration', 'polyspectrum']:
            with self.subTest(case=case):
                row = self.run_case(case)
                c = row['counter_rank_sums']
                self.assertEqual(c['payload_sent_bytes'], c['payload_received_bytes'])
                if case == 'fft_roundtrip':
                    self.assertEqual(c['forward_ffts'], 2)
                    self.assertEqual(c['inverse_ffts'], 2)
                if case in ['grid_copy', 'grid_clone']:
                    self.assertEqual(c['grid_storage_allocations'], 2)
                    self.assertEqual(c['grid_copy_operations'], 2 if case == 'grid_copy' else 0)
                    self.assertEqual(c['grid_copied_bytes'] == 0, case == 'grid_clone')
                if case == 'polyspectrum':
                    self.assertGreater(c['inverse_ffts'], 0)
                    self.assertEqual(c['forward_ffts'], 0)
                    self.assertGreater(row['work_mesh'], row['mesh'])

    def test_cold_run(self):
        row = self.run_case('polyspectrum', warmup=0)
        self.assertGreater(row['counter_rank_sums']['grid_storage_allocations'], 0)

    def test_invalid_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'row.json')
            Path(path).write_text('previous report')
            for arguments in [['--case', 'unknown'], ['--mesh', '7'], ['--repeat', '0'], ['--seed', '-1'], ['--message-bytes', '0']]:
                result = subprocess.run([str(EXE), '--case', 'grid_clone', '--output', path, *arguments],
                                        env=ENV, capture_output=True, timeout=90)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(Path(path).read_text(), 'previous report')

    def test_runner_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            command = [sys.executable, str(Path(__file__).with_name('run.py')), '--exe', str(EXE), '--output', str(output),
                       '--ranks', '1', '--threads', '1', '--cases', 'grid_clone', '--mesh', '8', '--repeat', '1']
            result = subprocess.run(command, env=ENV, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertTrue(manifest['completed'])
            self.assertEqual(len(manifest['binary_sha256']), 64)
            rows = (output / 'results.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 1)
            original = (output / 'results.jsonl').read_bytes()
            result = subprocess.run(command, env=ENV, capture_output=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / 'results.jsonl').read_bytes(), original)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
