"""Small dependency-free HMC integration checks; no mixing/coverage claim."""
import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--exe', required=True, type=Path)
parser.add_argument('--mpi', choices=['true', 'false'], default='false')
parser.add_argument('--mpirun', default='mpirun')
parser.add_argument('--np-list', default='1 2 4')
a = parser.parse_args()
exe = a.exe.resolve()
env = dict(os.environ, OMP_NUM_THREADS='2', OMP_WAIT_POLICY='passive')
env.pop('FML_PENCILS', None)
with tempfile.TemporaryDirectory(prefix='fml-pm-hmc-') as temp:
    for n in (list(map(int, a.np_list.split())) if a.mpi == 'true' else [1]):
        launch = shlex.split(a.mpirun) + ['-np', str(n)] if a.mpi == 'true' else []
        cases = [('fixed', ['full', '0', '8'], '.full.csv'),
                 ('joint', ['full', '1', '8'], '.full.csv'),
                 ('conditional', ['chain', '0', '.1', '4', '4', '8', '0'], '.chain.csv')]
        for name, args, suffix in cases:
            prefix = Path(temp) / f'{n}-{name}'
            subprocess.run([*launch, str(exe), args[0], str(prefix), *args[1:]], env=env, check=True)
            with Path(str(prefix) + '.meta.csv').open() as file:
                meta = next(csv.DictReader(file))
            with Path(str(prefix) + suffix).open() as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == 8 and all(int(row['draw']) == i for i, row in enumerate(rows))
            assert all(math.isfinite(float(value)) for row in rows for value in row.values())
            assert int(meta['divergences']) == 0 and .5 < float(meta['acceptance']) <= 1
            if name == 'fixed':
                assert all(float(row['log_amplitude']) == 0 and abs(float(row['log_noise']) - math.log(.001)) < 1e-14 for row in rows)
            elif name == 'joint':
                assert all(-1 <= float(row['log_amplitude']) <= 1 and -14 <= float(row['log_noise']) <= -3 for row in rows)
            print(f'[PASS] PM HMC {name}, {n} rank(s): finite, all draws retained, no divergences', flush=True)
