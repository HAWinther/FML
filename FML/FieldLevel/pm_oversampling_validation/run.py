"""Explicit PM oversampling campaign; uses the existing bounded native drivers."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--bin-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--jobs', type=int, default=2)
p.add_argument('--phase', choices=['all', 'cases', 'force', 'analyze'], default='all')
a = p.parse_args()
if not 1 <= a.jobs <= 4:
    p.error('--jobs must be 1..4')
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
binary = a.bin_dir.resolve()
here = Path(__file__).resolve().parent
probe = here.parent/'pm_resolution_validation'
records, lock = [], threading.Lock()
env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='passive', OPENBLAS_NUM_THREADS='1',
           MPLCONFIGDIR=str(out/'mplcache'), XDG_CACHE_HOME=str(out/'cache'))
env.pop('FML_PENCILS', None)
def run(name, args):
    cmd = list(map(str, args))
    start = time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    with lock:
        records.append(dict(label=name, argv=cmd, exit_code=result.returncode, seconds=time.monotonic()-start))
        (out/('commands-'+a.phase+'.json')).write_text(json.dumps(records, indent=2)+'\n')
    print(name, result.returncode, round(time.monotonic()-start, 2), flush=True)
    if result.returncode:
        raise RuntimeError(name+' failed; inspect '+str(out/(name+'.log')))
if a.phase in ['all', 'cases']:
    if (out/'science').exists():
        p.error('Use a fresh science directory for cases/all')
    (out/'science').mkdir()
    cases = {}
    for family, configs in [
        ('dense-force', [(64, m, 16) for m in [8, 16, 32, 64]]),
        ('particles-M16', [(n, 16, 16) for n in [16, 32, 48, 64]]),
        ('particles-M32', [(n, 32, 16) for n in [32, 48, 64]]),
        ('time', [(32, 16, s) for s in [4, 8, 16, 32]])]:
        for config in configs:
            cases.setdefault(config, []).append(family)
    plan = [dict(n=n, m=m, steps=s, families=f) for (n, m, s), f in cases.items()]
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    run('data-reference', [binary/'pm_resolution', 'case', out/'science/data-reference', 16, 32, 16, '-', 0])
    data = out/'science/data-reference.data.csv'
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run, f'case-{n}-{m}-{s}',
                              [binary/'pm_resolution', 'case', out/f'science/case-{n}-{m}-{s}', n, m, s, data,
                               int((n, m, s) == (32, 16, 16))]) for n, m, s in cases]
        for future in futures:
            future.result()
if a.phase in ['all', 'force']:
    run('force-probe', [binary/'pm_force_probe', out/'science/data-reference.initial.csv', out/'science/native_force.csv'])
    run('numpy-probe', [sys.executable, probe/'assignment_probe.py', '--initial', out/'science/data-reference.initial.csv', '--output', out/'science/numpy-force'])
if a.phase in ['all', 'force', 'analyze']:
    force = out/'force-input'
    force.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(out/'science/native_force.csv', force/'native.csv')
    shutil.copy2(out/'science/numpy-force/linear_force.csv', force/'linear_force.csv')
    run('force-analysis', [sys.executable, probe/'force_analysis.py', '--input', force, '--output', out/'analysis'])
if a.phase in ['all', 'analyze']:
    run('analysis', [sys.executable, here/'analyze.py', '--input', out, '--output', out/'analysis'])
