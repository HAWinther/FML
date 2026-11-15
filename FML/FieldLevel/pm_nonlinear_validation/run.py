"""Explicit many-mode PM validation; scientific accuracy failures remain visible."""
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
p.add_argument('--phase', choices=['all', 'cases', 'fits', 'analyze'], default='all')
a = p.parse_args()
if not 1 <= a.jobs <= 4:
    p.error('--jobs must be 1..4')
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
exe = a.bin_dir.resolve()/'pm_nonlinear'
here = Path(__file__).resolve().parent
records, lock = [], threading.Lock()
env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='passive', OPENBLAS_NUM_THREADS='1',
           MPLCONFIGDIR=str(out/'mplcache'), XDG_CACHE_HOME=str(out/'cache'))
env.pop('FML_PENCILS', None)
def run(label, cmd):
    cmd = list(map(str, cmd))
    start = time.monotonic()
    with (out/(label+'.log')).open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    with lock:
        records.append(dict(label=label, argv=cmd, exit_code=result.returncode, seconds=time.monotonic()-start))
        (out/('commands-'+a.phase+'.json')).write_text(json.dumps(records, indent=2)+'\n')
    print(label, result.returncode, round(time.monotonic()-start, 2), flush=True)
    if result.returncode:
        raise RuntimeError(label+' failed; inspect its log')
fixtures = [dict(name='weak91', seed=91, power=.0005, final=.6),
            dict(name='strong91', seed=91, power=.002, final=1.),
            dict(name='strong173', seed=173, power=.002, final=1.)]
# Fixed data first; every later comparison sees this identical noisy realization.
configs = [(32,16,16), (48,16,16), (64,16,16), (64,32,16), (64,32,32), (64,64,32), (64,32,64)]
plan = dict(fixtures=fixtures, cases=[dict(n=n,m=m,steps=s) for n,m,s in configs],
            fits=[dict(fixture=f,n=n,m=16,steps=16,iterations=12) for f,n in [('weak91',16),('weak91',32),('strong91',32)]],
            accuracy=dict(field=.05,gradient=.15,noise=.1), active=122, observed=256)
def command(f, prefix, n, m, steps, data, fd=0, fit=0):
    return [exe, prefix, n, m, steps, f['seed'], f['power'], f['final'], data, fd, fit]
if a.phase in ['all','cases']:
    if (out/'science').exists():
        p.error('Use a fresh output directory for cases/all')
    (out/'science').mkdir()
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run, f['name']+'-reference', command(f,out/'science'/(f['name']+'-reference'),64,32,32,'-')) for f in fixtures]
        for future in futures: future.result()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = []
        for f in fixtures:
            for n,m,s in configs:
                if (n,m,s)==(64,32,32): continue
                prefix=out/'science'/f"{f['name']}-{n}-{m}-{s}"
                fd=int((n,m,s)==(32,16,16) and f['name']!='strong173')
                futures.append(pool.submit(run,prefix.name,command(f,prefix,n,m,s,out/'science'/(f['name']+'-reference.data.csv'),fd)))
        for future in futures: future.result()
if a.phase in ['all','fits']:
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = []
        for fit in plan['fits']:
            f=next(f for f in fixtures if f['name']==fit['fixture'])
            prefix=out/'science'/f"fit-{f['name']}-{fit['n']}"
            futures.append(pool.submit(run,prefix.name,command(f,prefix,fit['n'],fit['m'],fit['steps'],out/'science'/(f['name']+'-reference.data.csv'),fit=fit['iterations'])))
        for future in futures: future.result()
if a.phase in ['all','analyze']:
    run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
