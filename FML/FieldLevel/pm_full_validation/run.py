"""Bounded full-field PM campaign, with optional distributed/sanitizer replay."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

p=argparse.ArgumentParser()
p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--jobs',type=int,default=4);p.add_argument('--warmup',type=int,default=256)
p.add_argument('--draws',type=int,default=1024);p.add_argument('--leapfrog',type=int,default=32)
p.add_argument('--mpi-bin-dir',type=Path);p.add_argument('--pencil-bin-dir',type=Path);p.add_argument('--sanitizer-bin-dir',type=Path)
p.add_argument('--mpirun',default='mpirun');a=p.parse_args()
if not(1<=a.jobs<=4 and 64<=a.warmup<=8192 and 128<=a.draws<=16384 and 2<=a.leapfrog<=128):p.error('Invalid bounded campaign counts')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'science').exists():p.error('Use a fresh output directory')
(out/'science').mkdir();(out/'replay').mkdir();binary=a.bin_dir.resolve();here=Path(__file__).resolve().parent
records=[];lock=threading.Lock();env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1');env.pop('FML_PENCILS',None)
def run(label,args,extra=None):
    cmd=list(map(str,args));start=time.monotonic()
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT)
    with lock:
        records.append(dict(label=label,argv=cmd,environment=extra or {},exit_code=r.returncode,seconds=time.monotonic()-start))
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,r.returncode,round(time.monotonic()-start,2),flush=True)
    if r.returncode:raise RuntimeError(f'{label} failed; see log')
# Complete fixed-parameter chains first, then joint field/spectrum/noise chains.
for family,joint in [('fixed',0),('joint',1)]:
    prefixes=[out/'science'/f'{family}{i}' for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures=[pool.submit(run,f'{family}{i}',[binary/'pm_full',prefix,joint,i,a.warmup,a.draws,.08,a.leapfrog,1,8,1]) for i,prefix in enumerate(prefixes)]
        for future in futures:future.result()
    run('prepare-'+family,[sys.executable,here/'prepare.py',*(['--joint'] if joint else []),*prefixes])
    run('diagnostics-'+family,[binary/'pm_full_diagnostics',out/f'science/{family}.diagnostics.csv',*[str(x)+'.features.csv' for x in prefixes]])
variants=[('serial',binary,[],{})]
if a.mpi_bin_dir:
    variants += [(f'mpi{n}',a.mpi_bin_dir.resolve(),[*shlex.split(a.mpirun),'-np',n],{}) for n in [1,2,4]]
if a.pencil_bin_dir:
    variants += [('pencil4',a.pencil_bin_dir.resolve(),[*shlex.split(a.mpirun),'-np',4],{'FML_PENCILS':'2 2'}),
                 ('pencil4-thread1',a.pencil_bin_dir.resolve(),[*shlex.split(a.mpirun),'-np',4],{'FML_PENCILS':'2 2','OMP_NUM_THREADS':'1'})]
if a.sanitizer_bin_dir:variants += [('sanitizer',a.sanitizer_bin_dir.resolve(),[],{'ASAN_OPTIONS':'detect_leaks=0','UBSAN_OPTIONS':'halt_on_error=1'})]
for name,directory,launch,extra in variants:
    for joint in [0,1]:run(f'{name}-{joint}',[*launch,directory/'pm_full',out/'replay'/f'{name}-{joint}',joint,0,16,24,.06,8,0,4,1],extra)
run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
