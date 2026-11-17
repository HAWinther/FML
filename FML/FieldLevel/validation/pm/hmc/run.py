"""Portable bounded PM HMC campaign using prebuilt validation binaries."""
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

parser=argparse.ArgumentParser()
parser.add_argument('--bin-dir',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--mpi-bin-dir',type=Path)
parser.add_argument('--pencil-bin-dir',type=Path)
parser.add_argument('--sanitizer-bin-dir',type=Path)
parser.add_argument('--mpirun',default='mpirun')
parser.add_argument('--jobs',type=int,default=2)
a=parser.parse_args()
if not 1<=a.jobs<=8:parser.error('--jobs must be in 1..8')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'science').exists():parser.error('Use a fresh output directory')
(out/'science').mkdir();(out/'replay').mkdir()
serial=a.bin_dir.resolve();records=[];lock=threading.Lock()
env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1');env.pop('FML_PENCILS',None)
def run(label,cmd,extra=None):
    cmd=list(map(str,cmd));start=time.monotonic()
    with (out/(label+'.log')).open('w') as file:
        result=subprocess.run(cmd,env=dict(env,**(extra or {})),stdout=file,stderr=subprocess.STDOUT)
    with lock:
        records.append(dict(label=label,argv=cmd,environment=extra or {},exit_code=result.returncode,seconds=time.monotonic()-start))
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(label+' failed; see log')
science=[('ref41',['reference',out/'science/ref41',41,6]),('ref81',['reference',out/'science/ref81',81,6]),
         ('ref109',['reference',out/'science/ref109',109,8.1]),('sweep',['sweep',out/'science/sweep'])]
science += [(f'fixed{i}',['chain',out/f'science/fixed{i}',i,.1,8,64,256,0]) for i in range(4)]
science += [(f'adapt{i}',['chain',out/f'science/adapt{i}',i+4,.1,8,128,256,1]) for i in range(2)]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
    futures=[pool.submit(run,label,[serial/'pm_hmc',*args]) for label,args in science]
    for future in futures:future.result()
for family,count in [('fixed',4),('adapt',2)]:
    run('diagnostics-'+family,[serial/'pm_hmc_diagnostics',out/f'science/{family}.diagnostics.csv',
        *[out/f'science/{family}{i}.chain.csv' for i in range(count)]])
variants=[('serial',serial,[],{})]
if a.mpi_bin_dir:
    variants += [(f'mpi{n}',a.mpi_bin_dir.resolve(),[*shlex.split(a.mpirun),'-np',str(n)],{}) for n in [1,2,4]]
if a.pencil_bin_dir:
    launch=[*shlex.split(a.mpirun),'-np','4'];directory=a.pencil_bin_dir.resolve()
    variants += [('pencil4',directory,launch,{'FML_PENCILS':'2 2'}),
                 ('pencil4-thread1',directory,launch,{'FML_PENCILS':'2 2','OMP_NUM_THREADS':'1'})]
if a.sanitizer_bin_dir:
    variants += [('sanitizer',a.sanitizer_bin_dir.resolve(),[],{'ASAN_OPTIONS':'detect_leaks=0','UBSAN_OPTIONS':'halt_on_error=1'})]
for label,directory,launch,extra in variants:
    for case,arguments in [('conditional',['chain',0,.1,8,16,24,0]),('full-fixed',['full',0,16]),('full-joint',['full',1,16])]:
        run(label+'-'+case,[*launch,directory/'pm_hmc',arguments[0],out/'replay'/f'{label}-{case}',*arguments[1:]],extra)
# Analyze the available variants; mathematical gates are identical in serial-only mode.
run('analysis',[sys.executable,Path(__file__).with_name('analyze.py'),'--input',out,'--output',out/'analysis'])
