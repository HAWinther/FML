"""Explicit bounded convergence and larger-mode inference campaign."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
if not 1<=a.jobs<=4:p.error('--jobs must be 1..4')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'science').exists():p.error('Use a fresh science directory')
(out/'science').mkdir();binary=a.bin_dir.resolve();here=Path(__file__).resolve().parent
records=[];lock=threading.Lock();env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1');env.pop('FML_PENCILS',None)
def run(name,args):
    cmd=list(map(str,args));start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    with lock:
        records.append(dict(label=name,argv=cmd,exit_code=r.returncode,seconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(name,r.returncode,round(time.monotonic()-start,2),flush=True)
    if r.returncode:raise RuntimeError(name+' failed')
# Fixed data are generated once, then reused for all forward/gradient comparisons.
run('data-reference',[binary/'pm_resolution','case',out/'science/data-reference',16,32,16,'-',0])
data=out/'science/data-reference.data.csv'
cases={}
for family,configurations in [
    ('force',[(8,m,16) for m in [8,16,32,64]]),
    ('time',[(8,32,s) for s in [2,4,8,16,32,64]]),
    ('particle',[(n,32,16) for n in [4,8,16,32]])]:
    for n,m,s in configurations:cases.setdefault((n,m,s),[]).append(family)
plan=dict(cases=[dict(n=n,m=m,steps=s,families=families) for (n,m,s),families in cases.items()],
          inference=dict(n=6,m=12,steps=4,chains=4,warmup=256,draws=512,leapfrog=16,stride=8))
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
# Independent simulations and chains share one bounded executor.
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
    futures=[pool.submit(run,f'chain{i}',[binary/'pm_resolution','chain',out/f'science/chain{i}',6,12,4,i,256,512,16,8]) for i in range(4)]
    futures += [pool.submit(run,f'case-{n}-{m}-{s}',[binary/'pm_resolution','case',out/f'science/case-{n}-{m}-{s}',n,m,s,data,int((n,m,s) in [(8,16,16),(16,32,16)])]) for n,m,s in cases]
    for future in futures:future.result()
run('prepare',[sys.executable,here/'prepare.py',*[out/f'science/chain{i}' for i in range(4)]])
run('diagnostics',[binary/'pm_resolution_diagnostics',out/'science/diagnostics.csv',*[out/f'science/chain{i}.features.csv' for i in range(4)]])
run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
