"""Bounded forward-only PM force/particle reference study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--bin-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--np',type=int,default=4)
p.add_argument('--mpirun',default='mpirun')
p.add_argument('--timeout-seconds',type=int,default=300)
p.add_argument('--phase',choices=['all','cases','analyze'],default='all')
a=p.parse_args()
if not 0<=a.np<=64 or not 1<=a.timeout_seconds<=3600:
    p.error('--np must be 0..64; --timeout-seconds must be 1..3600')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent;exe=a.bin_dir.resolve()/'pm_nonlinear'
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',
         MPLCONFIGDIR=str(out/'mplcache'),XDG_CACHE_HOME=str(out/'cache'))
env.pop('FML_PENCILS',None)
records=[]
def stop(child):
    try: os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait()
def run(label,args):
    cmd=list(map(str,args));start=time.monotonic();record=dict(label=label,argv=cmd)
    with (out/(label+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=child.pid;records.append(record)
        (out/('commands-'+a.phase+'.json')).write_text(json.dumps(records,indent=2)+'\n')
        try: code=child.wait(timeout=a.timeout_seconds)
        except subprocess.TimeoutExpired: stop(child);code=124;record['timed_out']=True
        except KeyboardInterrupt: stop(child);code=130;record['interrupted']=True
    record.update(exit_code=code,seconds=time.monotonic()-start)
    (out/('commands-'+a.phase+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    print(label,code,round(record['seconds'],2),flush=True)
    if code: raise SystemExit(code if 0<code<256 else 1)
configs=[(n,m,32) for m in [32,64] for n in [64,96,128]]+[(128,64,64)]
if a.phase in ['all','cases']:
    if (out/'science').exists(): p.error('Use a fresh science directory for cases/all')
    (out/'science').mkdir()
    plan=dict(cases=[dict(n=n,m=m,steps=s) for n,m,s in configs],seed=91,power=.002,final=1.,
              active=122,observed=256,noise_power=1e-4,accuracy=dict(field=.05,noise=.1),
              np=a.np,omp_threads=1,timeout_seconds=a.timeout_seconds,
              inputs={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['reference.data.csv','reference.initial.csv']})
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    launch=shlex.split(a.mpirun)+['-np',str(a.np)] if a.np else []
    for n,m,s in configs:
        label=f'case-{n}-{m}-{s}'
        run(label,[*launch,exe,out/'science'/label,n,m,s,91,.002,1.,here/'reference.data.csv',0,0,1])
if a.phase in ['all','analyze']:
    run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
