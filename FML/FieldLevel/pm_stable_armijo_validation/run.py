"""Stable Armijo: frozen-state regression and unchanged-target reconstruction."""
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
p.add_argument('--timeout-seconds',type=int,default=900)
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
if a.phase in ['all','cases']:
    if (out/'science').exists():p.error('Use a fresh science directory')
    (out/'science').mkdir()
    prior=here.parent/'pm_optimizer_validation'
    plan=json.loads((prior/'plan.json').read_text())
    plan['runs']=[dict(name='stable-lbfgs',optimizer=2,iterations=128,stable=1),dict(name='stable-gd',optimizer=1,iterations=24,stable=1)]
    plan.update(np=a.np,omp_threads=1,timeout_seconds=a.timeout_seconds)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for path in (prior/'science').glob('reference.*.csv'):(out/'science'/path.name).write_bytes(path.read_bytes())
    data=out/'science/reference.data.csv'
    if hashlib.sha256(data.read_bytes()).hexdigest()!=plan['data_sha256']:p.error('Frozen data changed')
    launch=shlex.split(a.mpirun)+['-np',str(a.np)] if a.np else []
    run('frozen',[*launch,a.bin_dir.resolve()/'pm_stable_frozen',out/'science/frozen',data,here.parent/'pm_stationarity_validation/science/stalled',32,16,16])
    for case in plan['runs']:
        run(case['name'],[*launch,exe,out/'science'/case['name'],32,16,16,91,.002,1.,data,0,case['iterations'],0,1,16,case['optimizer'],5,1])
if a.phase in ['all','analyze']:
    run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
