"""Capture or probe the frozen stalled reconstruction, without changing its solver."""
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
p.add_argument('--stable-changes',action='store_true',help='Capture/probe the stable Armijo iteration-98 endpoint')
p.add_argument('--phase',choices=['all','capture','probe','analyze'],default='probe')
a=p.parse_args()
if not 0<=a.np<=64 or not 1<=a.timeout_seconds<=3600:
    p.error('--np must be 0..64; --timeout-seconds must be 1..3600')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
study=here.parent/'pm_stable_stationarity_validation' if a.stable_changes else here
exe=a.bin_dir.resolve()/'pm_stationarity_probe'
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
science=out/'science';science.mkdir(exist_ok=True)
data=here.parent/'pm_optimizer_validation/science/reference.data.csv'
launch=shlex.split(a.mpirun)+['-np',str(a.np)] if a.np else []
if a.phase in ['all','capture']:
    if (science/'stalled.meta.csv').exists():p.error('Use a fresh capture directory')
    run('capture',[*launch,a.bin_dir.resolve()/'pm_stationarity_capture',science/'stalled',data,32,16,16,128,*([1] if a.stable_changes else [])])
if a.phase in ['all','probe']:
    if not (science/'stalled.meta.csv').exists():
        for path in (study/'science').glob('stalled.*.csv'):(science/path.name).write_bytes(path.read_bytes())
    if not (science/'stalled.meta.csv').exists():p.error('No frozen snapshot; run --phase capture first')
    plan=dict(stable_changes=a.stable_changes,n=32,m=16,steps=16,force_window=1,reference_cic_mesh=16,seed=91,power=.002,final=1.,
              data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),np=a.np,omp_threads=1,
              snapshot_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in science.glob('stalled.*.csv')})
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if (science/'probe.directions.csv').exists():p.error('Use a fresh probe directory')
    run('probe',[*launch,exe,science/'probe',data,science/'stalled',32,16,16,1])
if a.phase in ['all','probe','analyze']:
    run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis',*(['--reference',here.parent/'pm_stable_armijo_validation/science/stable-lbfgs.fit_coefficients.csv','--line-expectation','measure'] if a.stable_changes else [])])
