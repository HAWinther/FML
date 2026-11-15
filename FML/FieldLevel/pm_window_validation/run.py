"""Bounded comparison of uncompensated and CIC-compensated PM forces."""
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
p.add_argument('--probe-dir',type=Path,required=True)
p.add_argument('--np',type=int,default=4)
p.add_argument('--mpirun',default='mpirun')
p.add_argument('--timeout-seconds',type=int,default=300)
p.add_argument('--phase',choices=['all','cases','probes','analyze'],default='all')
a=p.parse_args()
if not 0<=a.np<=64 or not 1<=a.timeout_seconds<=3600:
    p.error('--np must be 0..64; --timeout-seconds must be 1..3600')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent;exe=a.bin_dir.resolve()/'pm_nonlinear';probe=a.probe_dir.resolve()/'pm_force_probe';data=here.parent/'pm_force_reference_validation/reference.data.csv'
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
def run(label,args,allow_accuracy_failure=False):
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
    if code and not (allow_accuracy_failure and code==1): raise SystemExit(code if 0<code<256 else 1)
forward=[(128,32,32),(96,64,32),(128,64,32),(128,64,64)]
gradients=[(64,32,32),(64,64,32)]
if a.phase in ['all','cases']:
    if (out/'science').exists(): p.error('Use a fresh science directory for cases/all')
    (out/'science').mkdir()
    plan=dict(forward=[dict(n=n,m=m,steps=s) for n,m,s in forward],gradients=[dict(n=n,m=m,steps=s) for n,m,s in gradients],
              windows=[0,1],seed=91,power=.002,final=1.,active=122,observed=256,noise_power=1e-4,
              accuracy=dict(field=.05,gradient=.15,noise=.1),
              probe_kicks={'0':[1e-8,1e-9],'1':[1e-8,1e-9,1e-10]},probe_acceptance_max_delta_a={'0':1e-8,'1':1e-9},
              np=a.np,omp_threads=1,timeout_seconds=a.timeout_seconds,
              data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),linear_sha256=hashlib.sha256((here/'linear.initial.csv').read_bytes()).hexdigest())
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    launch=shlex.split(a.mpirun)+['-np',str(a.np)] if a.np else []
    for window in [0,1]:
        for n,m,s in forward:
            label=f'forward-w{window}-{n}-{m}-{s}'
            run(label,[*launch,exe,out/'science'/label,n,m,s,91,.002,1.,data,0,0,1,window])
        for n,m,s in gradients:
            label=f'gradient-w{window}-{n}-{m}-{s}'
            run(label,[*launch,exe,out/'science'/label,n,m,s,91,.002,1.,data,0,0,0,window])
    label='directional-w1'
    run(label,[*launch,exe,out/'science'/label,32,16,16,91,.002,1.,data,1,0,0,1])
if a.phase in ['all','probes']:
    for window in [0,1]:
        directory=out/f'force-w{window}';directory.mkdir(exist_ok=True)
        run(f'native-probe-w{window}',[probe,here/'linear.initial.csv',directory/'native.csv',window])
        run(f'numpy-probe-w{window}',[sys.executable,here.parent/'pm_resolution_validation/assignment_probe.py','--initial',here/'linear.initial.csv','--output',directory,'--force-window',window])
        run(f'force-analysis-w{window}',[sys.executable,here.parent/'pm_resolution_validation/force_analysis.py','--input',directory,'--output',out/f'force-analysis-w{window}'],allow_accuracy_failure=True)
if a.phase in ['all','analyze']:
    run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
