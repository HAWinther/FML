"""Bounded joint TSC MAP and analysis using the frozen matched reference."""
import argparse,hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--np',type=int,default=4);p.add_argument('--mpirun',default='mpirun');p.add_argument('--timeout-seconds',type=int,default=900);p.add_argument('--phase',choices=['all','cases','analyze'],default='all');a=p.parse_args()
if not 0<=a.np<=64 or not 1<=a.timeout_seconds<=3600:p.error('Require np=0..64 and timeout-seconds=1..3600')
here=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);science=out/'science';science.mkdir(exist_ok=True);data=here.parent/'pm_tsc_validation/science/reference.data.csv';exe=a.bin_dir.resolve()/'pm_tsc_joint';records=[]
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplcache'),XDG_CACHE_HOME=str(out/'cache'));env.pop('FML_PENCILS',None)
launch=shlex.split(a.mpirun)+['-np',str(a.np)] if a.np else []
def stop(child):
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait()
def run(label,args):
 cmd=list(map(str,args));start=time.monotonic();record=dict(label=label,argv=cmd)
 with (out/(label+'.log')).open('w') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);record['pid']=child.pid;records.append(record)
  if label!='analysis':record['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
  (out/('commands-'+a.phase+'.json')).write_text(json.dumps(records,indent=2)+'\n')
  try:code=child.wait(timeout=a.timeout_seconds)
  except subprocess.TimeoutExpired:stop(child);code=124;record['timed_out']=True
  except KeyboardInterrupt:stop(child);code=130;record['interrupted']=True
 record.update(exit_code=code,seconds=time.monotonic()-start);(out/('commands-'+a.phase+'.json')).write_text(json.dumps(records,indent=2)+'\n');print(label,code,flush=True)
 if code:raise SystemExit(code if 0<code<256 else 1)
if a.phase in ['all','cases']:
 if any(science.iterdir()):p.error('Use a fresh science directory')
 plan=json.loads((here/'plan.json').read_text());plan.update(np=a.np,timeout_seconds=a.timeout_seconds,data_sha256=hashlib.sha256(data.read_bytes()).hexdigest());(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 run('joint',[*launch,exe,science/'joint',data,32,16,16,128,1,1])
if a.phase in ['all','analyze']:run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis'])
