"""Four bounded N16 pilot chains; preserve all statuses and mixing failures."""
import argparse,concurrent.futures,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scheme',choices=['joint','blocked','preconditioned'],default='joint');p.add_argument('--draws',type=int,default=128);p.add_argument('--leapfrog',type=int,default=8);p.add_argument('--predictive-stride',type=int,default=8);p.add_argument('--jobs',type=int,default=4);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('--phase',choices=['all','cases','analyze'],default='all');a=p.parse_args()
if not 1<=a.jobs<=4 or not 1<=a.timeout_seconds<=7200:p.error('jobs must be 1..4, timeout 1..7200')
if not 2<=a.draws<=4096 or not 1<=a.leapfrog<=32 or not 1<=a.predictive_stride<=a.draws:p.error('draws must be 2..4096, leapfrog 1..32, predictive stride 1..draws')
here=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);science=out/'science';science.mkdir(exist_ok=True);bins=a.bin_dir.resolve();env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplcache'),XDG_CACHE_HOME=str(out/'cache'));env.pop('FML_PENCILS',None);records=[];lock=threading.Lock();children=set()
data=here/'data/reference.data.csv';seed=here/'data/initial.physical.csv'
def save():
 with lock:(out/('commands-'+a.phase+'.json')).write_text(json.dumps(records,indent=2)+'\n')
def stop(child):
 if child.poll() is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait()
def run(label,args):
 cmd=list(map(str,args));record=dict(label=label,argv=cmd);start=time.monotonic()
 if label!='analysis':record['binary_sha256']=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()
 with (out/(label+'.log')).open('w') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);record['pid']=child.pid
  with lock:records.append(record);children.add(child)
  save()
  try:code=child.wait(timeout=a.timeout_seconds)
  except subprocess.TimeoutExpired:stop(child);code=124;record['timed_out']=True
  except KeyboardInterrupt:stop(child);code=130;record['interrupted']=True
 with lock:children.discard(child)
 record.update(exit_code=code,seconds=time.monotonic()-start);save();print(label,code,flush=True);return code
if a.phase in ['all','cases']:
 if any(science.iterdir()):p.error('Use a fresh science directory')
 plan=json.loads((here/'plan.json').read_text());plan.update(draws=a.draws,leapfrog=a.leapfrog,predictive_stride=a.predictive_stride,scheme=a.scheme,scalar_sweeps=4 if a.scheme!='joint' else 0,timeout_seconds=a.timeout_seconds,data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),map_seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest());(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 if run('target-check',[bins/'pm_tsc_posterior','check',science/'target',data,seed,16,16,16,0,0,2,2,.001,0,1]):raise SystemExit(1)
 if a.scheme=='preconditioned' and run('metric',[bins/'pm_tsc_posterior','metric',science/'metric',data,seed,16,16,16,0,0,2,2,.0001,0,1]):raise SystemExit(1)
 if a.scheme=='preconditioned':
  plan.update(field_metric='fixed regularized Hessian at common physical reference, A=0 N=N0',metric_fd_step=.0001,metric_sha256=hashlib.sha256((science/'metric.matrix.csv').read_bytes()).hexdigest());(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 def chain(i):return run('chain'+str(i),[bins/'pm_tsc_posterior','chain' if a.scheme=='joint' else a.scheme,science/('chain'+str(i)),data,seed,16,16,16,i,64,a.draws,a.leapfrog,.25,1,a.predictive_stride]+([science/'metric.matrix.csv'] if a.scheme=='preconditioned' else []))
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs)
 try:codes=list(pool.map(chain,range(4)))
 except BaseException:
  with lock:active=list(children)
  for child in active:stop(child)
  pool.shutdown(wait=True,cancel_futures=True)
  raise
 else:pool.shutdown(wait=True)
 if any(codes):raise SystemExit(1)
 if run('map',[bins/'pm_tsc_joint',science/'map',data,16,16,16,128,1,0]):raise SystemExit(1)
 if run('diagnostics',[bins/'pm_tsc_diagnostics',science/'diagnostics.csv',*[science/('chain'+str(i)+'.features.csv') for i in range(4)]]):raise SystemExit(1)
if a.phase in ['all','analyze']:raise SystemExit(run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis']))
