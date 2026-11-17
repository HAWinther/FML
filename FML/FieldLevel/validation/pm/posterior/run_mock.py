"""Independent same-model N16 mock with a frozen, predeclared sampler protocol."""
import argparse,concurrent.futures,csv,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ic-seed',type=int,default=191);p.add_argument('--noise-seed',type=int,default=192);p.add_argument('--jobs',type=int,default=4);p.add_argument('--timeout-seconds',type=int,default=7200);p.add_argument('--phase',choices=['all','cases','analyze'],default='all');a=p.parse_args()
if not 0<=a.ic_seed<=2147483647 or not 0<=a.noise_seed<=2147483647:p.error('IC/noise seeds must be nonnegative signed integers')
if a.ic_seed==91 or a.noise_seed==92:p.error('Use IC and noise seeds distinct from the original 91/92 pilot')
if not 1<=a.jobs<=4 or not 1<=a.timeout_seconds<=7200:p.error('jobs must be 1..4 and timeout 1..7200')
here=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);science=out/'science';science.mkdir(exist_ok=True);bins=a.bin_dir.resolve();env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplcache'),XDG_CACHE_HOME=str(out/'cache'));env.pop('FML_PENCILS',None);records=[];lock=threading.Lock();children=set();seeds=[a.ic_seed,a.noise_seed]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
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
 if label!='analysis':record['binary_sha256']=digest(Path(cmd[0]))
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
 plan=json.loads((here/'plan.json').read_text());plan.update(note='Independent same-model N16 mock; frozen original sampler protocol; one realization does not establish ensemble coverage. No diagnostic thinning.',scheme='preconditioned',scalar_sweeps=4,draws=512,warmup=64,leapfrog=6,predictive_stride=32,timeout_seconds=a.timeout_seconds,problem=dict(kind='same_model_independent',ic_seed=a.ic_seed,noise_seed=a.noise_seed,scope='one held-out mock; no ensemble coverage claim'))
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 if run('mock',[bins/'pm_tsc_posterior','mock',science/'mock','-','-',16,16,16,0,0,2,2,.001,0,1,*seeds]):raise SystemExit(1)
 data=science/'mock.data.csv';initialization=science/'map.physical.csv'
 if run('map',[bins/'pm_tsc_joint',science/'map',data,16,16,16,128,1,0,*seeds]):raise SystemExit(1)
 with (science/'map.fit.csv').open() as f:fit={k:float(v) for k,v in next(csv.DictReader(f)).items()}
 if fit['status']!=0 or fit['joint_gradient_norm']>=2e-5:
  (out/'preflight-failure.json').write_text(json.dumps(dict(reason='MAP initialization did not converge under the predeclared cap',fit=fit),indent=2)+'\n');raise SystemExit(1)
 plan.update(data_sha256=digest(data),map_seed_sha256=digest(initialization));plan['problem'].update(truth_sha256=digest(science/'mock.truth.csv'),signal_sha256=digest(science/'mock.truth_modes.csv'))
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 if run('target-check',[bins/'pm_tsc_posterior','check',science/'target',data,initialization,16,16,16,0,0,2,2,.001,0,1,*seeds]):raise SystemExit(1)
 if run('metric',[bins/'pm_tsc_posterior','metric',science/'metric',data,initialization,16,16,16,0,0,2,2,.0001,0,1,*seeds]):raise SystemExit(1)
 plan.update(field_metric='fixed regularized Hessian at data-only MAP reference, A=0 N=N0',metric_fd_step=.0001,metric_sha256=digest(science/'metric.matrix.csv'));(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 def chain(i):return run('chain'+str(i),[bins/'pm_tsc_posterior','preconditioned',science/('chain'+str(i)),data,initialization,16,16,16,i,64,512,6,.25,1,32,science/'metric.matrix.csv',*seeds])
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs)
 try:codes=list(pool.map(chain,range(4)))
 except BaseException:
  with lock:active=list(children)
  for child in active:stop(child)
  pool.shutdown(wait=True,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 if any(codes):raise SystemExit(1)
 if run('diagnostics',[bins/'pm_tsc_diagnostics',science/'diagnostics.csv',*[science/('chain'+str(i)+'.features.csv') for i in range(4)]]):raise SystemExit(1)
if a.phase in ['all','analyze']:raise SystemExit(run('analysis',[sys.executable,here/'analyze.py','--input',out,'--output',out/'analysis']))
