#!/usr/bin/env python3
"""Frozen paired dataset-204 comparison; fixed budgets, measured field-gradient cost."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,os,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent
CALIB=HERE.parent/'calibration';sys.path.insert(0,str(CALIB))
from common import Budget,sha,write
import analyze as calibration_analysis
import numpy as np
CASES=['baseline','half']
SOURCES=[HERE/'compare.py',HERE/'run.py',HERE/'FieldHMC.h',HERE.parent/'tempering/Main.cpp',HERE.parent/'tempering/analyze.py',HERE.parent/'tempering/check_prior.py',HERE.parent/'tempering/plan.json',CALIB/'analyze.py',CALIB/'study.py',CALIB/'common.py',HERE.parent/'tempering/sbc_ranks.py']

def prepare(root,source,exe,diag):
 if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('New comparison requires empty output')
 native=json.loads((source/'plan.json').read_text());source_files=[source/'plan.json',source/'ensemble0/target.json',source/'calibration-summary.json',*[source/f'ensemble{e}/CURRENT' for e in range(4)]]
 # Prefix hashes let the counted baseline replay be checked against the original.
 source_files += [source/f'ensemble{e}/draws/draw-{j}.rank-0.json' for e in range(4) for j in range(512)]
 frozen=dict(binary_sha256=sha(exe),diagnostics_sha256=sha(diag),sources={str(p):sha(p) for p in SOURCES},original={str(p):sha(p) for p in source_files},original_directory=str(source),plans={},warmup=256,samples=512,wall_limit=14400,batch_reservation=2400,scope='Targeted paired 256/512 integrator comparison on an already inspected failed dataset; not independent calibration')
 for name in CASES:
  p=json.loads((HERE/f'plan_{name}.json').read_text());expected=json.loads((source/'plan.json').read_text());expected.update(scope=p['scope'],samples=512);expected['field']=dict(native['field'],step_size=.15 if name=='baseline' else .075,n_leapfrog=8 if name=='baseline' else 16)
  if p!=expected:raise ValueError('Comparison changed controls other than declared field integrator/budget')
  directory=root/name;(directory/'plans').mkdir(parents=True)
  path=directory/'plans/mock204.json';write(path,p);frozen['plans'][name]=sha(path)
  write(directory/'FROZEN.json',dict(binary_sha256=sha(exe),scope='Counted-integrator cohort manifest; parent FROZEN.json binds full protocol'))
 write(root/'FROZEN.json',frozen);print('Frozen two 4-chain 256/512 cohorts; no chains launched')

def verify(root,exe,diag):
 f=json.loads((root/'FROZEN.json').read_text())
 if sha(exe)!=f['binary_sha256'] or sha(diag)!=f['diagnostics_sha256']:raise ValueError('Frozen binary changed')
 for group in ['sources','original']:
  for name,digest in f[group].items():
   if sha(name)!=digest:raise ValueError('Frozen input/source changed: '+name)
 for name,digest in f['plans'].items():
  if sha(root/name/'plans/mock204.json')!=digest:raise ValueError('Frozen cohort changed')
 return f

def run(root,exe,diag,resume,max_rounds,lock):
 f=verify(root,exe,diag)
 if not (root/'budget.json').exists() and any((root/name/'mock204').exists() for name in CASES):raise ValueError('Missing ledger beside sampling work')
 budget=Budget(root/'budget.json',f['wall_limit'],f['batch_reservation']);stop=False;children={};sent=set()
 def pause(sig=None,frame=None):
  nonlocal stop
  stop=True;(root/'PAUSE').touch()
  for name in CASES:
   d=root/name/'mock204'
   if d.exists():(d/'PAUSE').touch()
  for child in list(children.values()):
   if child.poll() is None and child.pid not in sent:child.send_signal(signal.SIGTERM);sent.add(child.pid)
 signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
 if resume:
  (root/'PAUSE').unlink(missing_ok=True)
  for name in CASES:(root/name/'mock204/PAUSE').unlink(missing_ok=True)
 rounds=0;reason=None
 while True:
  todo=[name for name in CASES if not (root/name/'mock204/completed.json').exists()]
  if not todo:break
  if stop or (root/'PAUSE').exists():reason='pause_requested';break
  if max_rounds is not None and rounds>=max_rounds:reason='bounded_round_limit';break
  if not budget.reserve():reason='sampling_time_budget';break
  def one(name):
   start=time.monotonic();cmd=[sys.executable,str(HERE/'run.py'),str(root/name/'mock204'),'--exe',str(exe),'--plan',str(root/name/'plans/mock204.json'),'--workers','4','--max-chunks','1']
   with (root/f'{name}.log').open('a') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),));children[name]=child;code=child.wait()
   if code:raise RuntimeError(name+' runner failed; retain data and inspect log')
   elapsed=time.monotonic()-start;d=root/name/'mock204';summaries=[json.loads((d/f'ensemble{e}/summary.json').read_text()) for e in range(4)]
   receipt=root/name/'cost.json';cost=json.loads(receipt.read_text()) if receipt.exists() else dict(active_cohort_wall_seconds=0.,batches=0)
   cost['active_cohort_wall_seconds']+=elapsed;cost['batches']+=1;write(receipt,cost)
   print(name,[s['completed_iterations'] for s in summaries],round(elapsed,2),'active seconds',flush=True)
  start=time.monotonic()
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(one,name) for name in todo]
    while not all(x.done() for x in futures):
     concurrent.futures.wait(futures,timeout=1)
     if stop or (root/'PAUSE').exists():pause()
    for x in futures:x.result()
  finally:budget.finish(time.monotonic()-start);children.clear()
  rounds+=1
 complete=all((root/name/'mock204/completed.json').exists() for name in CASES)
 write(root/('completed.json' if complete else 'paused.json'),dict(complete=complete,reason=reason,budget=budget.state))
 if complete:(root/'paused.json').unlink(missing_ok=True)
 print('Comparison sampling '+('complete' if complete else 'paused'),flush=True)

def analyze(root,exe,diag):
 f=verify(root,exe,diag)
 if not json.loads((root/'completed.json').read_text())['complete']:raise ValueError('Full fixed comparison budget required')
 study=dict(posterior_gates=dict(max_rhat=1.01,min_bulk_ess=400,min_tail_ess=400),interval_probabilities=[.5,.9,.95],endpoint_index=511,analysis_seed=940901)
 outcomes={};values={}
 for name in CASES:
  p=json.loads((root/name/'plans/mock204.json').read_text());result=calibration_analysis.dataset(root/name,p,study,diag);result.pop('ranks',None)
  summaries=json.loads((root/name/'mock204/completed.json').read_text())
  total=sum(r['field_gradient_evaluations_total'] for s in summaries for r in s['replicas'][:-1]);sampling=sum(r['field_gradient_evaluations_sampling'] for s in summaries for r in s['replicas'][:-1]);seconds=json.loads((root/name/'cost.json').read_text())['active_cohort_wall_seconds']
  wall_complete=json.loads((root/'budget.json').read_text())['unrefunded_crashed_batches']==0
  result.update(field_gradient_evaluations_total=total,field_gradient_evaluations_sampling=sampling,active_cohort_wall_seconds=seconds,wall_cost_complete=wall_complete)
  for d in result['diagnostics']:
   d['bulk_ess_per_total_field_gradient']=d['bulk_ess']/total if d['bulk_ess'] is not None else None
   d['bulk_ess_per_active_second']=d['bulk_ess']/seconds if d['bulk_ess'] is not None and wall_complete else None
  outcomes[name]=result;values[name]=np.stack([np.loadtxt(root/name/f'mock204/calibration.chain{e}.csv',delimiter=',',skiprows=1)[:,1:] for e in range(4)])
 # Every replayed baseline record must equal its actual original prefix.
 original=Path(f['original_directory'])
 for e in range(4):
  for j in range(512):
   if (root/f'baseline/mock204/ensemble{e}/draws/draw-{j}.rank-0.json').read_bytes()!=(original/f'ensemble{e}/draws/draw-{j}.rank-0.json').read_bytes():raise ValueError('Counted baseline differs from original prefix')
 report=dict(scope=f['scope'],complete=True,outcomes=outcomes,baseline_all_2048_records_match_original_prefix=True,original_pilot_unchanged=True,calibration_claim=False,limits=['Only 512 retained draws per chain; no extension until pass','ESS estimates and means remain provisional when gates fail','Gradient cost is actual field potential/gradient calls; scalar/cache/FFT overhead enters wall time separately','Cohorts ran concurrently; active wall seconds include startup/replay overhead and resource contention','Common random streams pair designs; cohorts are not independent replications','No independent rank/coverage assessment on this selected dataset'])
 write(root/'comparison-summary.json',report);plot(root,values,outcomes)
 print(json.dumps({k:dict(passed=v['passed'],divergences=v['measured_divergences'],field_gradient_calls=v['field_gradient_evaluations_total'],active_seconds=v['active_cohort_wall_seconds'],failed_coordinates=[d['parameter'] for d in v['diagnostics'] if not d['passed']]) for k,v in outcomes.items()},indent=2))

def plot(root,values,outcomes):
 os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=plt.get_cmap('tab10').colors
 for col,name in enumerate(CASES):
  for e in range(4):
   axes[0,col].plot(values[name][e,:,0],lw=.7,color=colors[e],label=f'Chain {e+1}')
   axes[1,col].plot(values[name][e,:,5],lw=.7,color=colors[e])
  a=outcomes[name];axes[0,col].set(title=f"{name}: {a['measured_divergences']} divergences; gates {'PASS' if a['passed'] else 'FAIL'}",ylabel='sigma8 / reference sigma8');axes[0,col].legend(fontsize=7,ncol=2)
  axes[1,col].set(xlabel='Retained draw',ylabel='cos100 white-field projection')
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.suptitle('Dataset 204 paired actual-chain comparison: 256 warmup +512 retained\nSame target, starts, seeds and integration time; all failed outcomes retained')
 fig.savefig(root/'comparison-overview.png',dpi=150);plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','analyze']);p.add_argument('output',type=Path);p.add_argument('--source',type=Path);p.add_argument('--exe',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-rounds',type=int);a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.max_rounds is not None and a.max_rounds<1:raise ValueError('Positive round limit required')
 with (root/'RUN.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.phase=='prepare':prepare(root,a.source.resolve(),a.exe.resolve(),a.diagnostics.resolve())
  elif a.phase=='run':run(root,a.exe.resolve(),a.diagnostics.resolve(),a.resume,a.max_rounds,lock)
  else:analyze(root,a.exe.resolve(),a.diagnostics.resolve())
if __name__=='__main__':main()
