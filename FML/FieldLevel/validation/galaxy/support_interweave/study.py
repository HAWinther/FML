#!/usr/bin/env python3
"""Fresh exact-support interweaving cohort versus a frozen refined-ladder baseline."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
GALAXY=HERE.parent
sys.path.insert(0,str(GALAXY/'calibration'))
from common import Budget,sha,write
import analyze as calibration_analysis
SOURCES=[*sorted(p for p in HERE.iterdir() if p.suffix in ['.py','.h','.cpp','.json'] or p.name=='Makefile'),
 GALAXY/'interweave/Rescaling.h',GALAXY/'integrator_comparison/FieldHMC.h',
 GALAXY/'tempering/Main.cpp',GALAXY/'tempering/Exchange.h',GALAXY/'tempering/MockSettings.h',GALAXY/'tempering/PriorPredictive.h',
 GALAXY/'tempering/analyze.py',GALAXY/'tempering/check_prior.py',GALAXY/'tempering/plan.json',GALAXY/'tempering/sbc_ranks.py',
 GALAXY/'calibration/analyze.py',GALAXY/'calibration/study.py',GALAXY/'calibration/common.py']

def prepare(root,a):
 if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('Fresh empty output required')
 source=a.source.resolve();protocol=json.loads((HERE/'study.json').read_text());plan=json.loads((HERE/'plan.json').read_text())
 original=json.loads((source/'plan.json').read_text());expected=dict(original,scope=plan['scope'],comparison=plan['comparison'],rescaling=plan['rescaling'],rescaling_policy_sha256=plan['rescaling_policy_sha256'])
 expected_rescaling=dict(original['rescaling'],dimension=18,target='Full positive-beta posterior plus exact global initial-support Jacobian; all 64 white prior coordinates retained')
 if plan!=expected or plan['rescaling']!=expected_rescaling or original['mock_id']!=203 or (original['warmup'],original['samples'],original['ensembles'])!=(256,512,4) or original['field']['step_size']!=.075 or len(original['betas'])!=13:raise ValueError('Undeclared target/kernel/budget changes')
 if protocol['posterior_gates']!=original['posterior_gates']:raise ValueError('Original gates must remain fixed')
 tests=json.loads(a.tests.read_text());controls=json.loads(a.controls.read_text())
 if not tests['passed'] or not controls['passed']:raise ValueError('Kernel/backend and native controls must pass before chains')
 for t in tests['results']:
  if sha(t['executable'])!=t['executable_sha256'] or sha(t['log'])!=t['log_sha256']:raise ValueError('Kernel/backend control artifact changed')
 previous=source.parent.parent/'comparison-summary.json';baseline=json.loads(previous.read_text())['outcomes']['refined']
 if baseline['endpoint_index']!=511 or not baseline['complete']:raise ValueError('Matching complete baseline required')
 files=[previous,source/'plan.json',source/'completed.json',source/'calibration-summary.json',source/'summary.json',a.tests.resolve(),a.controls.resolve()]
 for t in tests['results']:files.extend([Path(t['executable']),Path(t['log'])])
 for e in range(4):
  files += [source/f'ensemble{e}'/n for n in ['CURRENT','summary.json','target.json']]
  files += [source/f'ensemble{e}/draws/draw-{j}.rank-0.json' for j in range(512)]
  files.append(source/f'calibration.chain{e}.csv')
 (root/'candidate/plans').mkdir(parents=True);write(root/'candidate/plans/mock203.json',plan)
 policy=b'FML_SUPPORT_RESCALING_1 active\n'
 (root/'candidate/rescaling-policy.txt').write_bytes(policy)
 if sha(root/'candidate/rescaling-policy.txt')!=plan['rescaling_policy_sha256']:raise ValueError('Policy differs from plan')
 exe=a.exe.resolve();diag=a.diagnostics.resolve()
 frozen=dict(protocol=protocol,binary_sha256=sha(exe),diagnostics_sha256=sha(diag),sources={str(p):sha(p) for p in SOURCES},original={str(p):sha(p) for p in files},original_directory=str(source),plan_sha256=sha(root/'candidate/plans/mock203.json'),baseline_outcome=baseline)
 write(root/'candidate/FROZEN.json',dict(binary_sha256=sha(exe)));write(root/'FROZEN.json',frozen)
 print('Frozen fresh four-chain 256/512 exact-support cohort and complete reused baseline',flush=True)

def verify(root,exe,diag):
 f=json.loads((root/'FROZEN.json').read_text())
 if sha(exe)!=f['binary_sha256'] or sha(diag)!=f['diagnostics_sha256']:raise ValueError('Frozen executable changed')
 for group in ['sources','original']:
  for p,h in f[group].items():
   if sha(p)!=h:raise ValueError('Frozen source/input changed: '+p)
 if sha(root/'candidate/plans/mock203.json')!=f['plan_sha256']:raise ValueError('Frozen plan changed')
 plan=json.loads((root/'candidate/plans/mock203.json').read_text())
 if sha(root/'candidate/rescaling-policy.txt')!=plan['rescaling_policy_sha256']:raise ValueError('Frozen rescaling policy changed')
 return f

def run(root,a,lock):
 f=verify(root,a.exe,a.diagnostics);p=f['protocol'];directory=root/'candidate/mock203';stop=False;child=None
 if not (root/'budget.json').exists() and directory.exists():raise ValueError('Missing budget ledger beside native work')
 budget=Budget(root/'budget.json',p['wall_limit_seconds'],p['batch_reservation_seconds'])
 def pause(sig=None,frame=None):
  nonlocal stop
  stop=True;(root/'PAUSE').touch()
  if directory.exists():(directory/'PAUSE').touch()
  if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
 signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
 if a.resume:
  (root/'PAUSE').unlink(missing_ok=True);(directory/'PAUSE').unlink(missing_ok=True);(root/'paused.json').unlink(missing_ok=True)
 rounds=0;reason=None
 while not (directory/'completed.json').exists():
  if stop or (root/'PAUSE').exists():reason='pause_requested';break
  if a.max_rounds is not None and rounds>=a.max_rounds:reason='bounded_round_limit';break
  if not budget.reserve():reason='sampling_time_budget';break
  start=time.monotonic();cmd=[sys.executable,str(HERE/'run.py'),str(directory),'--exe',str(a.exe),'--plan',str(root/'candidate/plans/mock203.json'),'--workers','4','--max-chunks','1']
  with (root/'candidate.log').open('a') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),));code=child.wait()
  if code:raise RuntimeError('Native batch failed; retain full budget reservation and inspect logs')
  elapsed=time.monotonic()-start;receipt=root/'candidate/cost.json';cost=json.loads(receipt.read_text()) if receipt.exists() else dict(active_cohort_wall_seconds=0.,batches=0)
  cost['active_cohort_wall_seconds']+=elapsed;cost['batches']+=1;write(receipt,cost);budget.finish(elapsed);child=None;rounds+=1
  states=[json.loads((directory/f'ensemble{e}/summary.json').read_text()) for e in range(4)]
  print('candidate',[s['completed_iterations'] for s in states],f'{elapsed:.2f} active seconds',flush=True)
 complete=(directory/'completed.json').exists();write(root/('completed.json' if complete else 'paused.json'),dict(complete=complete,reason=reason,budget=budget.state))
 if complete:(root/'paused.json').unlink(missing_ok=True)
 print('Candidate sampling '+('complete' if complete else 'paused'),flush=True)

def analyze(root,a):
 f=verify(root,a.exe,a.diagnostics)
 if not json.loads((root/'completed.json').read_text())['complete']:raise ValueError('Fixed candidate budget incomplete')
 plan=json.loads((root/'candidate/plans/mock203.json').read_text());candidate=calibration_analysis.dataset(root/'candidate',plan,f['protocol'],a.diagnostics);candidate.pop('ranks',None)
 states=json.loads((root/'candidate/mock203/completed.json').read_text());total=sum(r['field_gradient_evaluations_total'] for s in states for r in s['replicas']);sampling=sum(r['field_gradient_evaluations_sampling'] for s in states for r in s['replicas'])
 seconds=json.loads((root/'candidate/cost.json').read_text())['active_cohort_wall_seconds'];attempts=np.sum([s['swap_attempts'] for s in states],axis=0);accepted=np.sum([s['swap_accepted'] for s in states],axis=0)
 candidate.update(field_gradient_evaluations_total=total,field_gradient_evaluations_sampling=sampling,active_cohort_wall_seconds=seconds,min_pooled_edge_acceptance=float(np.min(accepted/attempts)),wall_cost_complete=json.loads((root/'budget.json').read_text())['unrefunded_crashed_batches']==0)
 for d in candidate['diagnostics']:
  d['bulk_ess_per_total_field_gradient']=d['bulk_ess']/total if d['bulk_ess'] is not None else None
  d['bulk_ess_per_active_second']=d['bulk_ess']/seconds if d['bulk_ess'] is not None and candidate['wall_cost_complete'] else None
 source=Path(f['original_directory']);expected=(source/'ensemble0/target.json').read_bytes()
 for e in range(4):
  if (root/f'candidate/mock203/ensemble{e}/target.json').read_bytes()!=expected:raise ValueError('Candidate target differs from original baseline')
 baseline=dict(f['baseline_outcome'])
 def rescale_acceptance(data):
  cold_a=sum(s['replicas'][0]['rescale_accepted'] for s in data);cold_n=sum(s['replicas'][0]['rescale_attempts'] for s in data)
  all_a=sum(v['rescale_accepted'] for s in data for v in s['replicas']);all_n=sum(v['rescale_attempts'] for s in data for v in s['replicas'])
  return dict(cold=cold_a/cold_n,all_positive=all_a/all_n,cold_accepted=cold_a,cold_attempted=cold_n,all_accepted=all_a,all_attempted=all_n)
 baseline['rescaling_acceptance']=rescale_acceptance(json.loads((source/'completed.json').read_text()))
 candidate['rescaling_acceptance']=rescale_acceptance(states)
 ratios={x['parameter']:y['bulk_ess_per_total_field_gradient']/x['bulk_ess_per_total_field_gradient'] if x['bulk_ess_per_total_field_gradient'] and y['bulk_ess_per_total_field_gradient'] is not None else None for x,y in zip(baseline['diagnostics'],candidate['diagnostics'])}
 r=dict(complete=True,scope=f['protocol']['scope'],outcomes=dict(baseline=baseline,candidate=candidate),bulk_ess_per_gradient_ratios=ratios,all_targets_match_original=True,baseline_records_and_checkpoints_unchanged=True,baseline_reused=True,wall_time_comparable=False,calibration_claim=False,limits=f['protocol']['limits']+['All failed-chain ESS and posterior summaries remain provisional','Cached baseline used eight concurrent native workers across two cohorts; candidate uses four workers. Raw wall times do not certify a speedup','Primary HMC gradient counters include warmup but exclude auxiliary replay; wall seconds include replay'])
 write(root/'comparison-summary.json',r);plot(root,r,source);print(json.dumps({k:dict(passed=v['passed'],failed_coordinates=sum(not d['passed'] for d in v['diagnostics']),divergences=v['measured_divergences'],max_rhat=max((d['rhat'] for d in v['diagnostics'] if d['rhat'] is not None),default=None),min_bulk_ess=min((d['bulk_ess'] for d in v['diagnostics'] if d['bulk_ess'] is not None),default=None)) for k,v in r['outcomes'].items()},indent=2),flush=True)

def plot(root,r,source):
 os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for col,name in enumerate(['baseline','candidate']):
  directory=source if name=='baseline' else root/'candidate/mock203';values=np.stack([np.loadtxt(directory/f'calibration.chain{e}.csv',delimiter=',',skiprows=1)[:,1:] for e in range(4)])
  for e in range(4):axes[0,col].plot(values[e,:,0],lw=.6,label=f'Chain {e+1}');axes[1,col].plot(values[e,:,10],lw=.6)
  axes[0,col].set(title=name+': '+('PASS' if r['outcomes'][name]['passed'] else 'FAIL'),ylabel='sigma8 / reference sigma8');axes[0,col].legend(fontsize=7,ncol=2)
  axes[1,col].set(xlabel='Retained draw',ylabel='sin001 white-field projection')
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.suptitle('Exact initial-support amplitude/field interweaving: four chains, 256 warmup +512 retained\nSame target, refined ladder and half-step HMC; cached baseline; all failures preserved')
 fig.savefig(root/'support-comparison.png',dpi=150);plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','analyze']);p.add_argument('output',type=Path)
 for n in ['source','tests','controls','exe','diagnostics']:p.add_argument('--'+n,type=Path,required=n in ['exe','diagnostics'])
 p.add_argument('--resume',action='store_true');p.add_argument('--max-rounds',type=int);a=p.parse_args()
 if a.phase=='prepare' and any(getattr(a,n) is None for n in ['source','tests','controls']):p.error('prepare needs source baseline, kernel/backend tests and passed native controls')
 if a.max_rounds is not None and a.max_rounds<1:p.error('max-rounds must be positive')
 a.exe=a.exe.resolve();a.diagnostics=a.diagnostics.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 with (root/'RUN.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.phase=='prepare':prepare(root,a)
  elif a.phase=='run':run(root,a,lock)
  else:analyze(root,a)
if __name__=='__main__':main()
