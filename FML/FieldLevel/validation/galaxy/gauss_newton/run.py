#!/usr/bin/env python3
"""Freeze, construct and probe one low-rank nonlinear field metric; no posterior chains."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,shutil,subprocess,time
import numpy as np
HERE=Path(__file__).resolve().parent
CASES=['baseline','gauss_newton']
PROJECTIONS=['cos100','sin100','cos110','sin110','cos001','sin001']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sources():return [p for p in HERE.iterdir() if p.suffix in ['.cpp','.h','.py','.json'] or p.name=='Makefile']
def prepare(root,args):
 if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('Fresh empty output required')
 source=args.source.resolve();old=source.parent.parent/'FROZEN.json';identity=json.loads(old.read_text())
 if source.name!='mock203.txt' or sha(source)!=identity['inputs'][source.name]:raise ValueError('Original frozen fixture mismatch')
 original={p:h for p,h in identity['pilot_sources'].items() if '/mock203/' in p}
 original.update({str(source):sha(source),str(old):sha(old),str(args.unit_log.resolve()):sha(args.unit_log)})
 if 'PASS:' not in args.unit_log.read_text():raise ValueError('Independent native unit controls must pass before freeze')
 for p,h in original.items():
  if sha(p)!=h:raise ValueError('Original source changed: '+p)
 executables={name:dict(path=str(getattr(args,name).resolve()),sha256=sha(getattr(args,name))) for name in ['builder','probe','unit']}
 shutil.copyfile(source,root/'fixture.txt');shutil.copyfile(HERE/'plan.json',root/'plan.json')
 frozen=dict(scope=json.loads((root/'plan.json').read_text())['scope'],sources={str(p):sha(p) for p in sources()},original=original,executables=executables,fixture_sha256=sha(root/'fixture.txt'),plan_sha256=sha(root/'plan.json'))
 write(root/'FROZEN.json',frozen);print('Frozen one metric candidate and 192 conditional-field trajectories',flush=True)
def verify(root):
 f=json.loads((root/'FROZEN.json').read_text())
 for group in ['sources','original']:
  for p,h in f[group].items():
   if sha(p)!=h:raise ValueError('Frozen source/input changed: '+p)
 for e in f['executables'].values():
  if sha(e['path'])!=e['sha256']:raise ValueError('Frozen executable changed')
 for key,name in [('fixture_sha256','fixture.txt'),('plan_sha256','plan.json')]:
  if sha(root/name)!=f[key]:raise ValueError('Frozen study input changed')
 return f

def native(root,executable,arguments,label):
 start=time.monotonic()
 with (root/(label+'.log')).open('w') as log:subprocess.run([executable,*map(str,arguments)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,env=dict(os.environ,OMP_NUM_THREADS='1'))
 return time.monotonic()-start

def construct(root):
 f=verify(root);plan=json.loads((root/'plan.json').read_text())
 if (root/'metrics.json').exists():raise ValueError('Do not overwrite a completed construction')
 elapsed=native(root,f['executables']['builder']['path'],[root/'fixture.txt',root/'jacobian'],'build')
 checks=list(csv.DictReader((root/'jacobian.checks.csv').open()));rows=list(csv.DictReader((root/'jacobian.rows.csv').open()))
 if len(checks)!=12:raise ValueError('All twelve reference controls required')
 for r in checks:
  if max(float(r['fd_1e4']),float(r['fd_1e5']))>plan['operator_gates']['max_directional_jacobian_relative_error'] or float(r['gradient_reconstruction_error'])>plan['operator_gates']['max_gradient_reconstruction_relative_error']:raise ValueError('Jacobian controls fail')
 (root/'metrics').mkdir();metrics=[]
 for e in range(4):
  blocks=[]
  for draw in plan['reference_draws']:
   group=[r for r in rows if int(r['ensemble'])==e and int(r['draw'])==draw]
   if len(group)!=34 or [int(r['row']) for r in group]!=list(range(34)):raise ValueError('Unexpected bounded Jacobian layout')
   blocks.append(np.array([[float(r['x'+str(j)]) for j in range(64)] for r in group]))
  # Dense SVD is confined to this 64-coordinate reference construction. The applied
  # operator stores only retained vectors and has a fixed rank cap, not a D-by-D mass.
  stacked=np.concatenate(blocks)/np.sqrt(len(blocks));_,singular,vt=np.linalg.svd(stacked,full_matrices=False)
  eigenvalues=singular**2;keep=eigenvalues>plan['relative_eigenvalue_threshold']*eigenvalues[0];lam=eigenvalues[keep];basis=vt[keep]
  if not 0<len(lam)<=plan['rank_cap']:raise ValueError('Rank exceeds predeclared cap')
  orth=float(np.max(np.abs(basis@basis.T-np.eye(len(lam)))))
  error=float(np.linalg.norm(stacked.T@stacked-(basis.T*lam)@basis)/max(1,np.linalg.norm(stacked.T@stacked)))
  if orth>plan['operator_gates']['max_metric_orthonormality_error'] or error>1e-9:raise ValueError('Low-rank construction control fails')
  path=root/'metrics'/f'ensemble{e}.txt'
  with path.open('w') as stream:
   stream.write(f'FML_FROZEN_GN_1 64 {len(lam)}\n')
   for value,vector in zip(lam,basis):stream.write(' '.join(format(x,'.17g') for x in [value,*vector])+'\n')
  metrics.append(dict(ensemble=e,rank=len(lam),excess_eigenvalues=lam.tolist(),orthonormality_error=orth,relative_reconstruction_error=error,file=str(path),sha256=sha(path)))
 write(root/'metrics.json',dict(scope=plan['metric'],metrics=metrics,builder_seconds=elapsed,jacobian_rows_sha256=sha(root/'jacobian.rows.csv'),checks_sha256=sha(root/'jacobian.checks.csv')))
 print('Constructed four fixed low-rank metrics; all independent Jacobian controls pass',flush=True)

def probe(root):
 f=verify(root);m=json.loads((root/'metrics.json').read_text())
 for metric in m['metrics']:
  if sha(metric['file'])!=metric['sha256']:raise ValueError('Frozen metric changed')
 if (root/'trajectories.csv').exists():raise ValueError('Do not overwrite existing trajectories')
 elapsed=native(root,f['executables']['probe']['path'],[root/'fixture.txt',root/'metrics',root/'trajectories.csv'],'probe')
 write(root/'completed.json',dict(complete=True,probe_seconds=elapsed,trajectories_sha256=sha(root/'trajectories.csv'),metrics_sha256=sha(root/'metrics.json')))
 print('Completed exact fixed 192-trajectory budget',flush=True)

def analyze(root):
 verify(root);plan=json.loads((root/'plan.json').read_text());completed=json.loads((root/'completed.json').read_text());m=json.loads((root/'metrics.json').read_text())
 if sha(root/'trajectories.csv')!=completed['trajectories_sha256'] or sha(root/'metrics.json')!=completed['metrics_sha256']:raise ValueError('Completed artifacts changed')
 for metric in m['metrics']:
  if sha(metric['file'])!=metric['sha256']:raise ValueError('Frozen metric changed')
 rows=list(csv.DictReader((root/'trajectories.csv').open()))
 expected={(e,d,t,c) for e in range(4) for d in plan['reference_draws'] for t in range(8) for c in CASES}
 actual=[(int(r['ensemble']),int(r['draw']),int(r['trial']),r['case']) for r in rows]
 if len(rows)!=192 or len(set(actual))!=192 or set(actual)!=expected:raise ValueError('Incomplete/duplicate probe budget')
 outcomes={}
 for name in CASES:
  group=[r for r in rows if r['case']==name];calls=sum(int(r['gradient_calls']) for r in group)
  if any(not 2<=int(r['gradient_calls'])<=17 for r in group):raise ValueError('Invalid actual gradient count')
  jumps=np.array([[float(r[p]) for p in PROJECTIONS] for r in group]);acceptance=np.array([float(r['acceptance']) for r in group])
  if not np.isfinite(jumps).all() or (jumps<0).any() or not np.isfinite(acceptance).all():raise ValueError('Nonfinite jump or acceptance')
  frame_acceptance=[dict(ensemble=e,draw=d,mean=float(np.mean([float(r['acceptance']) for r in group if int(r['ensemble'])==e and int(r['draw'])==d]))) for e in range(4) for d in plan['reference_draws']]
  outcomes[name]=dict(trajectories=len(group),gradient_calls=calls,divergences=sum(int(r['divergences']) for r in group),mean_acceptance=float(acceptance.mean()),minimum_frame_mean_acceptance=min(r['mean'] for r in frame_acceptance),frame_acceptance=frame_acceptance,projection_mean_squared_jump=dict(zip(PROJECTIONS,jumps.mean(axis=0).tolist())),projection_squared_jump_per_gradient=dict(zip(PROJECTIONS,(jumps.sum(axis=0)/calls).tolist())))
 ratios={p:outcomes['gauss_newton']['projection_squared_jump_per_gradient'][p]/outcomes['baseline']['projection_squared_jump_per_gradient'][p] if outcomes['baseline']['projection_squared_jump_per_gradient'][p]>0 else None for p in PROJECTIONS}
 valid=all(x is not None and np.isfinite(x) and x>0 for x in ratios.values());geometric=float(np.exp(np.log(list(ratios.values())).mean())) if valid else None
 candidate=outcomes['gauss_newton'];g=plan['local_screen']
 passed=bool(valid and candidate['divergences']==g['candidate_divergences'] and candidate['mean_acceptance']>=g['minimum_mean_acceptance'] and candidate['minimum_frame_mean_acceptance']>=g['minimum_frame_mean_acceptance'] and geometric>g['minimum_geometric_mean_projection_jump_per_gradient_ratio'] and min(ratios.values())>=g['minimum_individual_projection_jump_per_gradient_ratio'])
 checks=list(csv.DictReader((root/'jacobian.checks.csv').open()))
 result=dict(scope=plan['scope'],complete=True,operator_controls_passed=True,local_screen_passed=passed,outcomes=outcomes,projection_jump_per_gradient_ratios=ratios,geometric_mean_ratio=geometric,metric_ranks=[v['rank'] for v in m['metrics']],maximum_directional_jacobian_error=max(float(r[k]) for r in checks for k in ['fd_1e4','fd_1e5']),maximum_gradient_reconstruction_error=max(float(r['gradient_reconstruction_error']) for r in checks),full_chain_mixing_claim=False,calibration_claim=False,original_inputs_unchanged=True,limits=plan['limits'])
 write(root/'summary.json',result);plot(root,result);print(json.dumps(result,indent=2),flush=True)

def plot(root,r):
 os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for name in CASES:
  data=r['outcomes'][name];axes[0].plot(range(12),[v['mean'] for v in data['frame_acceptance']],'o-',label=name)
 axes[0].set(xlabel='Fixed source frame (ensemble × draw)',ylabel='Mean acceptance over eight momenta',title='Actual conditional-field HMC, step 0.075 ×16');axes[0].legend()
 axes[1].bar(PROJECTIONS,[r['projection_jump_per_gradient_ratios'][p] for p in PROJECTIONS]);axes[1].axhline(1,color='black',ls='--');axes[1].set(ylabel='GN / baseline squared jump per gradient',title='Accepted six-projection jumps; not chain ESS');axes[1].tick_params(axis='x',rotation=35)
 for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('Frozen nonlinear low-rank metric: local screen '+('PASS' if r['local_screen_passed'] else 'FAIL')+'\nTwelve reused states; no new posterior chains or calibration claim')
 fig.savefig(root/'metric-probe.png',dpi=150);plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','construct','probe','analyze']);p.add_argument('output',type=Path)
 for name in ['source','builder','probe','unit','unit-log']:p.add_argument('--'+name,type=Path,dest=name.replace('-','_'))
 a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.phase=='prepare' and any(getattr(a,n) is None for n in ['source','builder','probe','unit','unit_log']):p.error('prepare requires fixture, three executables and passed unit log')
 with (root/'RUN.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.phase=='prepare':prepare(root,a)
  elif a.phase=='construct':construct(root)
  elif a.phase=='probe':probe(root)
  else:analyze(root)
if __name__=='__main__':main()
