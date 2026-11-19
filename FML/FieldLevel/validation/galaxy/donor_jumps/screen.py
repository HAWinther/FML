#!/usr/bin/env python3
"""One frozen in-sample joint-donor feasibility screen; no posterior chains."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,shutil,subprocess,time
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def prepare(root,a):
 if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('New empty screen directory required')
 source=a.source.resolve();plan=json.loads((HERE/'screen_plan.json').read_text())
 if 'All fixed-donor joint jumps tests passed' not in a.unit_log.read_text():raise ValueError('Analytic unit controls must pass first')
 files=[source/'plan.json',source/'completed.json',source/'ensemble0/target.json',a.unit_log.resolve()];target=json.loads((source/'ensemble0/target.json').read_text());frames=[]
 if target['mock']['id']!=203:raise ValueError('Dataset 203 required')
 for e in range(4):
  files.append(source/f'ensemble{e}/CURRENT')
  for draw in [0,255,511]:
   path=source/f'ensemble{e}/draws/draw-{draw}.rank-0.json';files.append(path);r=json.loads(path.read_text())
   if r['draw']!=draw or len(r['white'])!=64 or len(r['parameters'])!=5:raise ValueError('Donor layout mismatch')
   frames.append((e,draw,r))
 with (root/'donor-pool.txt').open('w') as stream:
  stream.write('FML_FIXED_DONORS_1 1 64 5 12\n')
  for e,d,r in frames:stream.write(f'{e} {d} '+' '.join(format(x,'.17g') for x in [*r['white'],*r['parameters']])+'\n')
 with (root/'screen-input.txt').open('w') as stream:
  stream.write('FML_DONOR_SCREEN_1 12 17 64\n')
  for z in target['data']:stream.write(' '.join(format(x,'.17g') for x in z)+'\n')
  for e,d,r in frames:stream.write(f'{e} {d} '+format(r['quadratic'][0],'.17g')+' '+' '.join(format(x,'.17g') for z in r['prediction'] for x in z)+'\n')
 shutil.copyfile(HERE/'screen_plan.json',root/'plan.json')
 sources=[p for p in HERE.iterdir() if p.suffix in ['.py','.cpp','.h','.json'] or p.name=='Makefile']
 sources += [HERE.parent/'support_interweave/Support.h',HERE.parent/'support_interweave/Native.h',HERE.parent/'support_interweave/make_driver.py',HERE.parent/'gauss_newton/Fixture.h',HERE.parent/'tempering/Main.cpp']
 f=dict(scope=plan['scope'],original={str(p):sha(p) for p in files},sources={str(p):sha(p) for p in sources},source_directory=str(source),executables={n:dict(path=str(getattr(a,n).resolve()),sha256=sha(getattr(a,n))) for n in ['exe','unit_exe']},inputs={n:sha(root/n) for n in ['donor-pool.txt','screen-input.txt','plan.json']})
 write(root/'FROZEN.json',f);print('Frozen twelve donors and 768 joint proposals; no chains',flush=True)
def verify(root):
 f=json.loads((root/'FROZEN.json').read_text())
 for group in ['sources','original']:
  for p,h in f[group].items():
   if sha(p)!=h:raise ValueError('Frozen source/input changed: '+p)
 for name,h in f['inputs'].items():
  if sha(root/name)!=h:raise ValueError('Screen input changed')
 for e in f['executables'].values():
  if sha(e['path'])!=e['sha256']:raise ValueError('Frozen executable changed')
 return f

def probe(root):
 f=verify(root)
 if (root/'proposals.csv').exists():raise ValueError('Preserve existing screen results')
 start=time.monotonic()
 with (root/'probe.log').open('w') as log:subprocess.run([f['executables']['exe']['path'],str(root/'donor-pool.txt'),str(root/'screen-input.txt'),str(root/'proposals.csv')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 write(root/'completed.json',dict(complete=True,seconds=time.monotonic()-start,proposals_sha256=sha(root/'proposals.csv')))
def analyze(root):
 verify(root);plan=json.loads((root/'plan.json').read_text());completed=json.loads((root/'completed.json').read_text());rows=list(csv.DictReader((root/'proposals.csv').open()))
 if not completed['complete'] or sha(root/'proposals.csv')!=completed['proposals_sha256'] or len(rows)!=768:raise ValueError('Incomplete/changed screen')
 if {(int(r['frame']),int(r['trial'])) for r in rows}!={(f,t) for f in range(12) for t in range(64)}:raise ValueError('Duplicate/missing frame trials')
 probability=np.array([float(r['acceptance_probability']) for r in rows]);physical=np.array([float(r['physical_jump_squared']) for r in rows]);cross=np.array([r['cross_ensemble_pair']=='1' and float(r['scale'])==1 for r in rows])
 if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ValueError('Invalid acceptance probabilities')
 mean=float(probability.mean());crossmean=float(probability[cross].mean());moving=float(np.mean(probability*(physical>plan['physical_jump_threshold'])));g=plan['gates']
 passed=mean>=g['mean_acceptance_probability_min'] and crossmean>=g['cross_ensemble_large_jump_mean_acceptance_probability_min'] and moving>=g['mean_acceptance_probability_for_physical_jump_gt_threshold_min']
 r=dict(scope=plan['scope'],complete=True,screen_passed=bool(passed),proposals=len(rows),mean_acceptance_probability=mean,cross_ensemble_large_jump_mean_acceptance_probability=crossmean,physical_jump_expected_acceptance_rate=moving,support_rejection_fraction=float(np.mean([int(r['support_rejection']) for r in rows])),actual_value_calls=sum(int(r['value_calls']) for r in rows),original_inputs_unchanged=True,chain_mixing_claim=False,calibration_claim=False,limits=plan['limits'])
 write(root/'summary.json',r);print(json.dumps(r,indent=2),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','probe','analyze']);p.add_argument('output',type=Path)
 for n in ['source','exe','unit-exe','unit-log']:p.add_argument('--'+n,dest=n.replace('-','_'),type=Path)
 a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.phase=='prepare' and any(getattr(a,n) is None for n in ['source','exe','unit_exe','unit_log']):p.error('prepare requires source, screen/unit binaries and passed unit log')
 with (root/'RUN.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.phase=='prepare':prepare(root,a)
  elif a.phase=='probe':probe(root)
  else:analyze(root)
if __name__=='__main__':main()
