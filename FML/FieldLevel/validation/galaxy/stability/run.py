#!/usr/bin/env python3
"""Freeze saved pilot states and run a fixed-budget conditional stability diagnostic."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,subprocess,time
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def source_paths(source,plan):
    paths=[source/'FROZEN.json',source/'study.json',source/'calibration-summary.json',source/'completed.json']
    for mid in plan['mocks']:
        d=source/f'mock{mid}';paths += [d/'plan.json',d/'calibration-summary.json']
        for e in range(plan['ensembles']):
            paths += [d/f'ensemble{e}/target.json',d/f'ensemble{e}/CURRENT']
            paths += [d/f'ensemble{e}/draws/draw-{j}.rank-0.json' for j in plan['draws']]
    return paths

def prepare(root,source,planpath,exe):
    if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('New probe requires empty directory')
    plan=json.loads(planpath.read_text())
    if plan['mocks']!=list(range(201,209)) or plan['draws']!=[0,1024,2047] or plan['ensembles']!=4 or plan['momentum_trials']!=8:raise ValueError('Bounded input/seed budget changed')
    if any(abs(c['step']*c['leaps']-plan['trajectory_length'])>1e-14 for c in plan['cases']):raise ValueError('Trajectory lengths differ')
    (root/'plan.json').write_bytes(planpath.read_bytes());(root/'inputs').mkdir()
    if not json.loads((source/'calibration-summary.json').read_text())['all_completed']:raise ValueError('Completed pilot required')
    originals={str(p):sha(p) for p in source_paths(source,plan)}
    summaries=[];starts=None
    for mid in plan['mocks']:
        d=source/f'mock{mid}';p=json.loads((d/'plan.json').read_text());target=json.loads((d/'ensemble0/target.json').read_text());starts=p['initial_parameters']
        if p['truth_mode']!='prior_predictive_v1' or p['samples']!=2048 or p['warmup']!=256:raise ValueError('Unexpected source target')
        lines=[f"FML_STABILITY_1 {mid} {len(target['data'])} {4*len(plan['draws'])} {plan['momentum_trials']} {len(plan['cases'])}"]
        lines += [f"{c['name']} {c['current']} {c['step']:.17g} {c['leaps']}" for c in plan['cases']]
        lines += [' '.join(format(v,'.17g') for v in mode) for mode in target['data']]
        for e in range(4):
            if (d/f'ensemble{e}/target.json').read_bytes()!=(d/'ensemble0/target.json').read_bytes():raise ValueError('Different source targets')
            for draw in plan['draws']:
                row=json.loads((d/f'ensemble{e}/draws/draw-{draw}.rank-0.json').read_text())
                if row['draw']!=draw or row['rank']!=0:raise ValueError('Invalid source draw')
                lines += [f'{e} {draw}', ' '.join(format(v,'.17g') for v in starts[e]),' '.join(format(v,'.17g') for v in row['parameters']),' '.join(format(v,'.17g') for v in row['white'])]
                lines += [' '.join(format(v,'.17g') for v in pair) for pair in row['prediction']]
                lines += [format(row['quadratic'][0],'.17g')]
        (root/f'inputs/mock{mid}.txt').write_text('\n'.join(lines)+'\n')
        summary=json.loads((d/'calibration-summary.json').read_text())
        summaries.append(dict(id=mid,passed=summary['passed'],diagnostics=summary['diagnostics'],transport_passed=summary['transport_passed'],measured_divergences=summary['measured_divergences'],roundtrips=summary['roundtrips']))
    for name,h in originals.items():
        if sha(name)!=h:raise ValueError('Pilot changed during input preparation')
    write(root/'pilot-outcomes.json',summaries)
    sourcefiles=[*HERE.glob('*.cpp'),*HERE.glob('*.h'),*HERE.glob('*.py'),HERE/'Makefile']
    identity=dict(plan_sha256=sha(root/'plan.json'),binary_sha256=sha(exe),sources={str(p):sha(p) for p in sourcefiles},pilot_sources=originals,inputs={p.name:sha(p) for p in (root/'inputs').iterdir()},status='Frozen before evaluating any comparison trajectories')
    write(root/'FROZEN.json',identity)
    print('Frozen 96 states, five cases and eight paired momentum trials; no comparison trajectories run')

def verify(root,exe):
    f=json.loads((root/'FROZEN.json').read_text())
    if sha(root/'plan.json')!=f['plan_sha256'] or sha(exe)!=f['binary_sha256']:raise ValueError('Frozen plan or executable changed')
    for group in ['sources','pilot_sources']:
        for name,h in f[group].items():
            if sha(name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in f['inputs'].items():
        if sha(root/'inputs'/name)!=h:raise ValueError('Frozen input changed')
    return json.loads((root/'plan.json').read_text())

def run(root,exe):
    plan=verify(root,exe);(root/'results').mkdir(exist_ok=True)
    def one(mid):
        prefix=root/f'results/mock{mid}';receipt=prefix.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['sha256']!={p.name:sha(p) for p in [prefix.with_suffix('.csv'),prefix.with_suffix('.checks.csv')]}:raise ValueError('Completed result changed')
            return old
        if prefix.with_suffix('.csv').exists() or prefix.with_suffix('.checks.csv').exists():raise ValueError('Partial result retained; inspect before retrying')
        start=time.monotonic();cmd=[str(exe),str(root/f'inputs/mock{mid}.txt'),str(prefix)]
        with prefix.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=plan['timeout_per_dataset'],env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive'))
        if r.returncode:raise RuntimeError(f'Probe {mid} failed; retain inputs/partial output and inspect log')
        record=dict(mock=mid,seconds=time.monotonic()-start,argv=cmd,sha256={p.name:sha(p) for p in [prefix.with_suffix('.csv'),prefix.with_suffix('.checks.csv')]})
        write(receipt,record);print('Completed fixed probe',mid,flush=True);return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=list(pool.map(one,plan['mocks']))
    verify(root,exe);write(root/'completed.json',dict(complete=True,datasets=results,source_pilot_unchanged=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run']);p.add_argument('output',type=Path);p.add_argument('--source',type=Path);p.add_argument('--plan',type=Path,default=HERE/'plan.json');p.add_argument('--exe',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='prepare':
            if a.source is None:raise ValueError('Source pilot required')
            prepare(root,a.source.resolve(),a.plan,a.exe.resolve())
        else:run(root,a.exe.resolve())
if __name__=='__main__':main()
