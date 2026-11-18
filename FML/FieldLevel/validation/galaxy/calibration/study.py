#!/usr/bin/env python3
"""Freeze and run a bounded prior-predictive study; no replacement or extension."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import numpy as np
from common import Budget,sha,write

HERE=Path(__file__).resolve().parent
TEMPERING=HERE.parent/'tempering'
SOURCE_FILES=[HERE/n for n in ['study.py','analyze.py','common.py','Diagnostics.cpp']]+[TEMPERING/n for n in ['run.py','analyze.py','check_prior.py','sbc_ranks.py','plan_mock1.json','plan.json']]

def plans(study):
    template=json.loads((TEMPERING/'plan_mock1.json').read_text())
    ints=[study[k] for k in ['warmup','samples','endpoint_index','analysis_seed','workers']]
    if any(type(v)!=int for v in ints) or study['workers']!=4 or study['warmup']<0 or study['warmup']%16 or study['samples']<48 or study['samples']%16 or study['warmup']+study['samples']>1000000 or study['endpoint_index']!=study['samples']-1 or not 1<=study['analysis_seed']<=2**32-1:
        raise ValueError('Unsupported frozen schedule/rank/worker controls')
    if study['role'] not in ['control_smoke','nonlinear_pilot']:raise ValueError('Unknown study role')
    if study['posterior_gates']!=template['posterior_gates']:raise ValueError('Do not weaken existing mixing gates')
    if study['interval_probabilities']!=[.5,.9,.95]:raise ValueError('Unsupported interval controls')
    limit=study['sampling_wall_seconds'];reservation=study['batch_reservation_seconds']
    if any(type(v)!=int or v<=0 for v in [limit,reservation]) or reservation!=2400 or limit<reservation:raise ValueError('Positive wall budget and fixed 2400-second batch reservation required')
    if study['role']=='nonlinear_pilot' and (study['warmup'],study['samples'],len(study['mocks']))!=(256,2048,8):raise ValueError('Nonlinear pilot requires eight fixed 256/2048 datasets')
    result=[];ids=set();seeds=set()
    for item in study['mocks']:
        if set(item)!={'id','truth_seed','noise_seed','initial_seed_base','scalar_truth_seed'} or any(type(v)!=int for v in item.values()):raise ValueError('Exact integer mock seed controls required')
        mid=item['id'];raw=[item['truth_seed'],item['noise_seed'],item['scalar_truth_seed'],*range(item['initial_seed_base'],item['initial_seed_base']+4)]
        if not 3<=mid<=65535 or mid in ids or 101<=mid<=108 or min(raw)<1 or max(raw)>2**32-1 or len(set(raw))!=7 or seeds.intersection(raw):raise ValueError('Overlapping/reserved mock IDs or generation seeds')
        ids.add(mid);seeds.update(raw)
        p=dict(template,scope=study['scope']+f'; dataset {mid}',warmup=study['warmup'],samples=study['samples'],truth_mode='prior_predictive_v1',truth_parameters=None,mock_id=mid,**{k:v for k,v in item.items() if k!='id'})
        result.append(p)
    if not result or study['analysis_seed'] in seeds:raise ValueError('Empty study or reused analysis seed')
    return result

def key_audit(selected):
    """Enumerate each disjoint mock/ensemble domain without storing all keys at once."""
    digest=hashlib.sha256();total=0;generation=set()
    for p in selected:
        mid=p['mock_id'];sweep=np.arange(p['warmup']+p['samples'],dtype=np.uint64)
        for role,raws in [(0,[p['truth_seed']]),(1,[p['scalar_truth_seed']]),(2,[p['noise_seed']]),(3,range(p['initial_seed_base'],p['initial_seed_base']+4))]:
            for raw in raws:
                key=(0x54<<56)|(role<<48)|(mid<<32)|raw
                if key in generation:raise ValueError('Generation key collision')
                generation.add(key);digest.update(key.to_bytes(8,'little'))
        for e in range(4):
            parts=[]
            for r in range(len(p['betas'])):
                for purpose in range(2 if r==len(p['betas'])-1 else 3):
                    parts.append(np.uint64((0x53<<56)|(mid<<32)|(e<<30)|(r<<24)|purpose)|(sweep<<np.uint64(4)))
            parts.append(np.uint64((0x53<<56)|(mid<<32)|(e<<30)|(63<<24)|3)|(sweep<<np.uint64(4)))
            values=np.concatenate(parts)
            if np.unique(values).size!=values.size or not np.all((values>>np.uint64(32))==((0x53<<24)|mid)):
                raise ValueError('Sampler key collision/domain overflow')
            total+=values.size;digest.update(values.astype('<u8').tobytes())
    # Distinct IDs/ensemble bits partition all arrays; 0x54 generation != 0x53 sampling.
    return dict(sampler_exchange_keys=total,generation_keys=len(generation),sha256=digest.hexdigest(),cross_domain_disjoint=True)

def expected_identity(study_path,exe,diagnostics):
    return dict(study_sha256=sha(study_path),binary_sha256=sha(exe),diagnostics_sha256=sha(diagnostics),
                sources={str(p.relative_to(HERE.parent)):sha(p) for p in SOURCE_FILES})

def prepare(root,study_path,exe,diagnostics):
    if any(p.name!='RUN.lock' for p in root.iterdir()):raise ValueError('New study needs empty output directory')
    study=json.loads(study_path.read_text());selected=plans(study);audit=key_audit(selected)
    (root/'study.json').write_bytes(study_path.read_bytes());(root/'plans').mkdir()
    for p in selected:write(root/f'plans/mock{p["mock_id"]}.json',p)
    identity=expected_identity(root/'study.json',exe,diagnostics)
    identity['plans']={p.name:sha(p) for p in sorted((root/'plans').glob('*.json'))}
    identity['key_audit']=audit
    write(root/'FROZEN.json',identity)
    print(json.dumps(dict(frozen=True,data_generated=False,key_audit=audit),indent=2))

def verify(root,exe,diagnostics):
    frozen=json.loads((root/'FROZEN.json').read_text());expected=expected_identity(root/'study.json',exe,diagnostics)
    if any(frozen[k]!=v for k,v in expected.items()):raise ValueError('Frozen study, implementation or binary changed')
    selected=plans(json.loads((root/'study.json').read_text()))
    expected_names={f'mock{p["mock_id"]}.json' for p in selected}
    if set(frozen['plans'])!=expected_names or {p.name for p in (root/'plans').iterdir()}!=expected_names:raise ValueError('Frozen mock set changed')
    for p in selected:
        path=root/f'plans/mock{p["mock_id"]}.json'
        if sha(path)!=frozen['plans'][path.name] or json.loads(path.read_text())!=p:raise ValueError('Frozen mock plan changed')
    return selected

def run(root,exe,diagnostics,resume,max_batches,lock):
    selected=verify(root,exe,diagnostics);study=json.loads((root/'study.json').read_text())
    if not (root/'budget.json').exists() and any((root/f'mock{p["mock_id"]}').exists() for p in selected):
        raise ValueError('Missing time ledger for existing native work; refuse to reset budget')
    budget=Budget(root/'budget.json',study['sampling_wall_seconds'],study['batch_reservation_seconds'])
    stopping=False;child=None;child_pause_sent=False
    def pause(sig=None,frame=None):
        nonlocal stopping,child_pause_sent
        stopping=True;(root/'PAUSE').touch()
        for p in selected:
            directory=root/f'mock{p["mock_id"]}'
            if directory.exists():(directory/'PAUSE').touch()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM);child_pause_sent=True
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    if resume:
        (root/'PAUSE').unlink(missing_ok=True)
        for p in selected:(root/f'mock{p["mock_id"]}/PAUSE').unlink(missing_ok=True)
    batches=0;reason=None
    for p in selected:
        directory=root/f'mock{p["mock_id"]}'
        while not (directory/'completed.json').exists():
            if stopping or (root/'PAUSE').exists():reason='pause_requested';break
            if max_batches is not None and batches>=max_batches:reason='bounded_batch_limit';break
            if not budget.reserve():reason='sampling_time_budget';break
            start=time.monotonic();child_pause_sent=False
            cmd=[sys.executable,str(TEMPERING/'run.py'),str(directory),'--exe',str(exe),'--plan',str(root/f'plans/mock{p["mock_id"]}.json'),'--workers','4','--max-chunks','1']
            try:
                with (root/f'mock{p["mock_id"]}.log').open('a') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                    while True:
                        try:code=child.wait(timeout=1);break
                        except subprocess.TimeoutExpired:
                            if (root/'PAUSE').exists() and not child_pause_sent:pause()
                if code:raise RuntimeError(f'Mock {p["mock_id"]} runner failed; retain files and inspect log')
            finally:
                budget.finish(time.monotonic()-start);child=None
            batches+=1
            progress=[json.loads(f.read_text())['completed_iterations'] for f in sorted(directory.glob('ensemble*/summary.json'))]
            print(f'Mock {p["mock_id"]}: sweeps {progress}; sampling seconds {budget.state["spent"]:.1f}',flush=True)
        if reason:break
    complete=all((root/f'mock{p["mock_id"]}/completed.json').exists() for p in selected)
    receipt=dict(complete=complete,reason=reason,sampling_budget=budget.state,mocks=[dict(id=p['mock_id'],complete=(root/f'mock{p["mock_id"]}/completed.json').exists()) for p in selected])
    write(root/('completed.json' if complete else 'paused.json'),receipt)
    if complete:(root/'paused.json').unlink(missing_ok=True)
    print(json.dumps(receipt,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','run']);parser.add_argument('output',type=Path)
    parser.add_argument('--exe',type=Path,required=True);parser.add_argument('--diagnostics',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=HERE/'plan.json');parser.add_argument('--max-batches',type=int);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.max_batches is not None and args.max_batches<1:raise ValueError('Positive batch limit required')
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='prepare':prepare(root,args.plan,args.exe.resolve(),args.diagnostics.resolve())
        else:run(root,args.exe.resolve(),args.diagnostics.resolve(),args.resume,args.max_batches,lock)

if __name__=='__main__':main()
