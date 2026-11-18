#!/usr/bin/env python3
"""Run the predeclared independent-mock stress test; no replacements or extra draws."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,signal,subprocess,sys,threading

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan_mocks.json'));ap.add_argument('--plan-dir',type=Path,default=Path(__file__).parent);ap.add_argument('--max-chunks',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.max_chunks is not None and args.max_chunks<1:raise RuntimeError('Positive chunk budget required')
    root=args.output.resolve();exe=args.exe.resolve();planpath=args.plan.resolve();plandir=args.plan_dir.resolve();study=json.loads(planpath.read_text());plans=[];used=set();ids=set()
    for item in study['mocks']:
        path=plandir/item['plan'];p=json.loads(path.read_text());mid=item['id']
        if mid in ids or p.get('mock_id')!=mid or p.get('seed_layout')!=3 or 'prefix' in p:raise RuntimeError('Distinct fresh layout-3 mock campaigns required')
        ids.add(mid)
        for key,expected in [('truth_parameters',study['truth_parameters']),('ensembles',study['ensembles_per_mock']),('warmup',study['warmup']),('samples',study['samples']),('posterior_gates',study['posterior_gates'])]:
            if p.get(key)!=expected:raise RuntimeError('Mock controls differ from study: '+key)
        seeds=[p['truth_seed'],p['noise_seed'],*range(p['initial_seed_base'],p['initial_seed_base']+p['ensembles'])]
        if len(set(seeds))!=len(seeds) or used.intersection(seeds) or set(seeds).intersection(range(4101,4107)):raise RuntimeError('Mock data/initial seeds overlap another mock or original')
        used.update(seeds);plans.append((mid,path))
    if study['workers_per_mock']!=4 or study['total_workers']!=4*len(plans) or len(plans)!=2:raise RuntimeError('This bounded study requires two mocks and eight total workers')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'RUN.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(study_sha256=sha(planpath),binary_sha256=sha(exe),mock_plans={str(mid):sha(p) for mid,p in plans})
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text())!=identity:raise RuntimeError('Study/plan/binary changed')
    else:
        if any(p.name!='RUN.lock' for p in root.iterdir()):raise RuntimeError('New study needs an empty output directory')
        (root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');(root/'plan.json').write_bytes(planpath.read_bytes())
    for mid,p in plans:(root/f'mock{mid}').mkdir(exist_ok=True)
    stop=threading.Event()
    def pause(sig,frame):
        stop.set()
        for mid,p in plans:(root/f'mock{mid}/PAUSE').touch()
        print('Study pause requested; waiting for both mock runners to save active chunks',flush=True)
    signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
    if args.resume:
        for mid,p in plans:(root/f'mock{mid}/PAUSE').unlink(missing_ok=True)
    if stop.is_set():
        for mid,p in plans:(root/f'mock{mid}/PAUSE').touch()
    def run(item):
        mid,p=item;cmd=[sys.executable,str(Path(__file__).with_name('run.py')),str(root/f'mock{mid}'),'--exe',str(exe),'--plan',str(p),'--workers',str(study['workers_per_mock'])]
        if args.max_chunks is not None:cmd+=['--max-chunks',str(args.max_chunks)]
        # The parent handles study pause markers; children drain their own native batches.
        with (root/f'mock{mid}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,start_new_session=True,pass_fds=(lock.fileno(),))
        complete=(root/f'mock{mid}/completed.json').exists();print(f'Mock {mid}: '+('fixed budget complete' if complete else 'paused at checkpoints'),flush=True);return complete
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:complete=all(list(pool.map(run,plans)))
    if complete:
        (root/'completed.json').write_text(json.dumps(dict(complete=True,mocks=sorted(ids),independent_mock_data=True,fixed_scalar_truth=True))+'\n');(root/'paused.json').unlink(missing_ok=True)
    else:(root/'paused.json').write_text(json.dumps(dict(complete=False,resume='Rerun with --resume'))+'\n')
if __name__=='__main__':main()
