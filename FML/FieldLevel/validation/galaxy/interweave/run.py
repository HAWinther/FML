#!/usr/bin/env python3
"""Run/resume the frozen four-chain interweaving pilot; never extend or pool replay draws."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--exe', type=Path, default=Path(__file__).resolve().parent/'interweave_pilot')
    parser.add_argument('--workers', type=int, default=2, choices=[1,2,4])
    args = parser.parse_args()
    root=args.output.resolve();exe=args.exe.resolve();root.mkdir(parents=True,exist_ok=True)
    plan_path=Path(__file__).resolve().parent/'plan.json'
    plan=json.loads(plan_path.read_text())
    identity={'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
    manifest=root/'identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:raise RuntimeError('Pilot plan/binary changed; refuse resume')
    else:
        if any(root.iterdir()):raise RuntimeError('New pilot needs an empty output directory')
        manifest.write_text(json.dumps(identity,indent=2)+'\n')
        shutil.copyfile(plan_path,root/'plan.json')
    def invoke(directory,chain,budget,label):
        phase='resume' if (directory/'CURRENT').exists() else 'start'
        cmd=[str(exe),str(directory),phase,str(chain),str(budget)]
        start=time.monotonic()
        with (root/(label+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,
                           env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive'))
        result=json.loads((directory/'summary.json').read_text())
        (root/(label+'.command.json')).write_text(json.dumps({'argv':cmd,'seconds':time.monotonic()-start,'summary':result},indent=2)+'\n')
        return result
    def replay_check(primary,branch):
        for draw in range(128,192):
            name=f'draw-{draw}.rank-0.json'
            if (primary/'draws'/name).read_bytes() != (branch/'draws'/name).read_bytes():
                raise RuntimeError('Restart replay mismatch: '+name)
        # generation numbers may differ; all numerical counters/metrics must match.
        left=json.loads((primary/'summary.json').read_text());right=json.loads((branch/'summary.json').read_text())
        if {k:v for k,v in left.items() if k!='generation'} != {k:v for k,v in right.items() if k!='generation'}:
            raise RuntimeError('Restart replay summary mismatch')
        (root/'restart-replay.json').write_text(json.dumps({'passed':True,'draws_compared':64,'pooled':False},indent=2)+'\n')
    def run_chain(chain):
        directory=root/f'chain{chain}';branch=root/'replay-chain0'
        # CURRENT is authoritative if a process stopped between checkpoint and summary publication.
        if (directory/'CURRENT').exists():
            invoke(directory,chain,0,f'chain{chain}-refresh')
        while True:
            summary=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {}
            done=summary.get('completed_iterations',0)
            if done>=plan['warmup']+plan['samples']:break
            if chain==0 and done==384 and not branch.exists():shutil.copytree(directory,branch)
            if chain==0 and done==448 and not (root/'restart-replay.json').exists():
                invoke(branch,chain,64,'restart-replay');replay_check(directory,branch)
            result=invoke(directory,chain,plan['chunk_trajectories'],f'chain{chain}-after{done}')
            print(f"Chain {chain}: {result['completed_iterations']}/{plan['warmup']+plan['samples']} outer sweeps; {result['completed_samples']} retained",flush=True)
        return json.loads((directory/'summary.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run_chain,range(plan['chains'])))
    if not json.loads((root/'restart-replay.json').read_text())['passed']:raise RuntimeError('Missing restart replay')
    (root/'completed.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Frozen budget complete; diagnostic gates still require analysis',flush=True)

if __name__=='__main__':main()
