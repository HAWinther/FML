#!/usr/bin/env python3
"""Copy a completed prefix into a separately declared longer campaign; never edit its source."""
from pathlib import Path
import argparse,fcntl,hashlib,json,shutil,subprocess

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan_long.json'));args=ap.parse_args()
    root=args.output.resolve();source=args.source.resolve();exe=args.exe.resolve();planpath=args.plan.resolve();plan=json.loads(planpath.read_text())
    if root==source or root in source.parents or source in root.parents:raise RuntimeError('Source and destination must be separate campaign trees')
    source_lock=(source/'RUN.lock').open('a+');fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((source/'plan.json').read_text());old_identity=json.loads((source/'identity.json').read_text());declared=plan['prefix']
    if declared['plan_sha256']!=sha(source/'plan.json') or old_identity['plan_sha256']!=declared['plan_sha256']:raise RuntimeError('Source plan identity differs')
    if old['samples']!=declared['retained_per_chain'] or plan['samples']<=old['samples']:raise RuntimeError('Expected a completed smaller retained budget')
    allowed={'version','scope','samples','replay','replay_start','comparison','prefix','stopping','limits'}
    if any(old.get(k)!=plan.get(k) for k in old.keys()|plan.keys() if k not in allowed):raise RuntimeError('Only the declared retained budget/replay/provenance may change')
    if not json.loads((source/'restart-replay.json').read_text())['passed']:raise RuntimeError('Source lacks exact replay')
    summaries=json.loads((source/'completed.json').read_text())
    if len(summaries)!=old['ensembles'] or any(s['completed_samples']!=old['samples'] or s['completed_iterations']!=old['samples']+old['warmup'] for s in summaries):raise RuntimeError('Incomplete source campaign')
    if root.exists() and any(root.iterdir()):raise RuntimeError('Import needs a new empty campaign')
    root.mkdir(parents=True,exist_ok=True);writer_lock=(root/'RUN.lock').open('a+');fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A second importer can have passed the empty check before the lock was created.
    if any(p.name!='RUN.lock' for p in root.iterdir()):raise RuntimeError('Import destination already populated')
    identity=dict(plan_sha256=sha(planpath),binary_sha256=sha(exe));(root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');shutil.copyfile(planpath,root/'plan.json')
    (root/'settings.txt').write_text(f"{plan['warmup']} {plan['samples']} {plan['ensembles']} {plan['seed_layout']}\n");(root/'ladder.txt').write_text(''.join(format(v,'.17g')+'\n' for v in plan['betas']))
    checks=[]
    for e in range(plan['ensembles']):
        original=source/f'ensemble{e}';directory=root/f'ensemble{e}';before=sha(original/'CURRENT')
        cmd=[str(exe),str(directory),'extend',str(e),'0',str(root/'ladder.txt'),str(plan['exchange_passes']),str(root/'settings.txt'),str(original),str(old['samples'])]
        with (root/f'import-ensemble{e}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        s=json.loads((directory/'summary.json').read_text());expected=dict(summaries[e]);expected['samples']=plan['samples']
        if s!=expected:raise RuntimeError('Import changed physical/counter/transport summary')
        shutil.copyfile(original/'target.json',directory/'target.json');aggregate=hashlib.sha256()
        for j in range(old['samples']):
            name=f'draw-{j}.rank-0.json';data=(original/'draws'/name).read_bytes();row=json.loads(data)
            if row['draw']!=j or row['rank']!=0:raise RuntimeError('Source draw index/rank differs')
            aggregate.update(data);(directory/'draws'/name).write_bytes(data)
        if sha(original/'CURRENT')!=before:raise RuntimeError('Source checkpoint changed during import')
        checks.append(dict(ensemble=e,source_checkpoint_sha256=before,target_sha256=sha(original/'target.json'),cold_prefix_sha256=aggregate.hexdigest(),retained=old['samples'],source_unchanged=True))
    previous=json.loads((source/'summary.json').read_text()) if (source/'summary.json').exists() else {}
    manifest=dict(complete=True,plan_sha256=identity['plan_sha256'],binary_sha256=identity['binary_sha256'],source=str(source),source_identity=old_identity,retained_per_chain=old['samples'],checks=checks,source_summed_main_process_seconds=previous.get('inclusive_main_process_seconds',previous.get('summed_main_process_seconds')),independent_replication=False)
    (root/'prefix-import.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Imported completed prefix without changing its source; continuation is not independent replication',flush=True)
if __name__=='__main__':main()
