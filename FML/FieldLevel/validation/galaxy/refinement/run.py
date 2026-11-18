#!/usr/bin/env python3
"""Evaluate the frozen six-case posterior refinement without changing the source chains."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,signal,subprocess,threading,time
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan.json'));ap.add_argument('--workers',type=int,choices=[1,2,3,4],default=4);ap.add_argument('--max-chunks',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=args.output.resolve();source=args.source.resolve();exe=args.exe.resolve();planpath=args.plan.resolve();plan=json.loads(planpath.read_text())
    if root==source or root in source.parents or source in root.parents:raise RuntimeError('Separate source/output trees required')
    if args.max_chunks is not None and args.max_chunks<1:raise ValueError('Positive chunk budget required')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'RUN.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(plan_sha256=sha(planpath),binary_sha256=sha(exe),source=str(source));stop=threading.Event()
    def pause(sig,frame):stop.set();print('Pause requested; waiting for active prediction batches',flush=True)
    signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text())!=identity:raise RuntimeError('Plan/binary/source changed')
    else:
        if any(p.name!='RUN.lock' for p in root.iterdir()):raise RuntimeError('New study requires empty output')
        shutil.copyfile(planpath,root/'plan.json');(root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    if args.resume:(root/'PAUSE').unlink(missing_ok=True)
    if not (root/'inputs.json').exists():
        source_lock=(source/'RUN.lock').open('a+');fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sha(source/'plan.json')!=plan['source']['plan_sha256'] or sha(source/'posterior-summary.json')!=plan['source']['posterior_summary_sha256']:raise RuntimeError('Frozen source identity differs')
        post=json.loads((source/'posterior-summary.json').read_text());summaries=json.loads((source/'completed.json').read_text());sp=json.loads((source/'plan.json').read_text())
        if not post['passed'] or len(summaries)!=4 or any(s['completed_samples']!=2048 for s in summaries):raise RuntimeError('Completed passing source chains required')
        expected=dict(mesh=4,product_mesh=8,density_mesh=16,boxsize=100,noise_power=1000,nufft_epsilon=1e-8,ensembles=4,samples=2048)
        if any(sp.get(k)!=v for k,v in expected.items()):raise RuntimeError('Source physical/numerical fixture differs')
        inputs=root/'inputs';inputs.mkdir(exist_ok=True);target_bytes=(source/'ensemble0/target.json').read_bytes();target=json.loads(target_bytes);nm=len(target['data']);data=inputs/'data.txt'
        data.write_text(f'FML_MODES_1 {nm}\n'+''.join(f"{j} {target['k'][j]:.17g} {target['weights'][j]:.17g} {v[0]:.17g} {v[1]:.17g}\n" for j,v in enumerate(target['data'])));(inputs/'target.json').write_bytes(target_bytes)
        coords=np.indices((4,4,4)).reshape(3,-1).T;basis=[]
        for wave in [(1,0,0),(1,1,0),(0,0,1)]:
            phase=2*np.pi*(coords@wave)/4;basis.extend([np.sqrt(2/64)*np.cos(phase),np.sqrt(2/64)*np.sin(phase)])
        checks=[];files={}
        for e in range(4):
            directory=source/f'ensemble{e}';assert (directory/'target.json').read_bytes()==target_bytes
            monitors=[];original=[];digest=hashlib.sha256()
            for start in range(0,2048,plan['chunk']):
                lines=[]
                for j in range(start,start+plan['chunk']):
                    raw=(directory/f'draws/draw-{j}.rank-0.json').read_bytes();digest.update(raw);r=json.loads(raw);assert r['draw']==j and r['rank']==0
                    w=np.asarray(r['white']);p=np.asarray(r['parameters']);pred=np.asarray(r['prediction']);assert w.shape==(64,) and p.shape==(5,) and pred.shape==(nm,2) and np.isfinite(w).all() and np.isfinite(p).all() and np.isfinite(pred).all()
                    lines.append(str(j)+' '+' '.join(format(v,'.17g') for v in [*w,*p])+'\n');monitors.append([*p,*(np.array(basis)@w)]);original.append([j,r['quadratic'][0],*pred.ravel()])
                path=inputs/f'ensemble{e}-start{start}.txt';path.write_text(f"FML_STATES_1 {plan['chunk']} {start}\n"+''.join(lines));files[path.name]=sha(path)
            np.save(inputs/f'monitors{e}.npy',np.array(monitors));np.save(inputs/f'original{e}.npy',np.array(original));files[f'monitors{e}.npy']=sha(inputs/f'monitors{e}.npy');files[f'original{e}.npy']=sha(inputs/f'original{e}.npy');checks.append(dict(ensemble=e,checkpoint_sha256=sha(directory/'CURRENT'),cold_records_sha256=digest.hexdigest(),retained=2048))
        files[data.name]=sha(data);files['target.json']=sha(inputs/'target.json')
        (root/'inputs.json').write_text(json.dumps(dict(files=files,source_checks=checks,target_sha256=sha(inputs/'target.json')),indent=2)+'\n');source_lock.close()
    manifest=json.loads((root/'inputs.json').read_text())
    for name,h in manifest['files'].items():
        if sha(root/'inputs'/name)!=h:raise RuntimeError('Sealed input changed: '+name)
    tasks=[(c,e) for c in plan['cases'] for e in range(4)]
    def evaluate(task):
        case,e=task;name=case['name'];directory=root/name/f'ensemble{e}';directory.mkdir(parents=True,exist_ok=True);new=0
        for start in range(0,2048,plan['chunk']):
            output=directory/f'start{start}.csv';meta=directory/f'start{start}.json'
            if output.exists() and meta.exists():
                if sha(output)!=json.loads(meta.read_text())['sha256']:raise RuntimeError('Completed batch changed')
                continue
            if stop.is_set() or (root/'PAUSE').exists() or (args.max_chunks is not None and new>=args.max_chunks):break
            if output.exists():output.unlink() # Atomic native result without committed receipt; repeat this batch.
            cmd=[str(exe),'predict',str(root/f'inputs/ensemble{e}-start{start}.txt'),str(root/'inputs/data.txt'),str(output),str(case['product']),str(case['density']),str(case['epsilon'])];begin=time.monotonic()
            with (directory/f'start{start}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,start_new_session=True,pass_fds=(lock.fileno(),),env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive'))
            rows=np.loadtxt(output,delimiter=',',skiprows=1);assert rows.shape==(plan['chunk'],2+2*len(json.loads((root/'inputs/target.json').read_text())['data'])) and np.array_equal(rows[:,0],np.arange(start,start+plan['chunk'])) and np.isfinite(rows).all()
            meta.write_text(json.dumps(dict(sha256=sha(output),argv=cmd,seconds=time.monotonic()-begin),indent=2)+'\n');new+=1
            print(f'{name} ensemble {e+1}: {start+plan["chunk"]}/2048 predictions',flush=True)
        return all((directory/f'start{start}.json').exists() for start in range(0,2048,plan['chunk']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:complete=all(list(pool.map(evaluate,tasks)))
    if complete:
        (root/'completed.json').write_text(json.dumps(dict(complete=True,cases=len(plan['cases']),ensembles=4,states_per_case=8192),indent=2)+'\n');(root/'paused.json').unlink(missing_ok=True)
    else:(root/'paused.json').write_text(json.dumps(dict(complete=False,resume='Rerun same command with --resume'))+'\n')
if __name__=='__main__':main()
