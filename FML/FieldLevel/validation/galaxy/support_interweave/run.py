#!/usr/bin/env python3
"""Run the frozen exact-initial-support interweaving cohort."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,time,signal,threading,fcntl

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--exe',type=Path,default=Path(__file__).resolve().parent/'support_pilot');ap.add_argument('--workers',type=int,default=2,choices=[1,2,3,4]);ap.add_argument('--plan',type=Path,default=Path(__file__).resolve().parent/'plan.json');ap.add_argument('--max-chunks',type=int,help='Pause each ensemble after this many additional checkpoint chunks');ap.add_argument('--resume',action='store_true',help='Clear the task-owned PAUSE marker and resume');args=ap.parse_args()
    root=args.output.resolve();exe=args.exe.resolve();root.mkdir(parents=True,exist_ok=True);planpath=args.plan.resolve();plan=json.loads(planpath.read_text());identity=dict(plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    reference=json.loads((Path(__file__).resolve().parent.parent/'tempering/plan.json').read_text())
    expected_rescaling=dict(reference['rescaling'],dimension=18,target='Full positive-beta posterior plus exact global initial-support Jacobian; all 64 white prior coordinates retained')
    if plan['rescaling']!=expected_rescaling:raise RuntimeError('Undeclared rescaling changes')
    policy_source=root.parent/'rescaling-policy.txt'
    policy=policy_source.read_bytes()
    if policy!=b'FML_SUPPORT_RESCALING_1 active\n' or hashlib.sha256(policy).hexdigest()!=plan.get('rescaling_policy_sha256'):raise RuntimeError('Unexpected frozen rescaling policy')
    identity['rescaling_policy_sha256']=plan['rescaling_policy_sha256']
    variable={'version','scope','betas','exchange_passes','swaps','roundtrip','comparison','limits','stopping','ensembles','warmup','samples','initial_parameters','local_seeds','swap_seed','replay','truth_seed','noise_seed','initial_seed_base','truth_parameters','field','rescaling'}
    for key,value in reference.items():
        if key not in variable and plan.get(key)!=value:raise RuntimeError('Compiled fixture/control differs from selected plan: '+key)
    if args.max_chunks is not None and args.max_chunks<1:raise RuntimeError('max-chunks must be positive')
    expected_starts=reference['initial_parameters']+[[.9,2.4,-.8,.6,50],[1.1,.8,1,-1,-40]]
    ensembles=plan['ensembles'];warmup=plan['warmup'];samples=plan['samples'];seed_layout=plan.get('seed_layout',1)
    if any(isinstance(v,bool) or not isinstance(v,int) for v in [ensembles,warmup,samples,seed_layout]) or not 1<=ensembles<=4 or warmup<0 or warmup%2 or samples<1 or warmup+samples>1000000 or seed_layout not in [1,2,3]:raise RuntimeError('Invalid runtime schedule')
    if seed_layout==1 and (warmup,samples,ensembles)!=(64,256,2):raise RuntimeError('New schedules require seed layout 2')
    if warmup%16 or samples<48 or samples%16:raise RuntimeError('Runner replay requires warmup and samples in chunks of 16, at least 48 retained draws')
    if plan['initial_parameters']!=expected_starts[:ensembles]:raise RuntimeError('Unsupported frozen initial parameters')
    replay_start=plan.get('replay_start',32)
    if isinstance(replay_start,bool) or not isinstance(replay_start,int) or replay_start<0 or replay_start%16 or replay_start+16>samples:raise RuntimeError('Invalid declared replay interval')
    settings_text=f'{warmup} {samples} {ensembles} {seed_layout}\n'
    mock_metadata=None
    truth_mode=plan.get('truth_mode','fixed')
    if truth_mode not in ['fixed','prior_predictive_v1']:raise RuntimeError('Unknown truth generation mode')
    prior_predictive=truth_mode=='prior_predictive_v1'
    if prior_predictive and (seed_layout!=3 or plan.get('truth_parameters') is not None):raise RuntimeError('Prior-predictive truth requires layout 3 and no fixed truth')
    if not prior_predictive and (plan.get('truth_parameters')!=reference['truth_parameters'] or 'scalar_truth_seed' in plan):raise RuntimeError('Fixed-truth settings differ from compiled fixture')
    if seed_layout==3:
        fields=[plan.get(k) for k in ['mock_id','truth_seed','noise_seed','initial_seed_base']]
        if any(isinstance(v,bool) or not isinstance(v,int) for v in fields):raise RuntimeError('Integer mock identity/seeds required')
        mid,truth_seed,noise_seed,initial_seed=fields
        if not 1<=mid<=65535 or not 1<=truth_seed<=2**32-1 or not 1<=noise_seed<=2**32-1 or not 1<=initial_seed<=2**32-ensembles or len(set([truth_seed,noise_seed,*range(initial_seed,initial_seed+ensembles)]))!=ensembles+2 or 'prefix' in plan:raise RuntimeError('Invalid mock seeds or forbidden prefix reuse')
        mock_metadata=dict(id=mid,truth_seed=truth_seed,noise_seed=noise_seed,initial_seed_base=initial_seed)
        if prior_predictive:
            scalar_seed=plan.get('scalar_truth_seed')
            if isinstance(scalar_seed,bool) or not isinstance(scalar_seed,int) or not 1<=scalar_seed<=2**32-1 or mid<3 or scalar_seed in [truth_seed,noise_seed,*range(initial_seed,initial_seed+ensembles)]:raise RuntimeError('Invalid scalar truth seed or reserved prior-predictive mock ID')
            settings_text+=f'FML_MOCK_2 {mid} {truth_seed} {noise_seed} {initial_seed} {scalar_seed}\n'
        else:settings_text+=f'FML_MOCK_1 {mid} {truth_seed} {noise_seed} {initial_seed}\n'
    elif 'mock_id' in plan or (plan.get('truth_seed'),plan.get('noise_seed'),plan.get('initial_seed_base'))!=(4101,4102,4103):raise RuntimeError('Changed mock seeds require layout 3')
    if seed_layout!=3:raise RuntimeError('Counted comparison requires explicit layout-3 metadata')
    field=plan.get('field',{})
    reference_field=dict(reference['field'])
    if set(field)!=set(reference_field) or any(field[k]!=v for k,v in reference_field.items() if k not in ['step_size','n_leapfrog']):raise RuntimeError('Only explicit field step/count may change')
    if type(field['n_leapfrog'])!=int or not 1<=field['n_leapfrog']<=128 or type(field['step_size']) not in [int,float] or not 0<field['step_size']<=1:raise RuntimeError('Invalid explicit field integrator')
    settings_text+=f"FML_FIELD_HMC_1 {field['step_size']:.17g} {field['n_leapfrog']}\n"
    pause=threading.Event()
    def pause_request(signum,frame):
        pause.set();print('Pause requested; waiting for active checkpoint chunks to finish',flush=True)
    signal.signal(signal.SIGINT,pause_request);signal.signal(signal.SIGTERM,pause_request)
    import math
    beta=plan['betas'];passes=plan.get('exchange_passes',1)
    if not 2<=len(beta)<=64 or beta[0]!=1 or beta[-1]!=0 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in beta) or any(beta[j]<=beta[j+1] for j in range(len(beta)-1)):raise RuntimeError('Invalid explicit ladder')
    if isinstance(passes,bool) or not isinstance(passes,int) or not 1<=passes<=8:raise RuntimeError('Invalid exchange pass count')
    ladder_text=''.join(format(v,'.17g')+'\n' for v in beta)
    # Keep the descriptor alive throughout every native child and replay operation.
    writer_lock=(root/'RUN.lock').open('a+')
    try:fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another runner holds this campaign writer lock') from None
    writer_lock.seek(0);writer_lock.truncate();writer_lock.write(str(os.getpid())+'\n');writer_lock.flush()
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text())!=identity:raise RuntimeError('Plan/binary changed; refuse resume')
    else:
        if any(p.name not in ['RUN.lock','PAUSE'] for p in root.iterdir()):raise RuntimeError('New experiment needs empty output directory')
        (root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');shutil.copyfile(planpath,root/'plan.json')
        (root/'ladder.txt').write_text(ladder_text)
        if seed_layout!=1:(root/'settings.txt').write_text(settings_text)
    policy_copy=root/'rescaling-policy.txt'
    if not policy_copy.exists():
        if any((root/f'ensemble{e}/CURRENT').exists() for e in range(plan['ensembles'])):raise RuntimeError('Missing policy beside checkpoint')
        policy_copy.write_bytes(policy)
    if policy_copy.read_bytes()!=policy:raise RuntimeError('Native policy changed')
    if not (root/'ladder.txt').exists() or (root/'ladder.txt').read_text()!=ladder_text:raise RuntimeError('Runtime ladder differs from selected plan')
    if seed_layout!=1 and (not (root/'settings.txt').exists() or (root/'settings.txt').read_text()!=settings_text):raise RuntimeError('Runtime settings differ from selected plan')
    if 'prefix' in plan:
        imported=json.loads((root/'prefix-import.json').read_text()) if (root/'prefix-import.json').exists() else {}
        if not imported.get('complete') or imported.get('plan_sha256')!=identity['plan_sha256'] or imported.get('binary_sha256')!=identity['binary_sha256'] or imported.get('retained_per_chain')!=plan['prefix']['retained_per_chain']:raise RuntimeError('Complete the declared prefix import before running')
    if args.resume:(root/'PAUSE').unlink(missing_ok=True)
    def invoke(directory,ensemble,budget,label):
        phase='resume' if (directory/'CURRENT').exists() else 'start';cmd=[str(exe),str(directory),phase,str(ensemble),str(budget),str(root/'ladder.txt'),str(passes)];
        if seed_layout!=1:cmd.append(str(root/'settings.txt'))
        start=time.monotonic()
        with (root/(label+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,start_new_session=True,pass_fds=(writer_lock.fileno(),),env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='passive'))
        s=json.loads((directory/'summary.json').read_text())
        if mock_metadata is not None and s.get('mock')!=mock_metadata:raise RuntimeError('Native mock metadata differs from selected plan')
        if prior_predictive and (s.get('truth_mode')!=truth_mode or s.get('scalar_truth_seed')!=plan['scalar_truth_seed']):raise RuntimeError('Native truth generation differs from selected plan')
        if s.get('field_integrator')!=dict(step_size=field['step_size'],n_leapfrog=field['n_leapfrog']):raise RuntimeError('Native field integrator mismatch')
        if s['betas']!=beta or s.get('exchange_passes',1)!=passes or s['warmup']!=plan['warmup'] or s['samples']!=plan['samples']:raise RuntimeError('Native controls differ from selected plan')
        (root/(label+'.command.json')).write_text(json.dumps(dict(argv=cmd,seconds=time.monotonic()-start,summary=s),indent=2)+'\n');return s
    def replay(primary,branch):
        if not (root/'restart-replay.json').exists():
            invoke(branch,0,0,'restart-replay-refresh')
            done=json.loads((branch/'summary.json').read_text())['completed_iterations']
            if done>warmup+replay_start+16:raise RuntimeError('Replay branch advanced beyond its declared boundary')
            if done<warmup+replay_start+16:invoke(branch,0,warmup+replay_start+16-done,'restart-replay')
            for draw in range(replay_start,replay_start+16):
                name=f'draw-{draw}.rank-0.json'
                if (primary/'draws'/name).read_bytes()!=(branch/'draws'/name).read_bytes():raise RuntimeError('Replica draw replay differs')
            if (primary/'CURRENT').read_bytes()!=(branch/'CURRENT').read_bytes():raise RuntimeError('Full ensemble restart differs')
            if (primary/'summary.json').read_bytes()!=(branch/'summary.json').read_bytes():raise RuntimeError('Replica counters/transport replay differs')
            (root/'restart-replay.json').write_text(json.dumps(dict(passed=True,full_ensemble_bytes=True,cold_draws_compared=16,pooled=False,replay_start=replay_start),indent=2)+'\n')
    def run_ensemble(e):
        directory=root/f'ensemble{e}';branch=root/'replay-ensemble0';chunks=0
        if (directory/'CURRENT').exists():invoke(directory,e,0,f'ensemble{e}-refresh')
        while True:
            s=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {};done=s.get('completed_iterations',0)
            if e==0 and done==warmup+replay_start and not branch.exists():shutil.copytree(directory,branch)
            if e==0 and done==warmup+replay_start+16:replay(directory,branch)
            if done>=plan['warmup']+plan['samples']:break
            if pause.is_set() or (root/'PAUSE').exists() or (args.max_chunks is not None and chunks>=args.max_chunks):break
            s=invoke(directory,e,plan['chunk_sweeps'],f'ensemble{e}-after{done}')
            chunks+=1
            print(f"Ensemble {e}: {s['completed_iterations']}/{warmup+samples} sweeps; {s['completed_samples']} cold records",flush=True)
        return json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else dict(completed_iterations=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run_ensemble,range(plan['ensembles'])))
    if any(s['completed_iterations']<warmup+samples for s in results):
        (root/'paused.json').write_text(json.dumps(dict(complete=False,ensembles=results,resume='Rerun the same command with --resume'),indent=2)+'\n');print('Paused cleanly at complete ensemble checkpoints',flush=True);return
    (root/'paused.json').unlink(missing_ok=True)
    if not json.loads((root/'restart-replay.json').read_text())['passed']:raise RuntimeError('Missing exact ensemble replay')
    (root/'completed.json').write_text(json.dumps(results,indent=2)+'\n');print('Frozen budget complete; analyze before making any convergence claim',flush=True)
if __name__=='__main__':main()
