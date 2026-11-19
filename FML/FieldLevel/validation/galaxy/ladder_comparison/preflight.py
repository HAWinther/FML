from pathlib import Path
import argparse,concurrent.futures,json,subprocess
parser=argparse.ArgumentParser(description='Native ladder identity, target, restart and cost controls')
parser.add_argument('output',type=Path);parser.add_argument('--exe',type=Path,required=True)
a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise ValueError('Controls require fresh empty output')
mod=Path(__file__).resolve().parent
exe=a.exe.resolve()
plans={n:json.loads((mod/f'plan_{n}.json').read_text()) for n in ['baseline','refined']}
p=plans['baseline'];settings=out/'preflight-settings.txt'
settings.write_text(f"16 64 4 3\nFML_MOCK_2 {p['mock_id']} {p['truth_seed']} {p['noise_seed']} {p['initial_seed_base']} {p['scalar_truth_seed']}\nFML_FIELD_HMC_1 0.075 16\n")
for name,p in plans.items():(out/f'{name}-ladder.txt').write_text(''.join(f'{v:.17g}\n' for v in p['betas']))
changed=list(plans['refined']['betas']);changed[1]=.7
(out/'changed-ladder.txt').write_text(''.join(f'{v:.17g}\n' for v in changed))
def call(name,ladder,phase,budget,label,ok=True):
 command=[str(exe),str(out/name),phase,'0',str(budget),str(out/f'{ladder}-ladder.txt'),'4',str(settings)]
 with (out/f'{label}.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 assert (r.returncode==0)==ok,(label,(out/f'{label}.log').read_text()[-2000:])
def run_one(case):
 name,ladder,split=case
 call(name,ladder,'start',32 if split else 48,name+'-start')
 if split:call(name,ladder,'resume',16,name+'-resume')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run_one,[('baseline-check','baseline',False),('refined-direct','refined',False),('refined-split','refined',True)]))
a=out/'refined-direct';b=out/'refined-split'
for name in ['CURRENT','summary.json','target.json']:
 assert (a/name).read_bytes()==(b/name).read_bytes(),name
for j in range(32):
 name=f'draws/draw-{j}.rank-0.json';assert (a/name).read_bytes()==(b/name).read_bytes(),name
assert (a/'target.json').read_bytes()==(out/'baseline-check/target.json').read_bytes()
for name in ['baseline-check','refined-direct']:
 s=json.loads((out/name/'summary.json').read_text());assert s['field_integrator']==dict(step_size=.075,n_leapfrog=16)
 for r in s['replicas']:
  assert r['field_gradient_evaluations_total']==(48*17 if r['beta'] else 0)
  assert r['field_gradient_evaluations_sampling']==(32*17 if r['beta'] else 0)
before={n:(b/n).read_bytes() for n in ['CURRENT','summary.json','target.json']}
call('refined-split','changed','resume',0,'changed-beta-reject',False)
call('refined-split','baseline','resume',0,'changed-count-reject',False)
for n,v in before.items():assert (b/n).read_bytes()==v
report=dict(passed=True,refined_native_restart_checkpoint_summary_draws_exact=True,all_32_refined_draws_exact=True,baseline_refined_target_exact=True,changed_ladder_or_count_reject_without_writes=True,both_ladder_gradient_counters_and_warmup_separation_exact=True)
(out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
