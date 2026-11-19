#!/usr/bin/env python3
"""Frozen-donor native compatibility, counter restart and immutable pool identity."""
from pathlib import Path
import argparse,concurrent.futures,json,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
for n in ['exe','old-exe','pool']:p.add_argument('--'+n,type=Path,required=True,dest=n.replace('-','_'))
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise ValueError('Fresh control output required')
plan=json.loads((Path(__file__).parent/'plan.json').read_text())
for name in ['legacy','active','old']:(out/name).mkdir()
for name in ['legacy','active','old']:(out/name/'rescaling-policy.txt').write_text('FML_SUPPORT_RESCALING_1 active\n')
(out/'legacy/donor-pool.txt').write_text('FML_FIXED_DONORS_1 0 64 5 0\n')
(out/'active/donor-pool.txt').write_bytes(a.pool.read_bytes())
settings=out/'settings.txt';settings.write_text(f"16 64 4 3\nFML_MOCK_2 {plan['mock_id']} {plan['truth_seed']} {plan['noise_seed']} {plan['initial_seed_base']} {plan['scalar_truth_seed']}\nFML_FIELD_HMC_1 0.075 16\n")
ladder=out/'ladder.txt';ladder.write_text(''.join(f'{v:.17g}\n' for v in plan['betas']))
def call(exe,directory,phase,budget,label,ok=True):
 with (out/(label+'.log')).open('w') as log:r=subprocess.run([str(exe.resolve()),str(directory),phase,'0',str(budget),str(ladder),'4',str(settings)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
 assert (r.returncode==0)==ok,(label,(out/(label+'.log')).read_text()[-2000:])
def one(case):
 name,exe,split=case;d=out/name;call(exe,d,'start',32 if split else 48,name.replace('/','-')+'-start')
 if split:call(exe,d,'resume',16,name.replace('/','-')+'-resume')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,[('old/ensemble0',a.old_exe,False),('legacy/ensemble0',a.exe,True),('active/direct',a.exe,False),('active/split',a.exe,True)]))
old=out/'old/ensemble0';legacy=out/'legacy/ensemble0';direct=out/'active/direct';split=out/'active/split'
for n in ['target.json','summary.json']:assert (old/n).read_bytes()==(legacy/n).read_bytes(),n
for n in ['CURRENT','target.json','summary.json']:assert (direct/n).read_bytes()==(split/n).read_bytes(),n
for j in range(32):
 n=f'draws/draw-{j}.rank-0.json';assert (old/n).read_bytes()==(legacy/n).read_bytes();assert (direct/n).read_bytes()==(split/n).read_bytes()
assert (old/'target.json').read_bytes()==(direct/'target.json').read_bytes()
for d in [legacy,direct]:
 s=json.loads((d/'summary.json').read_text())
 for r in s['replicas']:
  assert r['field_gradient_evaluations_total']==(48*17 if r['beta'] else 0)
  assert r['field_gradient_evaluations_sampling']==(32*17 if r['beta'] else 0)
pool=out/'active/donor-pool.txt';original=pool.read_bytes();before={n:(split/n).read_bytes() for n in ['CURRENT','target.json','summary.json']}
try:
 lines=original.decode().splitlines();fields=lines[1].split();fields[2]=format(float(fields[2])+.01,'.17g');lines[1]=' '.join(fields);pool.write_text('\n'.join(lines)+'\n');call(a.exe,split,'resume',0,'changed-pool-reject',False)
 pool.write_text('FML_FIXED_DONORS_1 0 64 5 0\n');call(a.exe,split,'resume',0,'disabled-pool-reject',False)
 pool.write_text('FML_FIXED_DONORS_1 1 64 5 12\n');call(a.exe,split,'resume',0,'truncated-pool-reject',False)
 pool.unlink();call(a.exe,split,'resume',0,'missing-pool-reject',False)
 for n,v in before.items():assert (split/n).read_bytes()==v
finally:pool.write_bytes(original)
s=json.loads((direct/'summary.json').read_text())
for v in s['replicas']:
 if v['beta']:
  assert v['donor_attempts']==32 and v['donor_values_sampling']+v['donor_support_rejections']==32
  assert v['donor_values_sampling']<=v['donor_values_total']<=48
  assert v['donor_accepted']<=v['donor_attempts'] and v['donor_large_accepted']<=v['donor_large_attempts']
 else:assert all(v[k]==0 for k in ['donor_attempts','donor_accepted','donor_values_total','donor_values_sampling'])
r=dict(passed=True,disabled_32_draws_summary_target_exact=True,donor_32_draws_full_checkpoint_summary_target_restart_exact=True,changed_or_disabled_pool_rejects_without_writes=True,missing_or_truncated_pool_rejects_without_writes=True,gradient_warmup_and_donor_value_counters_valid=True,target_unchanged=True,new_checkpoint_schema_intentionally_distinct=True)
(out/'controls.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
