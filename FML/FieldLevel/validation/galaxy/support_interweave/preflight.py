#!/usr/bin/env python3
"""Exact-support native compatibility, checkpoint-policy identity and restart."""
from pathlib import Path
import argparse,concurrent.futures,json,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
for n in ['exe','old-exe']:p.add_argument('--'+n,type=Path,required=True,dest=n.replace('-','_'))
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise ValueError('Fresh control output required')
plan=json.loads((Path(__file__).parent/'plan.json').read_text())
for name in ['legacy','active','old']:(out/name).mkdir()
for name in ['legacy','active']:(out/name/'rescaling-policy.txt').write_text(f'FML_SUPPORT_RESCALING_1 {name}\n')
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
policy=out/'active/rescaling-policy.txt';before={n:(split/n).read_bytes() for n in ['CURRENT','target.json','summary.json']}
try:
 policy.write_text('FML_SUPPORT_RESCALING_1 legacy\n');call(a.exe,split,'resume',0,'changed-policy-reject',False)
 policy.write_text('FML_SUPPORT_RESCALING_1 active extra\n');call(a.exe,split,'resume',0,'malformed-policy-reject',False)
 policy.unlink();call(a.exe,split,'resume',0,'missing-policy-reject',False)
 for n,v in before.items():assert (split/n).read_bytes()==v
finally:policy.write_text('FML_SUPPORT_RESCALING_1 active\n')
r=dict(passed=True,legacy_32_draws_summary_target_exact=True,active_32_draws_checkpoint_summary_target_restart_exact=True,changed_policy_rejects_without_writes=True,missing_or_malformed_policy_rejects_without_writes=True,actual_warmup_sampling_gradient_counts_exact=True,candidate_target_original_exact=True,old_checkpoint_identity_intentionally_distinct=True)
(out/'controls.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
