#!/usr/bin/env python3
"""Native metric identity, old-kernel replay and new-metric restart controls."""
from pathlib import Path
import argparse,concurrent.futures,json,shutil,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
for n in ['exe','old-exe','metrics']:p.add_argument('--'+n,type=Path,required=True,dest=n.replace('-','_'))
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise ValueError('Fresh empty control directory required')
plan=json.loads((Path(__file__).parent/'plan.json').read_text());metadata=json.loads((a.metrics/'metrics.json').read_text())
for name in ['legacy','candidate']:
 (out/name).mkdir()
 for m in metadata['metrics']:
  dest=out/name/f"field-metric-ensemble{m['ensemble']}.txt"
  dest.write_bytes(Path(m['file']).read_bytes() if name=='candidate' else b'FML_GN_LEGACY_1\n')
(out/'old').mkdir()
settings=out/'settings.txt';settings.write_text(f"16 64 4 3\nFML_MOCK_2 {plan['mock_id']} {plan['truth_seed']} {plan['noise_seed']} {plan['initial_seed_base']} {plan['scalar_truth_seed']}\nFML_FIELD_HMC_1 0.075 16\n")
ladder=out/'ladder.txt';ladder.write_text(''.join(f'{v:.17g}\n' for v in plan['betas']))
def call(exe,directory,phase,budget,label,ok=True):
 with (out/(label+'.log')).open('w') as log:r=subprocess.run([str(exe.resolve()),str(directory),phase,'0',str(budget),str(ladder),'4',str(settings)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
 assert (r.returncode==0)==ok,(label,(out/(label+'.log')).read_text()[-2000:])
def one(case):
 name,exe,split=case;directory=out/name
 call(exe,directory,'start',32 if split else 48,name.replace('/','-')+'-start')
 if split:call(exe,directory,'resume',16,name.replace('/','-')+'-resume')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,[('old/ensemble0',a.old_exe,False),('legacy/ensemble0',a.exe,True),('candidate/direct',a.exe,False),('candidate/split',a.exe,True)]))
old=out/'old/ensemble0';legacy=out/'legacy/ensemble0';direct=out/'candidate/direct';split=out/'candidate/split'
for name in ['summary.json','target.json']:assert (old/name).read_bytes()==(legacy/name).read_bytes(),name
for name in ['CURRENT','summary.json','target.json']:assert (direct/name).read_bytes()==(split/name).read_bytes(),name
for j in range(32):
 name=f'draws/draw-{j}.rank-0.json'
 assert (old/name).read_bytes()==(legacy/name).read_bytes(),name
 assert (direct/name).read_bytes()==(split/name).read_bytes(),name
assert (old/'target.json').read_bytes()==(direct/'target.json').read_bytes()
for path in [legacy,direct]:
 s=json.loads((path/'summary.json').read_text())
 for r in s['replicas']:
  assert r['field_gradient_evaluations_total']==(48*17 if r['beta'] else 0)
  assert r['field_gradient_evaluations_sampling']==(32*17 if r['beta'] else 0)
metric=out/'candidate/field-metric-ensemble0.txt';original=metric.read_bytes();before={n:(split/n).read_bytes() for n in ['CURRENT','summary.json','target.json']}
try:
 lines=original.decode().splitlines();fields=lines[1].split();fields[0]=format(float(fields[0])*1.01,'.17g');lines[1]=' '.join(fields);metric.write_text('\n'.join(lines)+'\n')
 call(a.exe,split,'resume',0,'changed-eigenvalue-reject',False)
 metric.write_bytes(b'FML_GN_LEGACY_1\n');call(a.exe,split,'resume',0,'changed-metric-kind-reject',False)
 metric.write_bytes(b'FML_FROZEN_GN_1 64 18\n');call(a.exe,split,'resume',0,'truncated-metric-reject',False)
 metric.unlink();call(a.exe,split,'resume',0,'missing-metric-reject',False)
 for n,v in before.items():assert (split/n).read_bytes()==v,n
finally:metric.write_bytes(original)
report=dict(passed=True,legacy_32_draws_summary_target_exact=True,new_metric_32_draws_checkpoint_summary_target_restart_exact=True,changed_metric_contents_or_kind_reject_without_writes=True,truncated_or_missing_metric_reject_without_writes=True,actual_gradient_and_warmup_counters_exact=True,candidate_target_matches_baseline=True,legacy_checkpoint_identity_intentionally_distinct=True)
(out/'controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
