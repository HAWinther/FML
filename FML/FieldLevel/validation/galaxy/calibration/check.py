#!/usr/bin/env python3
"""Focused controls for study identity, budget, projections and the Q-only failure."""
from pathlib import Path
import argparse
import copy
import csv
import json
import subprocess
import tempfile
import numpy as np
from common import Budget,write
from study import plans,key_audit,prepare,verify,run
from analyze import projections,recover,NAMES,finite

def rejected(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Invalid controls accepted')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnostics',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(Path(__file__).with_name('plan.json').read_text());selected=plans(plan);audit=key_audit(selected)
    assert audit['sampler_exchange_keys']==2433024 and audit['generation_keys']==56
    for mutate in [lambda p:p['mocks'][1].update(id=p['mocks'][0]['id']),lambda p:p['mocks'][0].update(scalar_truth_seed=p['mocks'][0]['truth_seed']),lambda p:p.update(endpoint_index=0),lambda p:p.update(workers=True),lambda p:p['posterior_gates'].update(min_bulk_ess=1)]:
        broken=copy.deepcopy(plan);mutate(broken);rejected(lambda:plans(broken))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);b=Budget(root/'budget.json',4800,2400)
        assert b.reserve();b.finish(10);assert b.state['spent']==10
        assert b.reserve();b=Budget(root/'budget.json',4800,2400)
        assert b.state['spent']==2410 and b.state['unrefunded_crashed_batches']==1 and not b.reserve()
        rejected(lambda:Budget(root/'budget.json',4801,2400))
        rejected(lambda:b.finish(-1))
        b=Budget(root/'overrun.json',2400,2400);assert b.reserve();b.finish(2401);assert not b.reserve()
    rng=np.random.default_rng(930711);x=rng.normal(size=(3,64));coords=np.indices((4,4,4)).reshape(3,-1).T;basis=[]
    for wave in [(1,0,0),(1,1,0),(0,0,1)]:
        phase=2*np.pi*(coords@wave)/4;basis.extend([np.sqrt(2/64)*np.cos(phase),np.sqrt(2/64)*np.sin(phase)])
    assert np.allclose(projections(x),x@np.array(basis).T,rtol=0,atol=2e-15)
    values=np.broadcast_to(np.arange(8)[:,None],(8,12)).reshape(4,2,12)
    rows=recover(values,np.full(12,100),[.5,.9,.95]);assert all(not r['intervals']['0.95']['contains_truth'] for r in rows)
    assert finite(dict(x=float('nan'),y=[float('inf'),2.]))==dict(x=None,y=[None,2.])
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);study=root/'study';study.mkdir();planpath=root/'plan.json';write(planpath,plan)
        exe=root/'native';exe.write_bytes(b'identity-test-only')
        prepare(study,planpath,exe,args.diagnostics);assert len(verify(study,exe,args.diagnostics))==8 and not list(study.glob('mock*/'))
        assert 'tempering/plan.json' in json.loads((study/'FROZEN.json').read_text())['sources']
        (study/'mock201').mkdir()
        with (study/'RUN.lock').open('a+') as lock:rejected(lambda:run(study,exe,args.diagnostics,False,1,lock))
        assert not (study/'budget.json').exists()
        mock=study/'plans/mock201.json';p=json.loads(mock.read_text());p['truth_seed']+=11;write(mock,p)
        rejected(lambda:verify(study,exe,args.diagnostics))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);prefix=root/'monitors';n=1024
        values=rng.normal(size=(4,n,12));values[:,:,11]+=20
        for case in ['iid','Q_only_offset','constant_Q']:
            x=values.copy()
            if case=='Q_only_offset':x[:,:,11]+=np.array([-6,-2,2,6])[:,None]
            if case=='constant_Q':x[:,:,11]=20
            for e in range(4):
                with (root/f'monitors.chain{e}.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(['draw',*NAMES]);writer.writerows([[j,*v] for j,v in enumerate(x[e])])
            subprocess.run([str(args.diagnostics),str(prefix),str(n)],check=True)
            d=list(csv.DictReader((root/'monitors.diagnostics.csv').open()));assert len(d)==12
            passes=[r['status']=='ok' and float(r['rhat'])<1.01 and min(float(r['bulk_ess']),float(r['tail_ess']))>=400 for r in d]
            assert all(passes[:11])
            assert passes[11]==(case=='iid')
    print('PASS: exact key manifest; malformed plans; conservative crash/time budget; FFT/direct projections; interval misses; immutable freeze; Q-only and constant-Q diagnostic failures')

if __name__=='__main__':main()
