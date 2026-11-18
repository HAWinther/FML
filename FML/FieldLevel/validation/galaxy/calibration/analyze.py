#!/usr/bin/env python3
"""Report every calibration outcome, including failed mixing and incomplete mocks."""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
from common import sha,write
from study import verify,TEMPERING
sys.path.insert(0,str(TEMPERING))
from sbc_ranks import endpoint_ranks
from check_prior import check_target
NAMES=['sigma8_ratio','b_delta','b_delta2','b_K2','b_laplacian','cos100','sin100','cos110','sin110','cos001','sin001','likelihood_Q']

def projections(white):
    x=np.asarray(white,dtype=float)
    if x.shape[-1]!=64 or not np.isfinite(x).all():raise ValueError('Complete finite white field required')
    f=np.fft.fftn(x.reshape(*x.shape[:-1],4,4,4),axes=(-3,-2,-1),norm='ortho')
    return np.stack([v for n in [(1,0,0),(1,1,0),(0,0,1)] for v in [np.sqrt(2)*f[...,n[0],n[1],n[2]].real,-np.sqrt(2)*f[...,n[0],n[1],n[2]].imag]],axis=-1)

def finite(value):
    if isinstance(value,float):return value if np.isfinite(value) else None
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    return value

def recover(x,truth,levels):
    flat=x.reshape(-1,len(NAMES));rows=[]
    for j,name in enumerate(NAMES):
        intervals={}
        for level in levels:
            lo,hi=np.quantile(flat[:,j],[(1-level)/2,1-(1-level)/2])
            intervals[str(level)]=dict(lower=float(lo),upper=float(hi),contains_truth=bool(lo<=truth[j]<=hi))
        rows.append(dict(parameter=name,truth=float(truth[j]),mean=float(flat[:,j].mean()),median=float(np.median(flat[:,j])),sd=float(flat[:,j].std(ddof=1)),intervals=intervals))
    return rows

def dataset(root,plan,study,diagnostics):
    mid=plan['mock_id'];directory=root/f'mock{mid}'
    if not (directory/'completed.json').exists():
        progress=[json.loads(f.read_text()) if f.exists() else dict(completed_samples=0,completed_iterations=0) for f in [directory/f'ensemble{e}/summary.json' for e in range(4)]]
        return dict(id=mid,complete=False,passed=False,provisional=True,reason='Declared sampling budget incomplete',retained_draws_by_chain=[s['completed_samples'] for s in progress],iterations_by_chain=[s['completed_iterations'] for s in progress])
    if sha(directory/'plan.json')!=sha(root/f'plans/mock{mid}.json'):raise ValueError('Native source plan changed')
    native_identity=json.loads((directory/'identity.json').read_text())
    frozen=json.loads((root/'FROZEN.json').read_text())
    if native_identity['plan_sha256']!=sha(directory/'plan.json') or native_identity['binary_sha256']!=frozen['binary_sha256']:raise ValueError('Native plan/binary identity mismatch')
    # The existing transport reconstruction also checks caches/labels/trips/counters.
    with (directory/'calibration-transport.log').open('w') as log:
        subprocess.run([sys.executable,str(TEMPERING/'analyze.py'),str(directory)],stdout=log,stderr=subprocess.STDOUT,check=True)
    transport=json.loads((directory/'summary.json').read_text())
    target_path=directory/'ensemble0/target.json';check_target(target_path)
    target=json.loads(target_path.read_text());target_bytes=target_path.read_bytes()
    if target['mock']!=dict(id=mid,truth_seed=plan['truth_seed'],noise_seed=plan['noise_seed'],initial_seed_base=plan['initial_seed_base']) or target['scalar_truth_seed']!=plan['scalar_truth_seed']:raise ValueError('Generation controls differ from frozen study')
    summaries=json.loads((directory/'completed.json').read_text())
    truth=np.r_[target['truth_parameters'],projections(target['truth_white']),target['truth_quadratic']]
    z=lambda v:np.asarray(v)[:,0]+1j*np.asarray(v)[:,1]
    data=z(target['data']);weights=np.asarray(target['weights']);variance=plan['noise_power']/plan['boxsize']**3
    chains=[];digest=hashlib.sha256();worst=0
    for e in range(4):
        if (directory/f'ensemble{e}/target.json').read_bytes()!=target_bytes:raise ValueError('Chain targets differ')
        digest.update((directory/f'ensemble{e}/CURRENT').read_bytes());rows=[]
        for draw in range(plan['samples']):
            raw=(directory/f'ensemble{e}/draws/draw-{draw}.rank-0.json').read_bytes();digest.update(raw);record=json.loads(raw)
            if record['draw']!=draw or record['rank']!=0:raise ValueError('Cold draw layout/index mismatch')
            p=np.asarray(record['parameters']);w=np.asarray(record['white']);q=record['quadratic'][0]
            rebuilt=float(.5*np.sum(weights*np.abs(z(record['prediction'])-data)**2)/variance)
            error=abs(rebuilt-q)/max(1,abs(rebuilt));worst=max(worst,error)
            if p.shape!=(5,) or w.shape!=(64,) or not np.isfinite(q) or error>1e-11:raise ValueError('Malformed cold state or inconsistent Q')
            rows.append(np.r_[p,projections(w),q])
        values=np.asarray(rows)
        if values.shape!=(plan['samples'],12) or not np.isfinite(values).all():raise ValueError('Invalid monitor shape/values')
        chains.append(values)
        with (directory/f'calibration.chain{e}.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['draw',*NAMES]);writer.writerows([[j,*v] for j,v in enumerate(values)])
    subprocess.run([str(diagnostics),str(directory/'calibration'),str(plan['samples'])],check=True)
    diagnostics_rows=[];gates=study['posterior_gates']
    for j,row in enumerate(csv.DictReader((directory/'calibration.diagnostics.csv').open())):
        if int(row['projection'])!=j or j>=12:raise ValueError('Wrong diagnostic coordinate ordering')
        d=dict(parameter=NAMES[j],status=row['status'],rhat=float(row['rhat']),bulk_ess=float(row['bulk_ess']),tail_ess=float(row['tail_ess']))
        d['passed']=d['status']=='ok' and np.isfinite([d['rhat'],d['bulk_ess'],d['tail_ess']]).all() and d['rhat']<gates['max_rhat'] and d['bulk_ess']>=gates['min_bulk_ess'] and d['tail_ess']>=gates['min_tail_ess']
        d['passed']=bool(d['passed']);diagnostics_rows.append(finite(d))
    if len(diagnostics_rows)!=12:raise ValueError('All twelve diagnostics required')
    passed=all(d['passed'] for d in diagnostics_rows) and transport['connectivity_passed']
    x=np.stack(chains);endpoint=x[:,study['endpoint_index'],:];ranks=[]
    for j,name in enumerate(NAMES):
        rng=np.random.default_rng([0x55,study['analysis_seed'],mid,j])
        rank,ties=endpoint_ranks(truth[j:j+1],endpoint[:,j][None,:],rng)
        ranks.append(dict(parameter=name,rank=int(rank[0]),equal_endpoints=int(ties[0]),L=4,provisional=not passed))
    result=dict(id=mid,complete=True,passed=bool(passed),provisional=not passed,diagnostics=diagnostics_rows,
                transport_passed=transport['connectivity_passed'],roundtrips=transport['roundtrips_by_ensemble'],
                measured_divergences=sum(r['divergences'] for s in summaries for r in s['replicas'][:-1]),
                recovery=recover(x,truth,study['interval_probabilities']),ranks=ranks,endpoint_index=study['endpoint_index'],
                target_sha256=sha(target_path),cold_records_and_checkpoints_sha256=digest.hexdigest(),max_cold_quadratic_error=worst)
    write(directory/'calibration-summary.json',result);return result

def plot(root,report):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete=[d for d in report['datasets'] if d['complete']]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for i,d in enumerate(complete):
        a=d['recovery'][0];lo=a['intervals']['0.95']['lower'];hi=a['intervals']['0.95']['upper']
        axes[0].errorbar(a['median'],i,xerr=np.array([[a['median']-lo],[hi-a['median']]]),fmt='o',color='tab:blue' if d['passed'] else 'tab:red',capsize=3)
        axes[0].plot(a['truth'],i,'kx')
    axes[0].set_yticks(range(len(complete)),[str(d['id']) for d in complete])
    axes[0].set_ylim(-.5,max(.5,len(complete)-.5))
    axes[0].set(xlabel='sigma8 / reference sigma8',ylabel='Dataset ID',title='95% intervals; crosses = generated truths\nRed = failed computational gates')
    for ax,name in zip(axes[1:],['sigma8_ratio','likelihood_Q']):
        counts=np.zeros(5,dtype=int);failed=np.zeros(5,dtype=int)
        for d in complete:
            rank=next(r for r in d['ranks'] if r['parameter']==name)['rank'];counts[rank]+=1;failed[rank]+=int(not d['passed'])
        ax.bar(np.arange(5),counts,color='tab:blue',label='All completed datasets')
        ax.bar(np.arange(5),failed,color='none',edgecolor='tab:red',hatch='///',label='Provisional subset')
        ax.axhline(len(complete)/5,ls='--',color='black',lw=.8)
        ax.set(xticks=range(5),xlabel='Truth rank among four endpoints',ylabel='Datasets',title=name+' (descriptive ranks)');ax.legend(fontsize=7)
    fig.suptitle(f"{len(complete)}/{len(report['datasets'])} declared datasets complete; no uniformity/coverage decision\nIndependent endpoints; every failed/incomplete outcome retained")
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.savefig(root/'calibration-overview.png',dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study',type=Path);parser.add_argument('--exe',type=Path,required=True);parser.add_argument('--diagnostics',type=Path,required=True)
    args=parser.parse_args();root=args.study.resolve()
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=verify(root,args.exe.resolve(),args.diagnostics.resolve());study=json.loads((root/'study.json').read_text())
        outcomes=[dataset(root,p,study,args.diagnostics.resolve()) for p in selected]
        all_complete=all(d['complete'] for d in outcomes);all_pass=all(d['passed'] for d in outcomes)
        report=dict(scope=study['scope'],role=study['role'],datasets=outcomes,all_completed=all_complete,all_computational_gates_pass=all_complete and all_pass,
                    calibrated_recovery_claim=False,rank_uniformity_test_performed=False,passing_subset_calibration_claim=False,
                    limits=['Small prior-predictive pilot, not precise coverage calibration','Every failed or incomplete mock remains in this report','Ranks are provisional if any computational gate fails; no passing-subset claim','Only final endpoint from each independent chain enters ranks; all cold states enter mixing checks/intervals','Twelve monitored quantities, not all field coordinates','Same-model synthetic target; no real-catalog or new production MPI claim'])
        write(root/'calibration-summary.json',report);plot(root,report)
        print(json.dumps(dict(all_completed=all_complete,all_computational_gates_pass=report['all_computational_gates_pass'],outcomes=[dict(id=d['id'],complete=d['complete'],passed=d['passed']) for d in outcomes]),indent=2))

if __name__=='__main__':main()
