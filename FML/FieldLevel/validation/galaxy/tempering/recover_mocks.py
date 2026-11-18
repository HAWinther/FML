#!/usr/bin/env python3
"""Report all predeclared fixed-truth mocks without claiming coverage calibration."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
NAMES=['sigma8_ratio','b_delta','b_delta2','b_K2','b_laplacian','cos100','sin100','cos110','sin110','cos001','sin001']
def field_truth(white):
    w=np.asarray(white,dtype=float);assert w.shape==(64,) and np.isfinite(w).all()
    f=np.fft.fftn(w.reshape(4,4,4),norm='ortho');out=[]
    for n in [(1,0,0),(1,1,0),(0,0,1)]:out.extend([np.sqrt(2)*f[n].real,-np.sqrt(2)*f[n].imag])
    return out

def summarize(x,truth,levels):
    assert x.shape[0]==4 and x.shape[2]==11 and np.isfinite(x).all()
    flat=x.reshape(-1,11);mean=flat.mean(axis=0);sd=flat.std(axis=0,ddof=1);median=np.median(flat,axis=0);rows=[]
    for j,name in enumerate(NAMES):
        intervals={}
        for level in levels:
            lo,hi=np.quantile(flat[:,j],[(1-level)/2,1-(1-level)/2]);intervals[str(level)]=dict(lower=float(lo),upper=float(hi),contains_truth=bool(lo<=truth[j]<=hi))
        rows.append(dict(parameter=name,truth=float(truth[j]),mean=float(mean[j]),median=float(median[j]),sd=float(sd[j]),mean_offset_sd=float((mean[j]-truth[j])/sd[j]) if sd[j]>0 else None,truth_cdf_fraction=float(np.mean(flat[:,j]<truth[j])),intervals=intervals))
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('study',type=Path);args=ap.parse_args();root=args.study.resolve();plan=json.loads((root/'plan.json').read_text());identity=json.loads((root/'identity.json').read_text());assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==identity['study_sha256'];assert json.loads((root/'completed.json').read_text())['complete']
    mocks=[];values=[];targets=[]
    for item in plan['mocks']:
        mid=item['id'];directory=root/f'mock{mid}';p=json.loads((directory/'plan.json').read_text());assert hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()==identity['mock_plans'][str(mid)]
        post=json.loads((directory/'posterior-summary.json').read_text());transport=json.loads((directory/'summary.json').read_text());target_bytes=(directory/'ensemble0/target.json').read_bytes();target=json.loads(target_bytes);targets.append(target_bytes)
        assert target['mock']==dict(id=mid,truth_seed=p['truth_seed'],noise_seed=p['noise_seed'],initial_seed_base=p['initial_seed_base']) and target['truth_parameters']==plan['truth_parameters']
        chains=[]
        for e in range(4):
            assert (directory/f'ensemble{e}/target.json').read_bytes()==target_bytes
            data=np.loadtxt(directory/f'joint.chain{e}.csv',delimiter=',',skiprows=1);assert data.shape==(p['samples'],12) and np.array_equal(data[:,0],np.arange(p['samples']));chains.append(data[:,1:])
        x=np.array(chains);values.append(x);truth=np.array([*target['truth_parameters'],*field_truth(target['truth_white'])]);recovery=summarize(x,truth,plan['recovery']['interval_probabilities'])
        mocks.append(dict(mock_id=mid,computational_passed=post['passed'],recovery_provisional=not post['passed'],source_plan=p,diagnostics=post['diagnostics'],transport_passed=transport['connectivity_passed'],roundtrips=[sum(v) for v in transport['roundtrips_by_ensemble']],measured_divergences=sum(r['divergences'] for s in transport['summaries'] for r in s['replicas'][:-1]),recovery=recovery,fitted_data_residuals=post['residual_shells'],limits=post['limits'],target_sha256=hashlib.sha256(target_bytes).hexdigest()))
    assert len(set(targets))==len(mocks)==2
    counts={str(level):dict(included=sum(m['recovery'][0]['intervals'][str(level)]['contains_truth'] for m in mocks),mocks=len(mocks),all_computational_gates_pass=all(m['computational_passed'] for m in mocks)) for level in plan['recovery']['interval_probabilities']}
    report=dict(scope=plan['scope'],all_mocks_completed=True,computational_passed=all(m['computational_passed'] for m in mocks),recovery_calibration_claim=False,no_pooling_across_mocks=True,mocks=mocks,amplitude_interval_inclusion_counts=counts,note='Fixed scalar truth, two independent mock realizations. Counts/ranks are descriptive; interval misses are not automatic failures and no coverage/SBC claim is made.',limits=plan['limits'])
    (root/'recovery-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors;fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for k,(mock,x) in enumerate(zip(mocks,values)):
        ax=axes[0,k];mid=mock['mock_id'];diag=mock['diagnostics'][0];row=mock['recovery'][0]
        for e in range(4):ax.hist(x[e,:,0],bins=np.linspace(.5,1.5,36),density=True,histtype='step',color=colors[e],label=f'Chain {e+1}')
        ax.axvline(row['truth'],color='black',ls='--',label='Generating ratio');ax.axhline(1,color='grey',ls=':',lw=.8,label='Uniform amplitude prior');ax.set(xlabel='sigma8 / reference sigma8',ylabel='Density',title=f"Mock {mid}: Rhat={diag['rhat']:.4f}, bulk ESS={diag['bulk_ess']:.0f}"+(' (provisional)' if mock['recovery_provisional'] else ''));ax.legend(fontsize=7,ncol=2)
    ax=axes[1,0]
    for k,mock in enumerate(mocks):
        for j,row in enumerate(mock['recovery'][:5]):
            lo=row['intervals']['0.95']['lower'];hi=row['intervals']['0.95']['upper'];sd=row['sd'];y=j+(k-.5)*.18
            if sd>0:ax.errorbar((row['median']-row['truth'])/sd,y,xerr=np.array([[(row['median']-lo)/sd],[(hi-row['median'])/sd]]),fmt='o',color=colors[k],capsize=3,label=f"Mock {mock['mock_id']}" if j==0 else None)
    ax.axvline(0,color='black',ls='--');ax.set_yticks(np.arange(5),NAMES[:5]);ax.set(xlabel='Offset from truth / own posterior SD',title='Medians and central 95% intervals');ax.legend(fontsize=8)
    ax=axes[1,1]
    for k,mock in enumerate(mocks):ax.plot(np.arange(11),[d['rhat'] for d in mock['diagnostics']],'o-',color=colors[k],ms=4,label=f"Mock {mock['mock_id']}")
    ax.axhline(1.01,color='black',ls='--',label='Frozen Rhat gate');ax.set_xticks(np.arange(11),NAMES,rotation=60,ha='right');ax.set(ylabel='Rank-normalized split/folded Rhat',title='Separate mock convergence diagnostics');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Independent fixed-truth mock recovery stress test\n'+('Two new datasets; computational checks pass; no coverage-calibration claim' if report['computational_passed'] else 'Two new datasets; failed-mixing summaries are provisional; no coverage-calibration claim'),fontsize=14);fig.savefig(root/'mock-recovery.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(computational_passed=report['computational_passed'],mocks=[dict(id=m['mock_id'],computational_passed=m['computational_passed'],amplitude=m['recovery'][0],diagnostic=m['diagnostics'][0]) for m in mocks],amplitude_interval_inclusion_counts=counts),indent=2))
if __name__=='__main__':main()
