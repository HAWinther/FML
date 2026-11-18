#!/usr/bin/env python3
"""Summarize deterministic paths without interpreting profiles as probability mass."""
from pathlib import Path
import argparse,csv,json,os,hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('audit',type=Path);args=ap.parse_args();root=args.audit.resolve()
    identity=json.loads((root/'identity.json').read_text());assert hashlib.sha256((root/'points.csv').read_bytes()).hexdigest()==identity['points_sha256']
    raw=list(csv.DictReader((root/'results.csv').open()));assert len(raw)==identity['point_count']==1010
    paths={}
    for r in raw:
        name=r['path'];values={k:float(v) for k,v in r.items() if k!='path'};assert all(np.isfinite(v) for v in values.values());paths.setdefault(name,[]).append(values)
    assert len(paths)==10
    rows=[]
    for name,values in paths.items():
        assert len(values)==101 and [int(v['index']) for v in values]==list(range(101))
        direct=np.array([v['u_direct'] for v in values]);profile=np.array([v['u_profile'] for v in values]);prior=np.array([v['white_prior'] for v in values]);t=np.array([v['t'] for v in values]);assert np.allclose(t,np.linspace(0,1,101),rtol=0,atol=1e-15)
        assert np.all(profile<=direct+1e-8)
        rows.append(dict(path=name,direct_endpoints=[float(direct[0]),float(direct[-1])],profile_endpoints=[float(profile[0]),float(profile[-1])],direct_max_above_endpoint_max=float(direct.max()-max(direct[0],direct[-1])),profile_max_above_endpoint_max=float(profile.max()-max(profile[0],profile[-1])),profile_max_above_endpoint_chord=float(np.max(profile-((1-t)*profile[0]+t*profile[-1]))),max_bias_profile_gain=float(np.max(direct-profile)),max_kkt=max(v['kkt'] for v in values),max_full_error=max(v['profile_full_error'] for v in values)))
    control=paths['rescale-control'];q=np.array([v['u_direct']-v['white_prior'] for v in control]);qp=np.array([v['u_profile']-v['white_prior'] for v in control]);assert np.ptp(q)<1e-9 and np.ptp(qp)<1e-9
    assert max(v['kkt'] for path in paths.values() for v in path)<1e-8
    assert max(v['profile_full_error'] for path in paths.values() for v in path)<1e-10
    # All three path constructions must share the exact endpoint states/potentials.
    for c in [0,2,3]:
        for field in ['u_direct','u_profile','white_prior']:
            for side in [0,-1]:assert np.ptp([paths[f'{kind}-to{c}'][side][field] for kind in ['white','physical','arc']])<1e-10
    # Exploratory reuse of exact profiles: with uniform bias priors, multiplying
    # the fixed-field bias objective by any beta>0 leaves its minimizer unchanged.
    temperatures=[]
    for beta in [1.,.5,.25,.125,.0625,0.]:
        barriers=[]
        for name,values in paths.items():
            if name=='rescale-control':continue
            u=np.array([v['white_prior']+beta*v['quadratic_profile'] for v in values])
            barriers.append(dict(path=name,barrier=float(u.max()-max(u[0],u[-1]))))
        temperatures.append(dict(beta=beta,minimum_barrier=min(v['barrier'] for v in barriers),maximum_barrier=max(v['barrier'] for v in barriers),paths=barriers))
    (root/'tempered-barriers.json').write_text(json.dumps(temperatures,indent=2)+'\n')
    summary=dict(scope='Exploratory deterministic audit; no new Monte Carlo draws',point_count=1010,path_count=10,profile_reference_cases=3,paths=rows,tempered_barriers=temperatures,control_likelihood_span=float(np.ptp(q)),control_profiled_likelihood_span=float(np.ptp(qp)),max_kkt=max(r['max_kkt'] for r in rows),max_full_error=max(r['max_full_error'] for r in rows),limits=json.loads((root/'plan.json').read_text())['limits'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    labels={'white':'Straight whitened field','physical':'Straight physical initial field','arc':'White-direction arc'}
    for col,c in enumerate([0,2,3]):
        for row,kind in enumerate(['white','physical','arc']):
            ax=axes[row,col];v=paths[f'{kind}-to{c}'];t=[p['t'] for p in v];ax.plot(t,[p['u_direct'] for p in v],label='Interpolated biases',color='#999999');ax.plot(t,[p['u_profile'] for p in v],label='Optimized bounded biases',color='#167eb7');ax.plot([0,1],[v[0]['u_profile'],v[-1]['u_profile']],'k:',lw=.8,label='Endpoint chord');ax.set(xlabel='Path coordinate t',ylabel='Relative negative log posterior',title=f'{labels[kind]}: chain 2 → {c+1}');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Exploratory geometry audit: point potentials with exact bounded-bias profiles\nProfiles are not marginalized densities or configuration weights',fontsize=13);fig.savefig(root/'posterior-paths.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,kind in zip(axes,['white','physical','arc']):
        for c,color in zip([0,2,3],['#167eb7','#d68131','#3a9a81']):
            v=paths[f'{kind}-to{c}'];t=np.array([p['t'] for p in v]);u=np.array([p['u_profile'] for p in v]);chord=(1-t)*u[0]+t*u[-1];ax.plot(t,u-chord,color=color,label=f'Chain 2 → {c+1}')
        ax.axhline(0,color='black',ls=':',lw=.8);ax.set(xlabel='Path coordinate t',ylabel='Profiled U − endpoint chord',title=labels[kind]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Residual structure after optimizing all four bounded biases\nA barrier on these paths does not exclude another route',fontsize=13);fig.savefig(root/'profiled-barriers.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    betas=np.array([v['beta'] for v in temperatures]);low=np.array([v['minimum_barrier'] for v in temperatures]);high=np.array([v['maximum_barrier'] for v in temperatures])
    ax.fill_between(betas,low,high,alpha=.2,label='Range across nine profiled paths');ax.plot(betas,low,'o-',label='Minimum path barrier');ax.plot(betas,high,'s-',label='Maximum path barrier');ax.set(xlabel='Likelihood inverse temperature beta',ylabel='Max profiled U_beta above higher endpoint',title='Exploratory likelihood tempering of existing paths');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.savefig(root/'tempered-barriers.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
