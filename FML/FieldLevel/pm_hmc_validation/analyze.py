"""Compare bounded two-mode conditional PM HMC with finite-grid quadrature."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))

def quadrature(path):
    rows = read(path); x = np.unique(rows['z0']); y = np.unique(rows['z1'])
    assert len(rows) == len(x) * len(y) and np.all(np.isfinite(rows['potential']))
    xx = rows['z0'].reshape(len(x), len(y)); yy = rows['z1'].reshape(xx.shape)
    assert np.all(xx == x[:, None]) and np.all(yy == y[None, :])
    w = np.exp(-rows['potential'].reshape(xx.shape) + rows['potential'].min())
    w[[0, -1], :] *= .5; w[:, [0, -1]] *= .5; w /= w.sum()
    f = np.stack([xx, yy, xx**2, yy**2, xx*yy])
    mean = np.sum(f*w, axis=(1,2)); variance = np.sum(f*f*w, axis=(1,2)) - mean*mean
    covariance = np.array([[mean[2]-mean[0]**2, mean[4]-mean[0]*mean[1]],
                           [mean[4]-mean[0]*mean[1], mean[3]-mean[1]**2]])
    edge = np.zeros(w.shape, dtype=bool); edge[[0,-1],:]=True; edge[:,[0,-1]]=True
    return dict(x=x, y=y, xx=xx, yy=yy, w=w, mean=mean, variance=variance,
                covariance=covariance, edge_mass=float(w[edge].sum()))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
gates=[]
def check(valid, description):
    gates.append(dict(check=description, passed=bool(valid)))

q = quadrature(a.input/'science/ref81.grid.csv'); coarse = quadrature(a.input/'science/ref41.grid.csv')
wide = quadrature(a.input/'science/ref109.grid.csv')
refinement=np.abs(q['mean']-coarse['mean']); expansion=np.abs(q['mean']-wide['mean'])
check(refinement.max()<.002, 'quadrature moment refinement')
check(expansion.max()<1e-6 and q['edge_mass']<1e-8, 'quadrature range expansion/boundary mass')
features=['z0','z1','z0_squared','z1_squared','cross']; results={}; chains={}
for family, count in [('fixed',4),('adapt',2)]:
    arrays=[read(a.input/f'science/{family}{i}.chain.csv') for i in range(count)]
    chains[family]=arrays
    z=np.array([np.column_stack([r['z0'],r['z1']]) for r in arrays])
    f=np.stack([z[:,:,0],z[:,:,1],z[:,:,0]**2,z[:,:,1]**2,z[:,:,0]*z[:,:,1]],axis=-1)
    with (a.input/f'science/{family}.diagnostics.csv').open() as file:
        diag={r['feature']:r for r in csv.DictReader(file)}
    metas=[read(a.input/f'science/{family}{i}.meta.csv')[0] for i in range(count)]
    check(all(m['divergences']==0 and .5<m['acceptance']<=1 for m in metas), family+' conditional acceptance/no post-warmup divergences')
    for name in ['z0','z1']:
        check(diag[name]['status']=='ok' and float(diag[name]['rhat'])<1.05 and float(diag[name]['ess_bulk'])>100,
              family+' '+name+' Rhat/ESS')
    estimates=f.mean(axis=(0,1)); errors=[]; se=[]
    for j,name in enumerate(features):
        n=f.shape[0]*f.shape[1]; ess=min(n,float(diag[name]['ess_bulk']))
        # Conservative agreement envelope: raw-feature batch means, independent
        # chain means, and an ESS-based reference variance estimate. This is not
        # an exact confidence interval or a posterior-coverage experiment.
        blocks=f[:,:,j].reshape(count,-1,32).mean(axis=2).ravel()
        batch_se=float(np.std(blocks,ddof=1)/np.sqrt(len(blocks)))
        chain_se=float(np.std(f[:,:,j].mean(axis=1),ddof=1)/np.sqrt(count))
        ess_se=float(np.sqrt(max(0.,q['variance'][j])/max(1.,ess)))
        uncertainty=max(batch_se,chain_se,ess_se)
        difference=abs(estimates[j]-q['mean'][j]); tolerance=5*uncertainty+2*refinement[j]
        check(difference<tolerance, family+' '+name+' quadrature moment agreement')
        errors.append(float(difference));se.append(uncertainty)
    results[family]=dict(draws=count*len(arrays[0]),means=estimates.tolist(),moment_errors=errors,
        conservative_mc_errors=se,mean_acceptance=float(np.mean([m['acceptance'] for m in metas])),
        epsilon=[float(m['epsilon']) for m in metas],divergences=int(sum(m['divergences'] for m in metas)),
        max_rhat=max(float(diag[n]['rhat']) for n in ['z0','z1']),
        min_bulk_ess=min(float(diag[n]['ess_bulk']) for n in ['z0','z1']),
        diagnostics=diag,repeated_draws=int(sum(np.sum(np.all(np.diff(v[:,:2],axis=0)==0,axis=1)) for v in z)))
energy=read(a.input/'science/sweep.energy.csv');sweep=[]
for eps in sorted(set(energy['epsilon']),reverse=True):
    rows=energy[energy['epsilon']==eps]
    sweep.append(dict(epsilon=float(eps),max_abs_delta_h=float(np.max(np.abs(rows['delta_h']))),max_reversal_error=float(np.max(rows['reversal_error']))))
check(sweep[-1]['max_abs_delta_h']<.01*sweep[0]['max_abs_delta_h'], 'energy error decreases at finer steps')
check(max(v['max_reversal_error'] for v in sweep)<1e-8, 'deterministic leapfrog reversibility')
comparisons={}; full_meta={}
for label in ['serial','mpi1','mpi2','mpi4','pencil4','pencil4-thread1','sanitizer']:
    if not (a.input/f'replay/{label}-conditional.chain.csv').exists(): continue
    for case,suffix in [('conditional','.chain.csv'),('full-fixed','.full.csv'),('full-joint','.full.csv')]:
        rows=read(a.input/f'replay/{label}-{case}{suffix}');base=read(a.input/f'replay/serial-{case}{suffix}')
        values=np.column_stack([rows[n] for n in rows.dtype.names]);bv=np.column_stack([base[n] for n in base.dtype.names])
        check(values.shape==bv.shape and np.all(np.isfinite(values)),label+' '+case+' schema/finite draws')
        error=float(np.max(np.abs(values-bv)));check(error<2e-8,label+' '+case+' rank/backend/thread replay')
        comparisons[label+'-'+case]=error
        meta=read(a.input/f'replay/{label}-{case}.meta.csv')[0]
        check(meta['divergences']==0 and .5<meta['acceptance']<=1,label+' '+case+' acceptance/divergences')
        if case.startswith('full'):
            final=np.loadtxt(a.input/f'replay/{label}-{case}.snapshot');ref=np.loadtxt(a.input/f'replay/serial-{case}.snapshot')
            error=float(np.max(np.abs(final-ref)));check(error<2e-8,label+' '+case+' canonical last state');comparisons[label+'-'+case+'-last']=error
            check(len(final)==(66 if case=='full-joint' else 64),label+' '+case+' field/parameter block sizes')
            check(np.all((-1<=rows['log_amplitude'])&(rows['log_amplitude']<=1)&(-14<=rows['log_noise'])&(rows['log_noise']<=-3)), label+' '+case+' parameter support')
        full_meta[label+'-'+case]=dict(acceptance=float(meta['acceptance']),divergences=int(meta['divergences']),epsilon=float(meta['epsilon']),draws=int(meta['draws']))
summary=dict(gates=gates,passed=all(g['passed'] for g in gates),reference_means=q['mean'].tolist(),
    reference_covariance=q['covariance'].tolist(),refinement_error=refinement.tolist(),range_expansion_error=expansion.tolist(),
    boundary_mass=q['edge_mass'],chains=results,energy=sweep,replay_errors=comparisons,replay_metadata=full_meta,
    limits='Two-mode conditional finite-grid reference, fixed spectrum/noise/other white directions. Full-field/joint runs are short integration/replay checks. No general posterior coverage or production calibration.')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'moment_comparison.csv').open('w') as file:
    writer=csv.writer(file);writer.writerow(['family','feature','quadrature','HMC','absolute_error','conservative_MC_error'])
    for family,result in results.items():
        for j,name in enumerate(features):writer.writerow([family,name,q['mean'][j],result['means'][j],result['moment_errors'][j],result['conservative_mc_errors'][j]])
import os
os.environ.setdefault('MPLCONFIGDIR',str(a.output/'mpl'));os.environ.setdefault('XDG_CACHE_HOME',str(a.output/'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
fig,axes=plt.subplots(2,2,figsize=(11,8.3),layout='constrained');ax=axes[0,0]
density=q['w']/((q['x'][1]-q['x'][0])*(q['y'][1]-q['y'][0]))
order=np.argsort(density.ravel())[::-1];cumulative=np.cumsum(q['w'].ravel()[order]);levels=sorted([density.ravel()[order[np.searchsorted(cumulative,p)]] for p in [.95,.68]])
ax.contour(q['xx'],q['yy'],density,levels=levels,colors=['#788795','#263d50'],linewidths=[1.,1.6])
for i,rows in enumerate(chains['fixed']):ax.scatter(rows['z0'][::2],rows['z1'][::2],s=8,alpha=.3,label='HMC draws' if i==0 else None,color='#2b7d9e')
ax.plot(1.4,-.9,'*',color='#bd5f32',ms=12,label='Injected truth');sig=np.sqrt(np.diag(q['covariance']))
ax.set_xlim(q['mean'][0]-4*sig[0],q['mean'][0]+4*sig[0]);ax.set_ylim(q['mean'][1]-4*sig[1],q['mean'][1]+4*sig[1]);ax.set_xlabel('Cosine-mode coefficient');ax.set_ylabel('Sine-mode coefficient');ax.set_title('Conditional posterior: quadrature and HMC',loc='left',fontweight='bold');ax.legend(fontsize=8)
ax=axes[0,1];x=np.linspace(0,1,80);mu=[];lo=[];hi=[]
for xx in x:
    f=.2*(q['xx']*np.cos(2*np.pi*xx)+q['yy']*np.sin(2*np.pi*xx));idx=np.argsort(f.ravel());cdf=np.cumsum(q['w'].ravel()[idx]);bounds=np.interp([.025,.975],cdf,f.ravel()[idx]);mu.append(float((f*q['w']).sum()));lo.append(bounds[0]);hi.append(bounds[1])
m=results['fixed']['means'];ax.fill_between(x,lo,hi,color='#89b5ca',alpha=.4,label='Quadrature 95% interval');ax.plot(x,mu,color='#263d50',label='Quadrature mean');ax.plot(x,.2*(m[0]*np.cos(2*np.pi*x)+m[1]*np.sin(2*np.pi*x)),'--',color='#247c9e',label='HMC mean');ax.plot(x,.2*(1.4*np.cos(2*np.pi*x)-.9*np.sin(2*np.pi*x)),':',color='#bd5f32',label='Injected truth');ax.set_xlabel('Periodic x / box length');ax.set_ylabel(r'Initial $\delta_{lin}(a=1)$');ax.set_title('Conditional initial-field reconstruction',loc='left',fontweight='bold');ax.legend(fontsize=8)
ax=axes[1,0];ax.loglog([v['epsilon'] for v in sweep],[v['max_abs_delta_h'] for v in sweep],'o-',color='#247c9e');ax.invert_xaxis();ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xticks([.4,.2,.1,.05,.025],['0.4','0.2','0.1','0.05','0.025']);ax.set_xlabel('Leapfrog step size');ax.set_ylabel(r'Max $|\Delta H|$ over four probes');ax.set_title('Fixed trajectory length: step-size sensitivity',loc='left',fontweight='bold');ax.text(.03,.06,'CIC is piecewise differentiable; no universal second-order claim.',transform=ax.transAxes,fontsize=8)
ax=axes[1,1];ax.axis('off');ax.set_title('Post-warmup conditional diagnostics',loc='left',fontweight='bold')
text=''
for family in ['fixed','adapt']:
 r=results[family];text+=f"{family.title()} step: {r['draws']} retained draws\nMean acceptance probability: {r['mean_acceptance']:.3f}\nMax Rhat: {r['max_rhat']:.3f}; min bulk ESS: {r['min_bulk_ess']:.0f}\nDivergences: {r['divergences']} (positive energy threshold 10)\n\n"
text+='Full-field/joint runs test integration and MPI replay.\nThey are separate from the conditional reference.';ax.text(.02,.96,text,va='top',fontsize=10)
for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.grid(alpha=.15)
fig.suptitle(r'Bounded PM HMC: two active modes; $4^3$ particles, $8^3$ force cells',fontsize=14)
fig.supxlabel('Conditional experiment fixes spectrum, noise and all other white directions; one mock does not establish coverage.',fontsize=9)
fig.savefig(a.output/'hmc_validation.png',dpi=165);fig.savefig(a.output/'hmc_validation.pdf');plt.close(fig)
failed=[g['check'] for g in gates if not g['passed']]
print(json.dumps(dict(passed=not failed,failed=failed,reference_mean=q['mean'][:2].tolist(),refinement_error=float(refinement.max()),range_error=float(expansion.max()),max_replay_error=max(comparisons.values()),fixed=results['fixed']['max_rhat'],adapted=results['adapt']['max_rhat']),indent=2))
if failed:raise SystemExit(1)
