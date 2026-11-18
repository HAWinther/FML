"""Analyze every declared chain/case, retaining failed mixing and refinement checks."""
import argparse, csv, hashlib, json, os
from pathlib import Path
import numpy as np
def density(grid, samples):
    h=max(float(np.std(samples,ddof=1))*len(samples)**(-.2),1e-12)
    return np.exp(-.5*((grid[:,None]-samples[None,:])/h)**2).mean(axis=1)/(np.sqrt(2*np.pi)*h)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def table(path): return np.atleast_1d(np.genfromtxt(path,names=True,delimiter=',',encoding='utf8',dtype=None))
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();out=a.input.resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((out/'manifest.json').read_text());plan=manifest['plan']
    if manifest['status']!='complete' or sha(out/'mock.data.csv')!=manifest['data_sha256']: raise ValueError('Incomplete or changed inputs')
    names=plan['parameter_order'];cases=[];colors=['#2865ad','#dd8333','#34875c']
    for n,nd,width in plan['cases']:
        case=out/f'n{n}-nd{nd}-w{width}';rows=[];metadata=[]
        for j in range(plan['chains']):
            f=case/f'chain{j}.chain.csv';key=str((case/f'chain{j}').relative_to(out))
            if sha(f)!=manifest['chain_sha256'][key]:raise ValueError('Changed chain')
            t=table(f)
            if len(t)!=plan['draws']:raise ValueError('Incomplete chain')
            rows.append(np.column_stack([t[name] for name in names]));metadata.append(table(case/f'chain{j}.meta.csv')[0])
        draws=np.asarray(rows)
        if not np.isfinite(draws).all():raise ValueError('Nonfinite draws')
        diagnostics=table(case/'diagnostics.csv');posterior={};mix=True
        for j,name in enumerate(names):
            x=draws[:,:,j].ravel();d=diagnostics[j]
            ok=str(d['status'])=='ok' and d['rhat']<plan['mixing_gates']['rhat_max'] and d['bulk_ess']>=plan['mixing_gates']['bulk_ess_min'] and d['tail_ess']>=plan['mixing_gates']['tail_ess_min']
            mix &= bool(ok)
            posterior[name]=dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),q025=float(np.quantile(x,.025)),q16=float(np.quantile(x,.16)),median=float(np.median(x)),q84=float(np.quantile(x,.84)),q975=float(np.quantile(x,.975)),rhat=float(d['rhat']),bulk_ess=float(d['bulk_ess']),tail_ess=float(d['tail_ess']),mixing_passed=bool(ok))
        cases.append(dict(label=f'N{n} / deposit {nd} / w{width}',directory=case.name,posterior=posterior,mixing_passed=mix,
                          mean_acceptance=float(np.mean([m['acceptance'] for m in metadata])),chain_wall_seconds_sum=float(sum(m['seconds'] for m in metadata)),draws=draws))
    reference=cases[-1];sd=reference['posterior']['alpha']['sd'];batch=64
    if plan['draws']%batch:raise ValueError('Declared paired batch comparison requires multiples of 64 draws')
    ref_blocks=reference['draws'][:,:,0].reshape(plan['chains'],-1,batch).mean(axis=2)
    for case in cases:
        blocks=case['draws'][:,:,0].reshape(plan['chains'],-1,batch).mean(axis=2)
        diff=(blocks-ref_blocks).ravel();shift=(case['posterior']['alpha']['mean']-reference['posterior']['alpha']['mean'])/sd
        mcse=float(diff.std(ddof=1)/np.sqrt(len(diff))/sd)
        case['refinement']=dict(mean_shift_in_reference_sd=shift,paired_batch_mcse_in_reference_sd=mcse,batch_length=batch,
            observed_shift_within_budget=bool(abs(shift)<plan['refinement_mean_shift_budget_in_reference_sd']),
            two_mcse_bound_within_budget=bool(abs(shift)+2*mcse<plan['refinement_mean_shift_budget_in_reference_sd']))
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    lo=min(c['draws'][:,:,0].min() for c in cases);hi=max(c['draws'][:,:,0].max() for c in cases);grid=np.linspace(lo,hi,500)
    for c,color in zip(cases,colors):
        axes[0].plot(grid,density(grid,c['draws'][:,:,0].ravel()),color=color,label=c['label'])
    axes[0].axvline(1,color='black',ls='--',label='Known amplitude');axes[0].set(xlabel=r'$\alpha=\sigma_8/\sigma_{8,\mathrm{fid}}$',ylabel='Marginal density');axes[0].legend(fontsize=8)
    for j in range(plan['chains']):axes[1].plot(cases[0]['draws'][j,:,0],lw=.65,alpha=.8,label=f'Chain {j+1}')
    axes[1].axhline(1,color='black',ls='--');axes[1].set(xlabel='Retained draw',ylabel=r'$\alpha$',title='Coarsest mesh: all retained amplitude draws');axes[1].legend(fontsize=8,ncol=2)
    fig.suptitle(('Fixed ICs; four biases marginalized — Gaussian tracer field' if all(c['mixing_passed'] for c in cases) else 'Preliminary fixed-IC field pilot — mixing gate NOT passed'),fontsize=12);fig.savefig(out/'posterior.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for j,(c,color) in enumerate(zip(cases,colors)):
        d=c['posterior']['alpha'];med=d['median'];axes[0].errorbar(med,j,xerr=[[med-d['q16']],[d['q84']-med]],fmt='o',color=color,capsize=4)
        r=c['refinement'];axes[1].errorbar(r['mean_shift_in_reference_sd'],j,xerr=2*r['paired_batch_mcse_in_reference_sd'],fmt='o',color=color,capsize=4)
    axes[0].axvline(1,color='black',ls='--');axes[0].set(xlabel=r'$\alpha$: median and central 68% interval')
    axes[1].axvspan(-.1,.1,color='#dddddd',alpha=.5);axes[1].axvline(0,color='black',ls='--');axes[1].set(xlabel='Mean shift / finest-mesh posterior SD',title='Error bars: 2 × paired batch MC estimate')
    for ax in axes:ax.set_yticks(range(len(cases)),[c['label'] for c in cases]);ax.grid(axis='x',alpha=.2)
    fig.suptitle(('Same data, physical ICs, filters, noise and priors across meshes' if all(c['mixing_passed'] for c in cases) else 'Preliminary same-data refinement — mixing gate NOT passed'),fontsize=12);fig.savefig(out/'refinement.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(5,5,figsize=(11,11),layout='constrained');x=cases[0]['draws'].reshape(-1,5)
    for i in range(5):
        for j in range(5):
            ax=axes[i,j]
            if i<j:ax.set_visible(False);continue
            if i==j:ax.hist(x[:,i],bins=40,density=True,color=colors[0],alpha=.75);ax.axvline(plan['truth'][i],color='black',ls='--')
            else:ax.scatter(x[:,j],x[:,i],s=1,alpha=.1,color=colors[0],rasterized=True);ax.axvline(plan['truth'][j],color='black',ls='--',lw=.7);ax.axhline(plan['truth'][i],color='black',ls='--',lw=.7)
            if i==4:ax.set_xlabel(names[j],fontsize=8)
            if j==0 and i>0:ax.set_ylabel(names[i],fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(('Coarsest-mesh joint posterior; dashed lines show synthetic truth' if cases[0]['mixing_passed'] else 'Preliminary joint draws — mixing gate NOT passed; dashed = truth'));fig.savefig(out/'corner.png',dpi=160);plt.close(fig)
    mode=table(out/cases[-1]['directory']/'mean.rank0.modes.csv');mesh=32
    data=np.zeros((mesh,mesh,mesh//2+1),complex);model=data.copy()
    for r in mode:
        key=(int(r['nx'])%mesh,int(r['ny'])%mesh,int(r['nz']))
        data[key]=complex(r['data_re'],r['data_im']);model[key]=complex(r['model_re'],r['model_im'])
    d=np.fft.irfftn(data*mesh**3,s=(mesh,)*3).real;m=np.fft.irfftn(model*mesh**3,s=(mesh,)*3).real
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');v=max(abs(d[:,:,mesh//2]).max(),abs(m[:,:,mesh//2]).max())
    residual=d-m;rv=max(abs(residual[:,:,mesh//2]).max(),1e-12)
    for j,(ax,f,title) in enumerate(zip(axes,[d,m,residual],['Observed field','Model at mean parameters','Residual (separate scale)'])):
        scale=rv if j==2 else v
        im=ax.imshow(f[:,:,mesh//2].T,origin='lower',extent=[0,100,0,100],cmap='RdBu_r',vmin=-scale,vmax=scale)
        ax.set(title=title,xlabel='Mpc/h',ylabel='Mpc/h');fig.colorbar(im,ax=ax,shrink=.8,label='Tracer overdensity')
    fig.suptitle(('Finest mesh: fit through retained Fourier modes' if cases[-1]['mixing_passed'] else 'Finest mesh: preliminary fit through retained Fourier modes'));fig.savefig(out/'fields.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');wave2=mode['nx']**2+mode['ny']**2+mode['nz']**2
    kk=[];dp=[];mp=[];rp=[];volume=plan['boxsize']**3
    for shell in np.unique(wave2):
        mask=wave2==shell;w=mode['weight'][mask]
        obs=mode['data_re'][mask]+1j*mode['data_im'][mask];pred=mode['model_re'][mask]+1j*mode['model_im'][mask]
        kk.append(mode['k'][mask].mean());dp.append(volume*np.average(abs(obs)**2,weights=w));mp.append(volume*np.average(abs(pred)**2,weights=w));rp.append(volume*np.average(abs(obs-pred)**2,weights=w))
    axes[0].loglog(kk,dp,'o-',label='Data');axes[0].loglog(kk,mp,'s--',label='Model at mean parameters');axes[0].set(ylabel=r'$P(k)$ [(Mpc/h)$^3$]');axes[0].legend(fontsize=8)
    axes[1].plot(kk,np.asarray(rp)/plan['noise_power'],'o-');axes[1].axhline(1,color='black',ls='--');axes[1].set(ylabel='Residual power / fixed noise power')
    for ax in axes:ax.set_xlabel('k [h/Mpc]');ax.grid(alpha=.2)
    fig.suptitle(('Finest-mesh fit; residuals at mean parameters' if cases[-1]['mixing_passed'] else 'Preliminary finest-mesh fit; residuals at mean parameters'));fig.savefig(out/'spectra.png',dpi=180);plt.close(fig)
    summary=dict(scope=plan['scope'],data_sha256=manifest['data_sha256'],all_mixing_passed=all(c['mixing_passed'] for c in cases),
                 refinement_observed_passed=all(c['refinement']['observed_shift_within_budget'] for c in cases),
                 refinement_two_mcse_passed=all(c['refinement']['two_mcse_bound_within_budget'] for c in cases),
                 monte_carlo_note='Paired length-64 batch means use common chain seeds; this is an approximate MC estimate, not a numerical error bound.',
                 cases=[{k:v for k,v in c.items() if k!='draws'} for c in cases])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
