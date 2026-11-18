"""Compare separately diagnosed campaigns with identical frozen targets; never pool their chains."""
import argparse,json,os
from pathlib import Path
import numpy as np

def density(grid,x):
    h=max(float(np.std(x,ddof=1))*len(x)**(-.2),1e-12)
    return np.exp(-.5*((grid[:,None]-x[None,:])/h)**2).mean(axis=1)/(np.sqrt(2*np.pi)*h)

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--trial',type=Path,required=True)
    a=p.parse_args();old=a.reference.resolve();new=a.trial.resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(new/'plot-cache'));os.environ.setdefault('XDG_CACHE_HOME',str(new/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    om=json.loads((old/'manifest.json').read_text());nm=json.loads((new/'manifest.json').read_text())
    osum=json.loads((old/'summary.json').read_text());nsum=json.loads((new/'summary.json').read_text())
    target_keys=['ic_seed','noise_seed','boxsize','reference_spectrum','initial_cutoff_in_fundamental_modes','initial_geometry',
                 'matter_cutoff_in_fundamental_modes','matter_geometry','likelihood_cutoff_in_fundamental_modes','likelihood_geometry',
                 'noise_power','truth','parameter_order','prior_lower','prior_upper','generator','cases']
    if om['data_sha256']!=nm['data_sha256'] or any(om['plan'][k]!=nm['plan'][k] for k in target_keys):
        raise ValueError('Campaigns do not have identical declared statistical targets/data')
    if any(om['plan'][k]!=nm['plan'][k] for k in ['chains','warmup','draws','amplitude_step','mixing_gates']):
        raise ValueError('This comparison requires equal declared budgets and mixing gates')
    rows=[]
    for oc,nc in zip(osum['cases'],nsum['cases']):
        if oc['directory']!=nc['directory']:raise ValueError('Different case ordering')
        o=oc['posterior']['alpha'];n=nc['posterior']['alpha']
        evals=[]
        for root in [old,new]:
            total=0
            for j in range(om['plan']['chains']):
                m=np.genfromtxt(root/oc['directory']/f'chain{j}.meta.csv',names=True,delimiter=',')
                total+=int(m['evaluations'])
            evals.append(total)
        rows.append(dict(label=nc['label'],old_mixing_passed=oc['mixing_passed'],new_mixing_passed=nc['mixing_passed'],
                         old_alpha_bulk_ess=o['bulk_ess'],new_alpha_bulk_ess=n['bulk_ess'],
                         ess_per_forward_ratio=(n['bulk_ess']/evals[1])/(o['bulk_ess']/evals[0]),
                         forward_evaluations=evals,alpha_mean_shift_in_new_sd=(n['mean']-o['mean'])/n['sd']))
    def alpha(root,case):
        return np.concatenate([np.genfromtxt(root/case/f'chain{j}.chain.csv',names=True,delimiter=',')['alpha'] for j in range(om['plan']['chains'])])
    ox=alpha(old,osum['cases'][-1]['directory']);nx=alpha(new,nsum['cases'][-1]['directory'])
    grid=np.linspace(min(ox.min(),nx.min()),max(ox.max(),nx.max()),500)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    axes[0].plot(grid,density(grid,ox),label='Original: mixing gate failed',color='#999999')
    axes[0].plot(grid,density(grid,nx),label='Joint Gaussian proposal',color='#2865ad')
    axes[0].axvline(1,color='black',ls='--',label='Injected amplitude');axes[0].set(xlabel=r'$\alpha=\sigma_8/\sigma_{8,\mathrm{fid}}$',ylabel='Marginal density',title='Finest mesh; separate campaigns');axes[0].legend(fontsize=8)
    x=np.arange(len(rows));width=.35
    axes[1].bar(x-width/2,[r['old_alpha_bulk_ess'] for r in rows],width,label='Original',color='#999999')
    axes[1].bar(x+width/2,[r['new_alpha_bulk_ess'] for r in rows],width,label='Joint',color='#2865ad')
    axes[1].axhline(nm['plan']['mixing_gates']['bulk_ess_min'],color='black',ls='--',label='Frozen ESS gate');axes[1].set_xticks(x,[r['label'] for r in rows],rotation=15,ha='right',fontsize=8);axes[1].set(ylabel='Amplitude bulk ESS',title='Same retained-draw budget',ylim=(0,1.35*max(r['new_alpha_bulk_ess'] for r in rows)));axes[1].legend(fontsize=8)
    fig.suptitle('Fixed-IC Gaussian tracer field: proposal comparison');fig.savefig(new/'proposal-comparison.png',dpi=180);plt.close(fig)
    result=dict(same_target=True,same_data=True,old_draws=om['plan']['draws'],new_draws=nm['plan']['draws'],cases=rows,
                note='Campaigns remain separate. Original mixing failures and their draws are preserved; no chains pooled or extended.')
    (new/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
