#!/usr/bin/env python3
"""Audit the predeclared likelihood-only posterior refinement, without rerunning MCMC."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
from weights import normalized,moments,quantiles,shift
NAMES=['sigma8_ratio','b_delta','b_delta2','b_K2','b_laplacian','cos100','sin100','cos110','sin110','cos001','sin001']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);args=ap.parse_args();root=args.campaign.resolve();plan=json.loads((root/'plan.json').read_text());identity=json.loads((root/'identity.json').read_text());assert sha(root/'plan.json')==identity['plan_sha256'];assert json.loads((root/'completed.json').read_text())['states_per_case']==8192
    manifest=json.loads((root/'inputs.json').read_text())
    for name,h in manifest['files'].items():assert sha(root/'inputs'/name)==h
    source=Path(identity['source'])
    for item in manifest['source_checks']:assert sha(source/f'ensemble{item["ensemble"]}/CURRENT')==item['checkpoint_sha256']
    target=json.loads((root/'inputs/target.json').read_text());data=np.array(target['data']);data=data[:,0]+1j*data[:,1];modeweights=np.array(target['weights']);variance=plan['noise_power']/plan['boxsize']**3
    x=np.stack([np.load(root/f'inputs/monitors{e}.npy') for e in range(4)]);original=np.stack([np.load(root/f'inputs/original{e}.npy') for e in range(4)]);assert x.shape==(4,2048,11)
    evaluated={};q={};pred={};seconds={};g=plan['gates'];worst_q=0
    for case in plan['cases']:
        name=case['name'];chains=[];elapsed=0
        for e in range(4):
            parts=[]
            for start in range(0,2048,plan['chunk']):
                p=root/name/f'ensemble{e}/start{start}.csv';meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['sha256'];elapsed+=meta['seconds'];parts.append(np.loadtxt(p,delimiter=',',skiprows=1))
            rows=np.concatenate(parts);assert rows.shape==original[e].shape and np.array_equal(rows[:,0],np.arange(2048)) and np.isfinite(rows).all();chains.append(rows)
        a=np.stack(chains);q[name]=a[:,:,1];pred[name]=a[:,:,2::2]+1j*a[:,:,3::2];seconds[name]=elapsed
        rebuilt=.5*np.sum(modeweights*np.abs(pred[name]-data)**2/variance,axis=2);error=np.max(np.abs(rebuilt-q[name])/np.maximum(1,np.abs(rebuilt)));worst_q=max(worst_q,float(error));assert error<=g['baseline_quadratic_scaled_error']
    origpred=original[:,:,2::2]+1j*original[:,:,3::2];baseline_pred_error=float(np.max(np.abs(pred['baseline']-origpred)/np.maximum(1,np.abs(origpred))));baseline_q_error=float(np.max(np.abs(q['baseline']-original[:,:,1])/np.maximum(1,np.abs(original[:,:,1]))));assert baseline_pred_error<=g['baseline_prediction_scaled_error'] and baseline_q_error<=g['baseline_quadratic_scaled_error']
    zero=np.zeros((4,2048));base_mean,base_sd,_,_=moments(x,zero);assert np.all(base_sd>0)
    logs={name:q['baseline']-v for name,v in q.items()};mom={};overlap={};case_means={};case_sd={}
    for name,logr in logs.items():
        mean,sd,w,logz=moments(x,logr);mom[name]=(mean,sd,w,logz);fractions=[];extrema=[]
        for e in range(4):
            cw,_=normalized(logr[e]);fractions.append(float(1/np.sum(cw*cw)/len(cw)));extrema.append(float(cw.max()*len(cw)))
        pooled=float(1/np.sum(w*w)/w.size);maximum=float(w.max()*w.size)
        ok=min([pooled,*fractions])>=g['min_importance_overlap_fraction_per_chain_and_pooled'] and max([maximum,*extrema])<=g['max_weight_relative_to_uniform']
        overlap[name]=dict(passed=bool(ok),pooled_overlap_fraction=pooled,chain_overlap_fractions=fractions,max_weight_relative_to_uniform=max([maximum,*extrema]),log_normalization_ratio=logz,note='Importance overlap index only; not independent draws or autocorrelation-corrected MCMC ESS')
        case_means[name]=mean.tolist();case_sd[name]=sd.tolist()
    comparisons=[]
    for a,b in plan['comparisons']:
        ma,sa,wa,_=mom[a];mb,sb,wb,_=mom[b];delta,error,_,_=shift(x,logs[a],logs[b],plan['batch_lengths']);z=delta/base_sd;ze=error/base_sd
        qa=np.array([quantiles(x[:,:,j].ravel(),wa.ravel()) for j in range(11)]);qb=np.array([quantiles(x[:,:,j].ravel(),wb.ravel()) for j in range(11)]);qshift=(qb-qa)/base_sd[:,None]
        distances=np.sqrt(np.sum(modeweights*np.abs(pred[b]-pred[a])**2/variance,axis=2));logdiff=q[a]-q[b];
        with np.errstate(divide='ignore',invalid='ignore'):sdchange=np.where(sa>0,np.abs(sb/sa-1),np.inf)
        predictive_a=np.sum(wa[:,:,None]*pred[a],axis=(0,1));predictive_b=np.sum(wb[:,:,None]*pred[b],axis=(0,1));predictive_distance=float(np.sqrt(np.sum(modeweights*np.abs(predictive_b-predictive_a)**2/variance)))
        gates=dict(overlap=overlap[a]['passed'] and overlap[b]['passed'],sample_prediction_distance=float(distances.max())<=g['max_sample_noise_weighted_prediction_distance'],log_ratio=float(np.max(np.abs(logdiff)))<=g['max_absolute_log_weight'],mean_shift=float(np.max(np.abs(z)))<=g['max_mean_shift_in_baseline_sd'],mean_shift_uncertainty=float(np.max(np.abs(z)+2*ze))<=g['max_mean_shift_plus_two_mcse_in_baseline_sd'],quantile_shift=float(np.max(np.abs(qshift)))<=g['max_quantile_shift_in_baseline_sd'],sd_change=float(sdchange.max())<=g['max_relative_sd_change'],predictive_mean=predictive_distance<=g['max_weighted_predictive_mean_distance'])
        rows=[dict(parameter=NAMES[j],mean_shift_sd=float(z[j]),paired_mcse_sd=float(ze[j]),upper_two_mcse_sd=float(abs(z[j])+2*ze[j]),quantile_shifts_sd=qshift[j].tolist(),relative_sd_change=float(sdchange[j]) if np.isfinite(sdchange[j]) else None) for j in range(11)]
        comparisons.append(dict(from_case=a,to_case=b,passed=all(gates.values()),gates=gates,max_model_distance=float(distances.max()),model_distance_quantiles=np.quantile(distances,[.5,.95,.99,1]).tolist(),max_abs_log_ratio=float(np.max(np.abs(logdiff))),log_ratio_quantiles=np.quantile(logdiff,[0,.05,.5,.95,1]).tolist(),max_mean_shift_sd=float(np.max(np.abs(z))),max_mean_shift_upper_sd=float(np.max(np.abs(z)+2*ze)),max_quantile_shift_sd=float(np.max(np.abs(qshift))),max_relative_sd_change=float(sdchange.max()) if np.isfinite(sdchange).all() else None,weighted_predictive_mean_distance=predictive_distance,coordinates=rows))
    passed=all(v['passed'] for v in overlap.values()) and all(c['passed'] for c in comparisons)
    report=dict(scope=plan['scope'],passed=passed,method='Likelihood-only importance sensitivity on all 8192 existing cold states; same latent dimension and priors',baseline_prediction_scaled_error=baseline_pred_error,baseline_quadratic_scaled_error=baseline_q_error,max_independent_quadratic_reconstruction_error=worst_q,overlap=overlap,comparisons=comparisons,baseline_mean=base_mean.tolist(),baseline_sd=base_sd.tolist(),case_means=case_means,case_sd=case_sd,parameters=NAMES,summed_process_wall_seconds_by_case=seconds,source_checkpoints_unchanged=True,gates=plan['gates'],uncertainty=plan['uncertainty'],limits=plan['limits'])
    (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=[c['name'] for c in plan['cases']];colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');ax=axes[0,0];edges=np.linspace(.5,1.5,36)
    for j,name in enumerate(names):ax.hist(x[:,:,0].ravel(),bins=edges,weights=mom[name][2].ravel(),density=True,histtype='step',color=colors[j],lw=1.2,label=name)
    ax.axvline(1,ls='--',color='black',lw=.8);ax.set(xlabel='sigma8 / reference sigma8',ylabel='Density',title='Reweighted amplitude marginals');ax.legend(fontsize=8)
    ax=axes[0,1];ix=np.arange(11)
    for j,c in enumerate(comparisons):ax.errorbar(ix+(j-2.5)*.09,[r['mean_shift_sd'] for r in c['coordinates']],yerr=[2*r['paired_mcse_sd'] for r in c['coordinates']],fmt='.',color=colors[j],label=c['from_case']+' → '+c['to_case'],capsize=2,ms=4)
    zoom=max(max(c['max_mean_shift_upper_sd'] for c in comparisons)*1.5,1e-12);ax.axhline(0,ls='--',color='black',lw=.7);ax.set_ylim(-zoom,zoom);ax.set_xticks(ix,NAMES,rotation=60,ha='right');ax.set(ylabel='Mean shift / baseline posterior SD',title='Paired shifts with ±2 batch-MCSE (zoom)');ax.text(.02,.97,'Gate: ±0.1 SD, outside zoom',transform=ax.transAxes,va='top',fontsize=9);ax.legend(fontsize=6,ncol=2,loc='lower left')
    ax=axes[1,0];labels=[c['from_case']+' → '+c['to_case'] for c in comparisons];ix=np.arange(len(labels));ax.semilogy(ix,[max(c['max_model_distance'],1e-16) for c in comparisons],'o-',label='Max per-state model distance');ax.semilogy(ix,[max(c['weighted_predictive_mean_distance'],1e-16) for c in comparisons],'s-',label='Weighted predictive-mean distance');ax.axhline(.1,ls='--',color='black',label='Gate');ax.set_xticks(ix,labels,rotation=40,ha='right');ax.set(ylabel='Noise-weighted distance',title='All 8192 sampled configurations');ax.legend(fontsize=8)
    ax=axes[1,1];ix=np.arange(len(names));ax.bar(ix,[min(overlap[n]['chain_overlap_fractions']) for n in names],color=colors[:len(names)]);ax.axhline(.9,ls='--',color='black',label='Overlap gate');ax.set_xticks(ix,names,rotation=35,ha='right');ax.set(ylim=(0,1.05),ylabel='Minimum per-chain overlap fraction',title='Importance overlap, not MCMC ESS');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Posterior numerical refinement: '+('PASS' if passed else 'declared gates NOT met')+'\nFour existing chains × 2048 draws; fixed data, priors, bands and 64 latent coordinates',fontsize=13);fig.savefig(root/'refinement-overview.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for j,ax in enumerate(axes.flat):
        if j==5:ax.axis('off');continue
        values=x[:,:,j].ravel();bins=np.linspace(values.min(),values.max(),40)
        for k,name in enumerate(names):ax.hist(values,bins=bins,weights=mom[name][2].ravel(),density=True,histtype='step',label=name,color=colors[k])
        ax.set(xlabel=NAMES[j],ylabel='Reweighted density');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Numerical sensitivity of scalar marginals; not independent refined-target chains');fig.savefig(root/'scalar-stability.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(passed=passed,baseline_replay_error=baseline_pred_error,comparisons=[{k:c[k] for k in ['from_case','to_case','passed','max_model_distance','max_mean_shift_sd','max_mean_shift_upper_sd','max_quantile_shift_sd']} for c in comparisons]),indent=2))
if __name__=='__main__':main()
