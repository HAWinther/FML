#!/usr/bin/env python3
"""Report the declared cold posterior campaign, including failed gates; never pool replay draws."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);ap.add_argument('--diagnostics',type=Path,required=True);args=ap.parse_args()
    root=args.campaign.resolve();plan=json.loads((root/'plan.json').read_text())
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==json.loads((root/'identity.json').read_text())['plan_sha256']
    summaries=json.loads((root/'completed.json').read_text());assert len(summaries)==4
    assert json.loads((root/'restart-replay.json').read_text())['passed']
    names=['sigma8_ratio','b_delta','b_delta2','b_K2','b_laplacian','cos100','sin100','cos110','sin110','cos001','sin001']
    coords=np.indices((4,4,4)).reshape(3,-1).T
    basis=[]
    for wave in [(1,0,0),(1,1,0),(0,0,1)]:
        phase=2*np.pi*(coords@wave)/4
        basis.extend([np.sqrt(2/64)*np.cos(phase),np.sqrt(2/64)*np.sin(phase)])
    targets=[];chains=[];predictions=[]
    for c in range(4):
        summary=summaries[c];assert summary['completed_samples']==plan['samples'] and summary['completed_iterations']==plan['warmup']+plan['samples']
        target=(root/f'ensemble{c}/target.json').read_bytes();targets.append(target)
        rows=[json.loads((root/f'ensemble{c}/draws/draw-{j}.rank-0.json').read_text()) for j in range(plan['samples'])]
        assert all(row['draw']==j and row['rank']==0 for j,row in enumerate(rows))
        white=np.array([r['white'] for r in rows]);p=np.array([r['parameters'] for r in rows]);pred=np.array([r['prediction'] for r in rows])
        assert white.shape==(plan['samples'],64) and p.shape==(plan['samples'],5)
        values=np.column_stack([p,white@np.array(basis).T]);assert np.isfinite(values).all() and np.isfinite(pred).all()
        chains.append(values);predictions.append(pred[:,:,0]+1j*pred[:,:,1])
        with (root/f'joint.chain{c}.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['draw',*names]);writer.writerows([[j,*v] for j,v in enumerate(values)])
    assert len(set(targets))==1,'Chain targets differ'
    subprocess.run([str(args.diagnostics.resolve()),str(root/'joint'),str(plan['samples'])],check=True)
    diagnostics=[];gate=plan['posterior_gates']
    for r in csv.DictReader((root/'joint.diagnostics.csv').open()):
        j=int(r['projection']);d=dict(parameter=names[j],rhat=float(r['rhat']),bulk_ess=float(r['bulk_ess']),tail_ess=float(r['tail_ess']),status=r['status'])
        d['passed']=d['status']=='ok' and d['rhat']<gate['max_rhat'] and d['bulk_ess']>=gate['min_bulk_ess'] and d['tail_ess']>=gate['min_tail_ess']
        diagnostics.append(d)
    assert len(diagnostics)==11
    x=np.array(chains);pooled=x.reshape(-1,11);truth=np.array(plan['truth_parameters']);target=json.loads(targets[0]);pred=np.concatenate(predictions)
    residual=np.array(target['data']);data=residual[:,0]+1j*residual[:,1];weights=np.array(target['weights']);k=np.array(target['k']);kf=2*np.pi/plan['boxsize'];shell=np.rint((k/kf)**2).astype(int)
    norm=plan['boxsize']**3/plan['noise_power'];powers=np.abs(data[None,:]-pred)**2*norm;meanpower=np.abs(data-pred.mean(axis=0))**2*norm
    shell_rows=[]
    for sh in np.unique(shell):
        mask=shell==sh;v=np.average(powers[:,mask],axis=1,weights=weights[mask]);q=np.quantile(v,[.05,.5,.95]);shell_rows.append(dict(k=float(k[mask][0]),real_modes=float(weights[mask].sum()),posterior_residual_power_q05_q50_q95=q.tolist(),posterior_mean_residual_power=float(np.average(meanpower[mask],weights=weights[mask]))))
    connectivity=json.loads((root/'summary.json').read_text())
    report=dict(scope=plan['scope'],passed=all(d['passed'] for d in diagnostics) and connectivity['connectivity_passed'] and all(r['divergences']==0 for s in summaries for r in s['replicas'][:-1]),diagnostics=diagnostics,chains=summaries,pooled_descriptive_scalar_quantiles={name:np.quantile(pooled[:,j],[.05,.5,.95]).tolist() for j,name in enumerate(names[:5])},scalar_correlations=np.corrcoef(pooled[:,:5].T).tolist(),residual_shells=shell_rows,limits=['Synthetic same-model data; no coverage or real-catalog claim','Six field projections, not all field coordinates','Descriptive pooled quantiles are provisional when any mixing gates fail','Residuals use fitted data; not an independent predictive validation','Replay draws excluded; fixed budget not extended'],target_identical_across_chains=True,restart_replay=True)
    if 'prefix' in plan:report['limits'].append(f"Includes the previously examined {plan['prefix']['retained_per_chain']}-draw prefixes exactly once; not an independent replication")
    report['connectivity_passed']=connectivity['connectivity_passed']
    report['posterior_convergence_claim']=report['passed']
    (root/'posterior-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#167eb7','#d68131','#3a9a81','#9865b0']
    fig,axes=plt.subplots(6,2,figsize=(13,16),layout='constrained')
    for j,ax in enumerate(axes.flat):
        if j==11:ax.axis('off');continue
        for c in range(4):ax.plot(np.arange(plan['samples'])+1,x[c,:,j],color=colors[c],lw=.6,alpha=.8,label=f'Chain {c+1}')
        if j<5:ax.axhline(truth[j],ls='--',color='black',lw=1,label='Generating value')
        d=diagnostics[j];ax.set_title(f"{names[j]}: Rhat={d['rhat']:.3f}, bulk ESS={d['bulk_ess']:.0f}");ax.set_xlabel('Retained draw');ax.grid(alpha=.2)
    axes[0,0].legend(ncol=3,fontsize=8);fig.suptitle('Cold posterior validation: fixed budget, all retained draws',fontsize=15);fig.savefig(root/'traces.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(5,5,figsize=(12,12),layout='constrained')
    for i in range(5):
        for j in range(5):
            ax=axes[i,j]
            if i<j:ax.axis('off');continue
            for c in range(4):
                if i==j:ax.hist(x[c,:,j],bins=25,density=True,histtype='step',color=colors[c],label=f'Chain {c+1}')
                else:ax.scatter(x[c,::8,j],x[c,::8,i],s=3,alpha=.25,color=colors[c])
            ax.axvline(truth[j],color='black',ls='--',lw=.8)
            if i!=j:ax.axhline(truth[i],color='black',ls='--',lw=.8)
            if i==4:ax.set_xlabel(names[j])
            if j==0:ax.set_ylabel('Density' if i==0 else names[i])
    axes[0,0].legend(fontsize=7);fig.suptitle('Joint scalar distributions and degeneracies\nProvisional if convergence gates fail; dashed lines show generating values',fontsize=14);fig.savefig(root/'scalar-posterior.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    ax=axes[0];kk=np.array([s['k'] for s in shell_rows]);q=np.array([s['posterior_residual_power_q05_q50_q95'] for s in shell_rows]);ax.fill_between(kk,q[:,0],q[:,2],alpha=.2,label='Draw residual power, 5–95%');ax.plot(kk,q[:,1],'o-',label='Draw residual median');ax.plot(kk,[s['posterior_mean_residual_power'] for s in shell_rows],'s-',label='Residual of posterior mean');ax.axhline(1,color='black',ls='--',label='Generating noise power');ax.set_xlabel('k [h/Mpc]');ax.set_ylabel('Residual power / generating noise power');ax.set_title('Fitted-data residuals; few modes per shell');ax.legend(fontsize=8)
    ax=axes[1];xx=np.arange(11);ax.bar(xx,[d['bulk_ess'] for d in diagnostics],label='Bulk ESS');ax.bar(xx,[d['tail_ess'] for d in diagnostics],alpha=.4,label='Tail ESS');ax.axhline(400,color='black',ls='--',label='ESS gate');ax.set_xticks(xx,names,rotation=65,ha='right');ax.set_ylabel('Effective draws');ax.set_title('PASS' if report['passed'] else 'Fixed-budget mixing gates NOT met');ax.legend(fontsize=8)
    fig.savefig(root/'residuals-diagnostics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for c in range(4):
        axes[0,0].plot(np.arange(plan['samples'])+1,x[c,:,0],color=colors[c],lw=.7,label=f'Chain {c+1}')
        axes[0,1].hist(x[c,:,0],bins=30,density=True,histtype='step',color=colors[c],label=f'Chain {c+1}')
        axes[1,0].plot(np.arange(plan['samples'])+1,x[c,:,2],color=colors[c],lw=.7)
    axes[0,0].axhline(1,color='black',ls='--',lw=1);axes[0,0].set(xlabel='Retained draw',ylabel='sigma8 / reference sigma8',title=f"Amplitude: Rhat={diagnostics[0]['rhat']:.3f}, bulk ESS={diagnostics[0]['bulk_ess']:.1f}");axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].axvline(1,color='black',ls='--',lw=1);axes[0,1].set(xlabel='sigma8 / reference sigma8',ylabel='Density',title='Amplitude distributions'+(': unconverged' if not diagnostics[0]['passed'] else ''))
    axes[1,0].axhline(.3,color='black',ls='--',lw=1);axes[1,0].set(xlabel='Retained draw',ylabel='b_delta2',title='Chain-specific quadratic-bias traces')
    xx=np.arange(11);axes[1,1].bar(xx,[d['bulk_ess'] for d in diagnostics]);axes[1,1].axhline(400,color='black',ls='--',label='Declared ESS gate');axes[1,1].set_xticks(xx,names,rotation=60,ha='right');axes[1,1].set(ylabel='Bulk ESS',title=f"{sum(not d['passed'] for d in diagnostics)}/11 coordinate gates fail");axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Tempered cold field/amplitude/bias validation: '+('PASS' if report['passed'] else 'mixing gates NOT met')+f"\n4 chains × {plan['samples']} retained draws; dashed lines show generating values",fontsize=13)
    fig.savefig(root/'posterior-overview.png',dpi=150);plt.close(fig)
    print(json.dumps({'passed':report['passed'],'diagnostics':diagnostics},indent=2))
if __name__=='__main__':main()
