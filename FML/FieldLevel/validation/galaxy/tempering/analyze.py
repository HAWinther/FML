#!/usr/bin/env python3
"""Audit configurable replica transport; a connectivity pass is not cold convergence."""
from pathlib import Path
import argparse,csv,hashlib,json,os
from fractions import Fraction
import numpy as np

def beta_label(value):
    if value==0:return '0'
    if value==1:return '1'
    q=Fraction(value).limit_denominator(4096)
    return f'{q.numerator}/{q.denominator}' if abs(float(q)-value)<1e-12 else f'{value:g}'

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);ap.add_argument('--baseline',type=Path);ap.add_argument('--previous',type=Path);args=ap.parse_args();root=args.campaign.resolve();baseline=args.baseline.resolve() if args.baseline else None;plan=json.loads((root/'plan.json').read_text());identity=json.loads((root/'identity.json').read_text());betas=plan['betas'];k=len(betas);edges=k-1;passes=plan.get('exchange_passes',1)
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==identity['plan_sha256']
    summaries=json.loads((root/'completed.json').read_text());assert len(summaries)==plan['ensembles']
    assert json.loads((root/'restart-replay.json').read_text())['passed']
    target_bytes=((baseline/'chain0/target.json') if baseline else (root/'ensemble0/target.json')).read_bytes();target=json.loads(target_bytes);
    if plan.get('seed_layout')==3:
        assert target['mock']==dict(id=plan['mock_id'],truth_seed=plan['truth_seed'],noise_seed=plan['noise_seed'],initial_seed_base=plan['initial_seed_base']) and target['truth_parameters']==plan['truth_parameters']
    z=np.array(target['data']);data=z[:,0]+1j*z[:,1];weights=np.array(target['weights']);variance=plan['noise_power']/plan['boxsize']**3
    histories=[];times=[];amplitudes=[];energies=[];attempts=[];accepted=[];trips=[];gate_rows=[];prior_energies=[];worst_quad=0
    for e,s in enumerate(summaries):
        assert s['completed_iterations']==plan['warmup']+plan['samples'] and s['completed_samples']==plan['samples'] and s['betas']==betas and s.get('exchange_passes',1)==passes
        assert (root/f'ensemble{e}/target.json').read_bytes()==target_bytes
        phases=np.zeros(k,dtype=int);roundtrips=np.zeros(k,dtype=int);a=np.zeros(edges,dtype=int);b=a.copy();history=[];time=[];alpha=[];quadratics=[];previous=None;trace_rows=[]
        for draw in range(plan['samples']):
            row=json.loads((root/f'ensemble{e}/draws/draw-{draw}.rank-0.json').read_text());assert row['draw']==draw and row['rank']==0
            labels=np.array(row['labels'],dtype=int);assert sorted(labels.tolist())==list(range(k))
            rp=np.array(row['replica_parameters']);p=np.array(row['parameters']);white=np.array(row['white']);q=np.array(row['quadratic']);pred=np.array(row['prediction'])
            assert rp.shape==(k,5) and p.shape==(5,) and white.shape==(64,) and q.shape==(k,) and np.isfinite(rp).all() and np.isfinite(white).all() and np.isfinite(q).all() and np.all(q>=0) and np.array_equal(p,rp[0])
            assert np.all(rp>=np.array([.5,0,-4,-4,-100])) and np.all(rp<=np.array([1.5,4,4,4,100]))
            aa=np.array(row['swap_attempts'],dtype=int);bb=np.array(row['swap_accepted'],dtype=int)
            assert aa.shape==(edges,) and bb.shape==(edges,) and np.all(bb>=0) and np.all(bb<=aa)
            if passes==1:trace=[dict(parity=(plan['warmup']+draw)%2,attempts=aa.tolist(),accepted=bb.tolist(),labels=labels.tolist())]
            else:trace=row['exchange_trace'];assert len(trace)==passes
            start=np.array(trace[0]['labels'],dtype=int)
            for j in np.flatnonzero(trace[0]['accepted']):start[j],start[j+1]=start[j+1],start[j]
            if previous is not None:assert np.array_equal(start,previous),'Local updates changed walker labels'
            if passes>1:
                refreshed=int(row['refreshed_prior_label']);assert refreshed==start[-1]
                if phases[refreshed]==1:phases[refreshed]=2
            moved=start.copy();aggregate_a=np.zeros(edges,dtype=int);aggregate_b=aggregate_a.copy()
            for pass_index,event in enumerate(trace):
                parity=(plan['warmup']+draw+pass_index)%2;assert event['parity']==parity
                pa=np.array(event['attempts'],dtype=int);pb=np.array(event['accepted'],dtype=int);expected=np.array([j%2==parity for j in range(edges)],dtype=int)
                assert np.array_equal(pa,expected) and pb.shape==(edges,) and np.all(pb>=0) and np.all(pb<=pa)
                for j in np.flatnonzero(pb):moved[j],moved[j+1]=moved[j+1],moved[j]
                after=np.array(event['labels'],dtype=int);assert np.array_equal(after,moved) and sorted(after.tolist())==list(range(k))
                cold=int(after[0]);hot=int(after[-1])
                if phases[cold]==2:roundtrips[cold]+=1
                phases[cold]=1
                if passes==1 and phases[hot]==1:phases[hot]=2
                aggregate_a+=pa;aggregate_b+=pb;history.append(after.copy());time.append(draw+(pass_index+1)/passes)
            assert np.array_equal(moved,labels) and np.array_equal(aggregate_a,aa) and np.array_equal(aggregate_b,bb)
            previous=labels.copy();rebuilt=.5*np.sum(weights*np.abs(pred[:,0]+1j*pred[:,1]-data)**2/variance);error=abs(rebuilt-q[0])/max(1,abs(rebuilt));worst_quad=max(worst_quad,error);assert error<1e-11
            before=q.copy()
            for event in reversed(trace):
                for j in np.flatnonzero(event['accepted']):before[j],before[j+1]=before[j+1],before[j]
            prior_energies.append(before[-1]);a+=aa;b+=bb;alpha.append(p[0]);quadratics.append(q);trace_rows.append([draw,*p,int(labels[0]),q[0]])
        assert a.tolist()==s['swap_attempts'] and b.tolist()==s['swap_accepted'] and roundtrips.tolist()==s['roundtrips'] and phases.tolist()==s['transport_phase']
        total=plan['warmup']+plan['samples'];n=plan['samples']
        assert all(r['cache_builds']==5*total and r['scalar_attempts']==4*n and r['rescale_attempts']==4*n and r['prior_refreshes']==0 for r in s['replicas'][:-1]);assert s['replicas'][-1]['prior_refreshes']==total
        attempts.append(a);accepted.append(b);trips.append(roundtrips);histories.append(np.array(history));times.append(np.array(time));amplitudes.append(np.array(alpha));energies.append(np.array(quadratics))
        gate_rows.append(dict(ensemble=e,each_edge_exchanged=bool(np.all(b>=plan['gates']['min_accepted_per_edge_per_ensemble'])),roundtrips=int(roundtrips.sum()),roundtrip_gate=bool(roundtrips.sum()>=plan['gates']['min_cold_prior_cold_roundtrips_per_ensemble']),divergence_gate=all(r['divergences']==0 for r in s['replicas'][:-1])))
        with (root/f'cold-ensemble{e}.csv').open('w') as f:writer=csv.writer(f);writer.writerow(['draw','sigma8_ratio','b_delta','b_delta2','b_K2','b_laplacian','walker_label','base_quadratic']);writer.writerows(trace_rows)
    pooled_attempts=np.array(attempts).sum(axis=0);pooled_accepted=np.array(accepted).sum(axis=0);rates=pooled_accepted/pooled_attempts;rate_gate=bool(np.all(rates>=plan['gates']['min_pooled_edge_acceptance']))
    passed=rate_gate and all(r['each_edge_exchanged'] and r['roundtrip_gate'] and r['divergence_gate'] for r in gate_rows)
    summary=dict(scope=plan['scope'],connectivity_passed=passed,posterior_convergence_claim=False,ensemble_gates=gate_rows,betas=betas,exchange_passes=passes,prior_refresh_qualified_roundtrips=passes>1,pooled_swap_attempts=pooled_attempts.tolist(),pooled_swap_accepted=pooled_accepted.tolist(),pooled_swap_rates=rates.tolist(),pooled_rate_gate=rate_gate,roundtrips_by_ensemble=[v.tolist() for v in trips],summaries=summaries,max_cold_quadratic_reconstruction_error=worst_quad,serialized_targets_match_baseline=True,exact_full_ensemble_replay=True,summed_main_process_seconds=sum(json.loads(p.read_text())['seconds'] for p in root.glob('ensemble*-after*.command.json')),limits=plan['limits']+['Connectivity is not sufficient posterior mixing evidence','No excluded ensembles, retuned gates, extra sweeps or pooled hot/replay draws'])
    if 'prefix' in plan:
        imported=json.loads((root/'prefix-import.json').read_text());assert imported['complete'] and imported['retained_per_chain']==plan['prefix']['retained_per_chain'] and imported['plan_sha256']==identity['plan_sha256']
        summary['prefix_import']=imported;summary['new_main_process_seconds']=summary['summed_main_process_seconds']
        prefix_cost=imported.get('source_summed_main_process_seconds')
        summary['inclusive_main_process_seconds']=summary['summed_main_process_seconds']+prefix_cost if prefix_cost is not None else None
        summary['limits'].append('Includes previously examined cold prefixes exactly once; not an independent replication')
    if plan.get('seed_layout')==3:summary['target_reference_scope']='Same independently generated mock across its four ensembles; no match to original mock asserted'
    prior=np.array(prior_energies);candidate=[]
    for beta in [1/16,1/32,1/64,1/128,1/256,1/512,1/1024,0]:
        logw=-beta*prior;w=np.exp(logw-logw.max());ess=w.sum()**2/(w@w);candidate.append(dict(beta=beta,prior_weight_ess=float(ess),fraction=float(ess/len(w))))
    overlap=dict(scope='Exploratory prior overlap only, no posterior weights or modified gate',prior_draws=len(prior),prior_quadratic_quantiles=np.quantile(prior,[0,.05,.5,.95,1]).tolist(),candidates=candidate);summary['exploratory_prior_overlap']=overlap
    (root/'prior-overlap-diagnostic.json').write_text(json.dumps(overlap,indent=2)+'\n')
    if args.previous:
        old=json.loads((args.previous.resolve()/'summary.json').read_text());summary['previous_comparison']=dict(previous_betas=old['betas'],previous_exchange_passes=old.get('exchange_passes',1),previous_connectivity_passed=old['connectivity_passed'],previous_pooled_prior_edge_rate=old['pooled_swap_rates'][-1],previous_summed_process_seconds=old['summed_main_process_seconds'],process_seconds_ratio=summary['summed_main_process_seconds']/old['summed_main_process_seconds'],limits='Ladder and exchange frequency both change; not an isolated or matched-compute comparison')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab20').colors;fig,axes=plt.subplots(max(2,plan['ensembles']),2,figsize=(14,4.5*max(2,plan['ensembles'])),layout='constrained');edge=np.arange(edges);edge_labels=[beta_label(betas[j])+' ↔ '+beta_label(betas[j+1]) for j in range(edges)]
    ax=axes[0,0]
    for e in range(plan['ensembles']):
        bar=ax.bar(edge+(e-(plan['ensembles']-1)/2)*(.8/plan['ensembles']),accepted[e]/attempts[e],.8/plan['ensembles'],label=f'Ensemble {e+1}',color=colors[2*e])
        for j,b in enumerate(bar):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f'{accepted[e][j]}/{attempts[e][j]}',ha='center',fontsize=6)
    ax.axhline(.05,color='black',ls='--',label='Pooled-rate gate');ax.set_xticks(edge,edge_labels,rotation=45,ha='right');ax.set_ylim(0,1.15);ax.set(ylabel='Measured exchange acceptance',title='Adjacent exchanges');ax.legend(fontsize=8)
    ax=axes[1,0]
    for e in range(plan['ensembles']):ax.plot(np.arange(plan['samples'])+1,amplitudes[e],lw=.8,color=colors[2*e],label=f'Cold ensemble {e+1}')
    ax.axhline(1,color='black',ls='--',lw=.8);ax.set(xlabel='Measured sweep',ylabel='sigma8 / reference sigma8',title='Cold amplitude traces; no convergence claim');ax.legend(fontsize=8)
    for e,ax in enumerate(axes[:plan['ensembles'],1]):
        positions=np.argsort(histories[e],axis=1)
        for label in range(k):ax.step(times[e],positions[:,label],where='mid',lw=.7,color=colors[label%len(colors)])
        tick=[beta_label(v) for v in betas];tick[0]+=' (cold)';tick[-1]+=' (prior)';ax.set_yticks(np.arange(k),tick,fontsize=8);ax.set_ylim(k-.8,-.2);ax.set(xlabel='Measured sweep (all exchange passes)',ylabel='Likelihood beta',title=f'Ensemble {e+1}: {trips[e].sum()} refreshed-prior round trips')
    for row in range(2,axes.shape[0]):axes[row,0].axis('off')
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Replica connectivity: '+('PASS' if passed else 'declared gates NOT met')+f"\n{plan['ensembles']} ensembles × {k} replicas; {passes} exchange passes; {plan['warmup']} warmup + {plan['samples']} measured sweeps",fontsize=14);fig.savefig(root/'connectivity.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,plan['ensembles'],squeeze=False,figsize=(6.5*plan['ensembles'],4.5),layout='constrained')
    for e,ax in enumerate(axes[0]):
        value=np.log10(1+energies[e]);bins=np.linspace(value.min(),value.max(),40)
        for j,beta in enumerate(betas):ax.hist(value[:,j],bins=bins,density=True,histtype='step',color=colors[j%len(colors)],label='beta='+beta_label(beta))
        ax.set(xlabel='log10(1 + original likelihood quadratic)',ylabel='Descriptive density',title=f'Ensemble {e+1}: energy overlap');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.savefig(root/'energy-overlap.png',dpi=150);plt.close(fig)
    print(json.dumps({key:summary[key] for key in ['connectivity_passed','ensemble_gates','pooled_swap_rates','roundtrips_by_ensemble','summed_main_process_seconds']},indent=2))
if __name__=='__main__':main()
