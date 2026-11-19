#!/usr/bin/env python3
"""Summarize every frozen conditional probe; never treat these as posterior chains."""
from pathlib import Path
import argparse,csv,json,fcntl,os
import numpy as np
from run import verify,sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--exe',type=Path,required=True);a=p.parse_args();root=a.campaign.resolve()
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=verify(root,a.exe.resolve())
        if not json.loads((root/'completed.json').read_text())['complete']:raise ValueError('Full bounded probe required')
        datasets=[];allchecks=True;frames=0;native_matches=0;worst_gradient=0;totalrows=0
        for mid in plan['mocks']:
            prefix=root/f'results/mock{mid}';receipt=json.loads(prefix.with_suffix('.json').read_text())
            if receipt['sha256']!={p.name:sha(p) for p in [prefix.with_suffix('.csv'),prefix.with_suffix('.checks.csv')]}:raise ValueError('Probe result checksum mismatch')
            rows=list(csv.DictReader(prefix.with_suffix('.csv').open()));checks=list(csv.DictReader(prefix.with_suffix('.checks.csv').open()))
            if len(checks)!=12 or len(rows)!=12*plan['momentum_trials']*len(plan['cases']):raise ValueError('Incomplete result budget')
            expected={(e,draw,trial,c['name']) for e in range(4) for draw in plan['draws'] for trial in range(plan['momentum_trials']) for c in plan['cases']}
            if {(int(r['ensemble']),int(r['draw']),int(r['trial']),r['case']) for r in rows}!=expected:raise ValueError('Probe row identities changed')
            gate=all(int(c['gradient_sweeps_passed'])==1 and int(c['gradient_checks'])==8 and int(c['skipped_bounds'])==0 and int(c['native_matches'])==len(plan['cases']) and float(c['prediction_error'])<=plan['forward_tolerance'] and float(c['quadratic_error'])<=plan['quadratic_tolerance'] for c in checks)
            allchecks &= gate;frames+=len(checks);native_matches+=sum(int(c['native_matches']) for c in checks);worst_gradient=max(worst_gradient,max(float(c['gradient_best_max_error']) for c in checks));totalrows+=len(rows)
            cases=[]
            for case in plan['cases']:
                selected=[r for r in rows if r['case']==case['name']]
                if any(float(r['step'])!=case['step'] or int(r['leaps'])!=case['leaps'] or int(r['current_metric'])!=case['current'] for r in selected):raise ValueError('Native case settings differ')
                acceptance=np.array([float(r['acceptance']) for r in selected]);div=np.array([int(r['divergences']) for r in selected]);energy=np.array([float(r['delta_H']) for r in selected]);cost=np.array([int(r['gradient_evaluations']) for r in selected]);jump=np.array([float(r['jump_rms']) for r in selected])
                if not np.isfinite(acceptance).all() or np.any((acceptance<0)|(acceptance>1)) or not np.isfinite(jump).all():raise ValueError('Invalid finite probe diagnostics')
                finite_energy=energy[np.isfinite(energy)]
                # One paired seed per state/trial is reused for every case.
                baseline={ (int(r['ensemble']),int(r['draw']),int(r['trial'])):float(r['acceptance']) for r in rows if r['case']=='baseline'}
                paired=[float(r['acceptance'])-baseline[(int(r['ensemble']),int(r['draw']),int(r['trial']))] for r in selected]
                cases.append(dict(name=case['name'],trajectories=len(selected),divergences=int(div.sum()),mean_acceptance=float(acceptance.mean()),median_acceptance=float(np.median(acceptance)),mean_paired_acceptance_change=float(np.mean(paired)),nonfinite_energy_errors=int(np.sum(~np.isfinite(energy))),finite_energy_error_q05_q50_q95=np.quantile(finite_energy,[.05,.5,.95]).tolist() if finite_energy.size else None,mean_jump_rms=float(jump.mean()),mean_squared_jump_per_gradient=float(np.sum(jump*jump)/cost.sum()),gradient_evaluations=int(cost.sum())))
            datasets.append(dict(id=mid,checks_passed=gate,cases=cases))
        report=dict(scope=plan['scope'],complete=True,all_controls_passed=bool(allchecks),states=frames,conditional_trajectories=totalrows,native_exact_matches=native_matches,max_best_scaled_gradient_error=worst_gradient,datasets=datasets,source_pilot_unchanged=True,posterior_mixing_claim=False,limits=plan['limits'])
        write(root/'summary.json',report);plot(root,report,plan)
        print(json.dumps(report,indent=2))

def plot(root,report,plan):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');x=np.arange(8);colors=plt.get_cmap('tab10').colors
    for k,case in enumerate(plan['cases']):
        rows=[d['cases'][k] for d in report['datasets']]
        axes[0].plot(x,[r['mean_acceptance'] for r in rows],'o-',ms=4,color=colors[k],label=case['name'])
        axes[1].plot(x,[r['divergences'] for r in rows],'o-',ms=4,color=colors[k],label=case['name'])
        axes[2].plot(x,[r['mean_squared_jump_per_gradient'] for r in rows],'o-',ms=4,color=colors[k],label=case['name'])
    axes[0].set(ylabel='Mean proposal acceptance probability',ylim=(-.02,1.02))
    axes[1].set(ylabel='Divergent field trajectories (of 96 per dataset)')
    axes[2].set(ylabel='Accepted white-field jump RMS² / gradient call',yscale='log')
    for ax in axes:ax.set_xticks(x,[str(d['id']) for d in report['datasets']]);ax.set_xlabel('Original pilot dataset');ax.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=7)
    fig.suptitle('Conditional HMC stability: fixed saved states, paired momenta, trajectory length 1.2\n'+('All derivative/native controls pass' if report['all_controls_passed'] else 'Controls FAIL; comparisons require investigation')+'; local jump diagnostics do not establish chain mixing')
    fig.savefig(root/'stability-overview.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
