#!/usr/bin/env python3
"""Analyze blocked output with the joint pilot's common monitors, then compare costs."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);ap.add_argument('--diagnostics',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);args=ap.parse_args()
    root=args.campaign.resolve();baseline=args.baseline.resolve()
    shared=Path(__file__).resolve().parent.parent/'joint/analyze.py'
    spec=importlib.util.spec_from_file_location('joint_analysis',shared);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(shared),str(root),'--diagnostics',str(args.diagnostics.resolve())];module.main()
    old=json.loads((baseline/'summary.json').read_text());new=json.loads((root/'summary.json').read_text())
    a=json.loads((baseline/'plan.json').read_text());b=json.loads((root/'plan.json').read_text())
    for key in ['mesh','product_mesh','density_mesh','boxsize','spectrum','noise_power','initial_band','matter_band','likelihood_band','truth_parameters','truth_seed','noise_seed','initial_seed_base','initial_parameters','chains','warmup','samples','gates']:
        assert a[key]==b[key],key
    assert (baseline/'chain0/target.json').read_bytes()==(root/'chain0/target.json').read_bytes()
    def seconds(directory):return sum(json.loads(p.read_text())['seconds'] for p in directory.glob('chain*-after*.command.json'))
    old_seconds=seconds(baseline);new_seconds=seconds(root);assert old_seconds>0 and new_seconds>0
    comparison=dict(same_serialized_target=True,same_initial_states=True,same_primary_draw_budget=True,same_gates=True,baseline_passed=old['passed'],blocked_passed=new['passed'],baseline_summed_process_seconds=old_seconds,blocked_summed_process_seconds=new_seconds,seconds_ratio=new_seconds/old_seconds,diagnostics=[])
    for left,right in zip(old['diagnostics'],new['diagnostics']):
        assert left['parameter']==right['parameter']
        comparison['diagnostics'].append(dict(parameter=right['parameter'],baseline_rhat=left['rhat'],blocked_rhat=right['rhat'],baseline_bulk_ess=left['bulk_ess'],blocked_bulk_ess=right['bulk_ess'],baseline_bulk_ess_per_second=left['bulk_ess']/old_seconds,blocked_bulk_ess_per_second=right['bulk_ess']/new_seconds))
    comparison['limits']=['Different kernels, integration settings and warmup adaptation; not an isolated one-setting ablation','ESS/s ratios are descriptive when either chain set fails stationarity/mixing gates','Summed main-chunk process wall seconds include initialization, cache construction and draw/checkpoint I/O; exclude preflight and replay','No pooled baseline or replay draws; no post-run tuning or extension']
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');x=np.arange(11);labels=[d['parameter'] for d in new['diagnostics']]
    for ax,key,title in [(axes[0],'rhat','Rank Rhat'),(axes[1],'bulk_ess','Bulk ESS')]:
        ax.plot(x,[d[key] for d in old['diagnostics']],'o-',label='Joint HMC baseline',color='#999999');ax.plot(x,[d[key] for d in new['diagnostics']],'o-',label='Blocked',color='#167eb7');ax.set_title(title);ax.axhline(1.01 if key=='rhat' else 400,color='black',ls='--',label='Gate');ax.set_xticks(x,labels,rotation=65,ha='right');ax.legend(fontsize=8)
    for c,color in enumerate(['#167eb7','#d68131','#3a9a81','#9865b0']):
        values=np.genfromtxt(root/f'joint.chain{c}.csv',delimiter=',',names=True);axes[2].plot(values['draw']+1,values['sigma8_ratio'],lw=.6,color=color,label=f'Chain {c+1}')
    axes[2].axhline(1,color='black',ls='--',lw=1);axes[2].set(xlabel='Retained draw',ylabel='sigma8 / reference sigma8',title='Blocked amplitude traces');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Blocked pilot: '+('mixing gates PASS' if new['passed'] else 'mixing gates NOT met')+'; unchanged target and primary draw budget',fontsize=13);fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
