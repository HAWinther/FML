"""Compare identical-target pilot kernels using actual PM evaluation counts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--allow-trajectory-change',action='store_true');p.add_argument('--allow-budget-change',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.allow_budget_change and not a.allow_trajectory_change:p.error('--allow-budget-change requires --allow-trajectory-change')
def load(root):
 plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'analysis/summary.json').read_text())
 diag=np.atleast_1d(np.genfromtxt(root/'science/diagnostics.csv',names=True,delimiter=',',dtype=None,encoding='utf8'));metadata=summary['metadata']
 if len(metadata)!=plan['chains'] or any(m[k]!=plan[k] for m in metadata for k in ['n','m','steps','warmup','draws','leapfrog']):raise ValueError('Saved plan and native metadata differ')
 for i,m in enumerate(metadata):
  with (root/'science'/('chain'+str(i)+'.meta.csv')).open() as f:native={k:float(v) for k,v in next(csv.DictReader(f)).items()}
  if m!=native:raise ValueError('Saved summary and native metadata differ')
 setup=0;seconds=0.
 if (root/'science/metric.meta.csv').exists():
  with (root/'science/metric.meta.csv').open() as f:metric=next(csv.DictReader(f))
  setup=int(metric['evaluations']);seconds=float(metric['seconds'])
 summary['metric_setup_evaluations']=setup
 return plan,summary,diag,sum(int(m['evaluations']) for m in metadata)+setup,sum(m['seconds'] for m in metadata)+seconds
bp,bs,bd,bc,bt=load(a.baseline);cp,cs,cd,cc,ct=load(a.candidate)
for key in ['n','m','steps','active','chains','warmup','epsilon','adapt_step_size','adapt_mass','target_acceptance','divergence_threshold','seed0','gates','data_sha256','map_seed_sha256']:
 if bp[key]!=cp[key]:raise ValueError('Target/protocol mismatch: '+key)
protocol_changes={k:dict(baseline=bp[k],candidate=cp[k]) for k in ['draws','leapfrog','predictive_stride'] if bp[k]!=cp[k]}
if protocol_changes and not a.allow_trajectory_change:raise ValueError('Sampling protocol differs; use --allow-trajectory-change explicitly')
if bp.get('scheme')==cp.get('scheme')=='preconditioned' and bp['metric_sha256']!=cp['metric_sha256']:raise ValueError('Fixed field metrics differ')
budget_ratio=cc/bc
if a.allow_trajectory_change and not a.allow_budget_change and not .95<=budget_ratio<=1.05:raise ValueError('Actual total PM budgets differ by more than 5 percent')
if not bs['consistency_passed'] or not cs['consistency_passed']:raise ValueError('Consistency checks failed')
if not np.array_equal(bd['feature'],cd['feature']):raise ValueError('Feature mismatch')
rows=[]
for i,name in enumerate(bd['feature']):
 row=dict(feature=str(name),baseline_bulk=float(bd['ess_bulk'][i]),candidate_bulk=float(cd['ess_bulk'][i]),baseline_tail=float(bd['ess_tail'][i]),candidate_tail=float(cd['ess_tail'][i]),baseline_rhat=float(bd['rhat'][i]),candidate_rhat=float(cd['rhat'][i]))
 for kind in ['bulk','tail']:
  row['baseline_'+kind+'_per_pm']=row['baseline_'+kind]/bc;row['candidate_'+kind+'_per_pm']=row['candidate_'+kind]/cc;row[kind+'_efficiency_ratio']=row['candidate_'+kind+'_per_pm']/row['baseline_'+kind+'_per_pm']
 rows.append(row)
with (a.output/'efficiency.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
result=dict(same_target=True,matched_pm_budget=.95<=budget_ratio<=1.05,budget_change_authorized=a.allow_budget_change,protocol_changes=protocol_changes,total_evaluation_budget_ratio=budget_ratio,baseline=dict(metric_setup_evaluations=bs['metric_setup_evaluations'],evaluations=bc,sampler_elapsed_seconds_sum=bt,mixing_passed=bs['mixing_passed'],**bs['diagnostic_extrema']),candidate=dict(metric_setup_evaluations=cs['metric_setup_evaluations'],evaluations=cc,sampler_elapsed_seconds_sum=ct,mixing_passed=cs['mixing_passed'],**cs['diagnostic_extrema']),
 median_bulk_efficiency_ratio=float(np.median([r['bulk_efficiency_ratio'] for r in rows])),median_tail_efficiency_ratio=float(np.median([r['tail_efficiency_ratio'] for r in rows])),
 worst_bulk_efficiency_ratio=float(np.min(cd['ess_bulk'])/cc/(np.min(bd['ess_bulk'])/bc)),selected=[r for r in rows if r['feature'] in ['A','N','potential','norm2']],
 limits=['ESS per total PM forward/adjoint evaluations includes metric setup and warmup cost; predictive-only forwards excluded equally.','Same finite N16 target; retained-draw/trajectory protocol changes are listed explicitly; stochastic pilot estimates, not asymptotic speed guarantees.','Scalar slice sweeps cost CPU but zero extra PM evaluations; aggregate sampler elapsed seconds are not CPU resource counters or campaign wall time.','Criteria unchanged: all Rhat <1.05, bulk ESS >400, tail ESS >200, acceptance >.5, zero retained divergences.'])
for name,diag,plan in [('baseline',bd,bp),('candidate',cd,cp)]:
 result[name]['feature_failures']=dict(bulk=int(np.sum(diag['ess_bulk']<=plan['gates']['ess_bulk'])),tail=int(np.sum(diag['ess_tail']<=plan['gates']['ess_tail'])),rhat=int(np.sum(diag['rhat']>=plan['gates']['rhat'])))
(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,key in zip(axes[:2],['bulk','tail']):
 baseline=np.array([r['baseline_'+key+'_per_pm'] for r in rows]);candidate=np.array([r['candidate_'+key+'_per_pm'] for r in rows]);ax.scatter(baseline,candidate,s=13,alpha=.5);lo=min(baseline.min(),candidate.min())*.8;hi=max(baseline.max(),candidate.max())*1.2;ax.plot([lo,hi],[lo,hi],'k:',lw=1);ax.set(xscale='log',yscale='log',xlabel='Baseline: ESS / PM evaluation',ylabel='Candidate: ESS / PM evaluation',title=key.capitalize()+' efficiency, all 248 features')
 for axis in [ax.xaxis,ax.yaxis]:
  axis.set_major_locator(FixedLocator([.02,.05,.1,.2]));axis.set_major_formatter(FuncFormatter(lambda v,pos:format(v,'.2g')));axis.set_minor_formatter(NullFormatter())
 for r in result['selected']:ax.annotate(r['feature'],(r['baseline_'+key+'_per_pm'],r['candidate_'+key+'_per_pm']),fontsize=8)
selected=result['selected'];x=np.arange(len(selected));axes[2].bar(x-.18,[r['bulk_efficiency_ratio'] for r in selected],width=.36,label='Bulk');axes[2].bar(x+.18,[r['tail_efficiency_ratio'] for r in selected],width=.36,label='Tail');axes[2].axhline(1,color='black',ls=':');axes[2].set(xticks=x,xticklabels=[r['feature'] for r in selected],ylabel='Candidate / baseline efficiency',title='Selected features');axes[2].legend(fontsize=8)
title=('L'+str(bp['leapfrog'])+', '+str(bp['draws'])+' draws vs L'+str(cp['leapfrog'])+', '+str(cp['draws'])+' draws') if a.allow_trajectory_change else (bp.get('scheme','joint')+' vs '+cp.get('scheme','joint'));fig.suptitle('Identical N16 target: '+title+'; '+('larger budget' if not .95<=budget_ratio<=1.05 else 'matched budget')+'; setup included');fig.savefig(a.output/'efficiency.png',dpi=170);fig.savefig(a.output/'efficiency.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
