"""Read-only derivative/CIC/roundoff diagnosis; retain all measured scales."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,help='Projected fit_coefficients CSV for this endpoint');p.add_argument('--line-expectation',choices=['hidden-descent','measure'],default='hidden-descent');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def read(path):return np.atleast_1d(np.genfromtxt(path,names=True,delimiter=',',dtype=None,encoding='utf8'))
root=a.input/'science';metrics=read(root/'probe.metrics.csv');r=read(root/'probe.directions.csv');line=read(root/'probe.line.csv');cache=read(root/'probe.cache.csv');checks=[]
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
check(len(metrics)==2 and len(r)==168 and len(line)==40 and len(cache)==10,'complete planned probes')
reference=a.reference or Path(__file__).resolve().parent.parent/'pm_optimizer_validation/science/lbfgs.fit_coefficients.csv'
check(np.array_equal(np.loadtxt(root/'stalled.projected.csv',delimiter=','),np.loadtxt(reference,delimiter=',')[2:]),'captured projected state and gradient match reference failure bitwise')
for array in [metrics,r,line,cache]:
 for field in array.dtype.names:
  if array.dtype[field].kind in 'iuf':check(np.all(np.isfinite(array[field])),'finite '+field)
check(np.max(abs(r['trajectory_error']))==0,'observer/default trajectories identical')
check(np.max(abs(line['unit_trial_error']))==0,'unit-step failed trial reproduced exactly')
check(np.max(abs(metrics['saved_value_error']))<2e-10 and np.max(abs(metrics['saved_gradient_error']))<2e-10,'full-state snapshot replay')
check(max(np.max(abs(cache[k])) for k in ['value_error','gradient_error','scalar_error','value_only_offset'])<2e-10,'fresh/reused cache, value/full and budgets agree')
check(np.array_equal(r['value_delta'],r['manual_value_delta']) and np.array_equal(line['value_delta'],line['manual_value_delta']),'public and independently assembled raw target changes agree')
fd={};gates=[]
for name in ['stalled','initial']:
 fd[name]={}
 for d in [0,1,2]:
  v=r[(r['point']==name)&(r['direction']==d)];rows=[]
  for h in sorted(set(v['h']),reverse=True):
   plus=v[(v['h']==h)&(v['sign']==1)][0];minus=v[(v['h']==h)&(v['sign']==-1)][0];analytic=float(plus['analytic']);right=float(plus['stable_delta']/h);left=float(-minus['stable_delta']/h);central=.5*(right+left)
   rows.append(dict(h=float(h),analytic=analytic,left=left,right=right,central=central,error=abs(central-analytic)/max(1.,abs(analytic)),left_changes=int(minus['changed_axes']),right_changes=int(plus['changed_axes'])))
  # Same derivative-error thresholds as preceding campaigns; apply to measured
  # branch-preserving pairs, above the most extreme force roundoff regime.
  smooth=[x for x in rows if x['left_changes']==0 and x['right_changes']==0 and 1e-10<=x['h']<=1e-5]
  gate=len(smooth)>=2 and min(x['error'] for x in smooth)<2e-5 and sum(x['error']<1e-4 for x in smooth)>=2
  gates.append(dict(point=name,direction=d,eligible_pairs=len(smooth),best_error=min((x['error'] for x in smooth),default=None),passed=bool(gate)))
  check(gate,name+f' direction {d}: refined same-stencil derivative','derivative')
  fd[name][str(d)]=rows
u0=float(metrics[metrics['point']=='stalled'][0]['potential'])
raw_pass=(u0+line['value_delta'])<=(u0+line['armijo_delta']);stable_pass=line['stable_delta']<=line['armijo_delta'];clear=stable_pass&(line['changed_axes']==0)
if a.line_expectation=='hidden-descent':
 check(np.sum(raw_pass)==0,'original raw line-search rejection reproduced')
 check(np.any(clear),'stable difference identifies stencil-preserving descent')
# Pick the largest tested stencil-preserving candidate with stable descent.
idx=int(np.flatnonzero(clear)[0]) if np.any(clear) else None;example={key:float(line[key][idx]) for key in line.dtype.names} if idx is not None else None
base=metrics[metrics['point']=='stalled'][0]
public_summary=None
if (root/'public.line.csv').exists():
 public=read(root/'public.line.csv')
 check(np.array_equal(public['alpha'],line['alpha']),'public API uses identical reconstructed steps')
 check(np.allclose(public['stable_delta'],line['stable_delta'],rtol=1e-10,atol=1e-18),'public stable differences match independent assembly')
 check(np.max(abs(public['returned_value_error']))==0,'public API returns unchanged ordinary values')
 check(np.array_equal(public['stable_accept'].astype(bool),stable_pass),'public and independent stable acceptance agree')
 public_summary=dict(stable_passes=int(sum(public['stable_accept'])),raw_passes=int(sum(public['raw_accept'])),max_difference=float(np.max(abs(public['stable_delta']-line['stable_delta']))))
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),derivative_gates=gates,public_api=public_summary,checks=checks,metrics=[{k:(str(v[k]) if metrics.dtype[k].kind in 'US' else float(v[k])) for k in metrics.dtype.names} for v in metrics],
 finite_differences=fd,cache_max_errors={k:float(np.max(abs(cache[k]))) for k in ['value_error','gradient_error','scalar_error','value_only_offset']},
 line_search=dict(raw_passes=int(sum(raw_pass)),stable_passes=int(sum(stable_pass)),same_stencil_stable_passes=int(sum(clear)),example=example),
 interpretation=['Inspect checks and finite_differences for measured same-stencil derivative agreement; failed checks are retained.',
                 'Line-search counts describe only the sampled direction and steps; they do not establish stationarity.',
                 'Stable differences use double-precision forward predictions, not a higher-precision force solution. No solver or assignment law is changed.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'derivatives.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['point','direction',*fd['stalled']['0'][0]],lineterminator='\n');writer.writeheader()
 for point,directions in fd.items():
  for d,rows in directions.items():
   for row in rows:writer.writerow(dict(point=point,direction=d,**row))
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
labels=['Negative gradient','Failed search direction','Random active direction']
for row,name in enumerate(['stalled','initial']):
 for d in range(3):
  ax=axes[row,d];v=fd[name][str(d)];h=np.array([x['h'] for x in v]);analytic=v[0]['analytic']
  for key,label,color in [('left','Left slope','tab:orange'),('right','Right slope','tab:blue')]:
   y=(np.array([x[key] for x in v])-analytic)/max(1.,abs(analytic));changes=np.array([x[key+'_changes'] for x in v]);ax.plot(h,y,'-',color=color,lw=1,label=label)
   ax.scatter(h,y,facecolors='none',edgecolors=color,s=24);ax.scatter(h[changes>0],y[changes>0],color=color,s=24)
  ax.axhline(0,color='black',ls=':',label='Adjoint');ax.set_xscale('log');ax.invert_xaxis();ax.set_yscale('symlog',linthresh=1e-7)
  ax.set(title=f'{name}: {labels[d]}',xlabel='White-field perturbation norm h',ylabel='Slope error / max(1, |adjoint|)');ax.legend(fontsize=7)
fig.suptitle('One-sided stable difference errors; filled markers change at least one CIC stencil axis')
fig.savefig(a.output/'derivatives.png',dpi=170);fig.savefig(a.output/'derivatives.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');alpha=line['alpha']
axes[0].plot(alpha,line['value_delta'],'o-',ms=3,label='Public value subtraction');axes[0].plot(alpha,line['stable_delta'],'o-',ms=3,label='Stable target difference');axes[0].plot(alpha,line['predicted_delta'],'k:',label='Adjoint linear prediction');axes[0].axhline(0,color='gray',lw=.7);axes[0].set_yscale('symlog',linthresh=1e-13);axes[0].set(ylabel='Objective change',title='The rejected search direction');axes[0].legend(fontsize=7)
axes[1].plot(alpha,line['changed_axes'],'o-',ms=3);axes[1].set(ylabel='Changed particle/node/axis stencils',title='CIC branch changes')
axes[2].plot(alpha,line['critical_distance'],'o-',ms=3);axes[2].axhline(0,color='black',ls=':');axes[2].set_yscale('symlog',linthresh=1e-10);axes[2].set(ylabel='Signed distance to nearest face [grid cells]',title=f"Node {int(base['node'])}, particle {int(base['particle'])}, axis {int(base['axis'])}")
for ax in axes:ax.set_xscale('log');ax.invert_xaxis();ax.set_xlabel('Original line-search step alpha')
fig.suptitle('Frozen failed L-BFGS endpoint; solver and force/assignment law unchanged')
fig.savefig(a.output/'line_search.png',dpi=170);fig.savefig(a.output/'line_search.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for d in [0,1,2]:
 v=r[(r['point']=='stalled')&(r['direction']==d)&(r['sign']==1)];axes[0].loglog(v['h'],np.maximum(v['changed_axes'],.3),'o-',ms=3,label=labels[d]);axes[1].plot(v['h'],v['critical_distance'],'o-',ms=3,label=labels[d])
axes[0].set(ylabel='Changed axes (zero displayed at 0.3)',title='Positive directional perturbations');axes[1].axhline(0,color='black',ls=':');axes[1].set_xscale('log');axes[1].set_yscale('symlog',linthresh=1e-10);axes[1].set(ylabel='Critical signed face distance [grid cells]',title='Nearest endpoint face')
for ax in axes:ax.invert_xaxis();ax.set_xlabel('White-field perturbation norm h');ax.legend(fontsize=7)
fig.savefig(a.output/'boundaries.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],failed=[c['check'] for c in checks if not c['passed']],cache=summary['cache_max_errors'],line_search=summary['line_search']),indent=2))
if not summary['passed']:raise SystemExit(1)
