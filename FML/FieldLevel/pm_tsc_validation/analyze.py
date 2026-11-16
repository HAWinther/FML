"""One matched-assignment TSC reconstruction; preserve every failed gate."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent.parent;plan=json.loads((a.input/'plan.json').read_text());checks=[]
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def meta(prefix,suffix):
 with Path(str(prefix)+'.'+suffix+'.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
mapped=a.output/'input';(mapped/'science').mkdir(parents=True,exist_ok=True);(mapped/'plan.json').write_text(json.dumps(dict(plan,budgets=[128]),indent=2)+'\n')
for f in (a.input/'science').glob('reference.*.csv'):(mapped/'science'/f.name).write_bytes(f.read_bytes())
for f in (a.input/'science').glob('tsc-lbfgs.*.csv'):(mapped/'science'/('fit-128'+f.name[len('tsc-lbfgs'):])).write_bytes(f.read_bytes())
report=a.output/'reconstruction'
with (a.output/'reconstruction.log').open('w') as log:r=subprocess.run([sys.executable,str(root/'pm_reconstruction_validation/analyze.py'),'--input',str(mapped),'--output',str(report)],stdout=log,stderr=subprocess.STDOUT)
check(r.returncode in [0,1] and (report/'summary.json').exists(),'reconstruction analysis completed')
measured=json.loads((report/'summary.json').read_text());check(measured['consistency_passed'],'physical and data consistency');check(measured['recovery_passed'],'coefficient recovery and correlation','recovery')
row=measured['rows'][-1];prefix=a.input/'science/tsc-lbfgs';row.update(meta(prefix,'fit_optimizer'));row.update(meta(prefix,'fit_change'))
check(row['forwards']==row['gradient_evaluations']+row['value_evaluations']+row['reference_evaluations'] and row['pullbacks']==row['gradient_evaluations'],'actual predictions and pullbacks match optimizer counters')
check(row['status']==0 and row['raw_gradient']<plan['gradient_tolerance'],'TSC reaches unchanged convergence tolerance','convergence')
change=np.atleast_1d(np.genfromtxt(str(prefix)+'.fit_change_trace.csv',delimiter=',',names=True));check(np.all(change['accepted_change'][1:]<=change['armijo_bound'][1:]),'all accepted stable changes satisfy Armijo')
fd=np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv',names=True,delimiter=','));fd_rows=[]
check(len(fd)==16 and np.all(np.isfinite(fd['numeric'])) and np.all(np.isfinite(fd['analytic'])),'complete finite truth-point derivative scan')
for probe in [0,1]:
 v=fd[fd['probe']==probe];error=abs(v['numeric']-v['analytic'])/np.maximum(1.,abs(v['analytic']));tail=error[-4:];check(np.min(tail)<2e-5 and np.sum(tail<1e-4)>=2,'TSC truth directional derivative '+str(probe),'derivative');fd_rows.append(dict(probe=probe,best_tail_error=float(np.min(tail))))
baseline=json.loads((root/'pm_stable_armijo_validation/summary.json').read_text())['rows']['stable-lbfgs']
summary=dict(passed=all(x['passed'] for x in checks),consistency_passed=all(x['passed'] for x in checks if x['kind']=='consistency'),checks=checks,row=row,baseline=baseline,derivatives=fd_rows,
 limits=['One fixed-parameter realization; broader refinement/calibration remains deferred.','CIC and TSC use matched assignment-specific reference mocks: different discrete targets, same seeds and physical ICs.','N128/M64/32-step reference is finite, not a continuum solution.','Wall times are indicative under uncontrolled host activity.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'comparison.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['assignment','iterations','status','raw_gradient','predictions','pullbacks','seconds','coefficient_error','correlation'])
 for name,v in [('CIC',baseline),('TSC',row)]:w.writerow([name,v['iterations'],v['status'],v['raw_gradient'],v['forwards'],v['pullbacks'],v['seconds'],v['relative_initial_error'],v['correlation']])
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for source,label in [(root/'pm_stable_armijo_validation/science/stable-lbfgs','CIC (line-search failed)'),(prefix,'TSC')]:
 t=np.atleast_1d(np.genfromtxt(str(source)+'.fit_trace.csv',names=True,delimiter=','));axes[0].semilogy(t['iteration'],t['gradient_norm'],'o-',ms=2,label=label)
axes[0].axhline(plan['gradient_tolerance'],color='gray',ls=':',label='Unchanged tolerance');axes[0].set(xlabel='Accepted iterations',ylabel='Raw max gradient',title='Convergence');axes[0].legend(fontsize=7)
x=np.arange(2);data=[baseline,row];axes[1].bar(x-.18,[v['forwards'] for v in data],.36,label='Predictions (including references)');axes[1].bar(x+.18,[v['pullbacks'] for v in data],.36,label='Pullbacks');axes[1].set(xticks=x,xticklabels=['CIC','TSC'],title='Actual model calls');axes[1].legend(fontsize=7)
axes[2].bar(x,[v['seconds'] for v in data]);axes[2].set(xticks=x,xticklabels=['CIC','TSC'],ylabel='MAP seconds',title='Indicative wall time')
fig.suptitle('Matched assignment-specific mocks; same seeds, physical ICs and numerical resolutions')
fig.savefig(a.output/'comparison.png',dpi=170);fig.savefig(a.output/'comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.8),layout='constrained')
for probe in [0,1]:
 v=fd[fd['probe']==probe];ax.loglog(v['h'],np.maximum(abs(v['numeric']-v['analytic'])/np.maximum(1.,abs(v['analytic'])),1e-16),'o-',label=['White direction','White + scalar direction'][probe])
ax.axhline(2e-5,color='gray',ls=':');ax.invert_xaxis();ax.set(xlabel='Perturbation h',ylabel='Scaled derivative error',title='TSC nonlinear posterior at truth');ax.legend(fontsize=8);fig.savefig(a.output/'derivatives.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],failed=[x['check'] for x in checks if not x['passed']],row=row),indent=2))
if not summary['passed']:raise SystemExit(1)
