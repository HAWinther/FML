"""Stable Armijo frozen-state checks and reconstruction work/convergence."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads((a.input/'plan.json').read_text());checks=[];rows={}
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def meta(prefix,suffix):
 with Path(str(prefix)+'.'+suffix+'.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
root=Path(__file__).resolve().parent.parent;prior=json.loads((root/'pm_optimizer_validation/summary.json').read_text());base=root/'pm_reconstruction_validation/analyze.py'
frozen=np.atleast_1d(np.genfromtxt(a.input/'science/frozen.line.csv',names=True,delimiter=','));old=np.atleast_1d(np.genfromtxt(root/'pm_stationarity_validation/science/probe.line.csv',names=True,delimiter=','))
check(len(frozen)==40 and np.array_equal(frozen['alpha'],old['alpha']),'complete frozen step sequence')
check(not np.any(frozen['raw_accept']) and np.any(frozen['stable_accept']),'previously rejected frozen descent accepted')
check(np.max(abs(frozen['stable_delta']-old['stable_delta']))<1e-20,'API stable changes reproduce independent diagnostic')
check(np.max(abs(frozen['returned_value_error']))<2e-10,'returned ordinary target matches value API')
for spec in plan['runs']:
 name=spec['name'];budget=spec['iterations'];prefix=a.input/'science'/name;mapped=a.output/(name+'-input');(mapped/'science').mkdir(parents=True,exist_ok=True);(mapped/'plan.json').write_text(json.dumps(dict(plan,budgets=[budget]),indent=2)+'\n')
 for f in (a.input/'science').glob('reference.*.csv'):(mapped/'science'/f.name).write_bytes(f.read_bytes())
 for f in (a.input/'science').glob(name+'.*.csv'):(mapped/'science'/('fit-'+str(budget)+f.name[len(name):])).write_bytes(f.read_bytes())
 report=a.output/name
 with (a.output/(name+'-analysis.log')).open('w') as f:r=subprocess.run([sys.executable,str(base),'--input',str(mapped),'--output',str(report)],stdout=f,stderr=subprocess.STDOUT)
 check(r.returncode in [0,1] and (report/'summary.json').exists(),name+' analysis completed')
 measured=json.loads((report/'summary.json').read_text());check(measured['consistency_passed'],name+' physical/data consistency');check(measured['recovery_passed'],name+' recovery','recovery')
 row=measured['rows'][-1];row.update(meta(prefix,'fit_optimizer'));row.update(meta(prefix,'fit_change'));rows[name]=row
 check(row['forwards']==row['gradient_evaluations']+row['value_evaluations']+row['reference_evaluations'],name+' actual predictions include references')
 change=np.atleast_1d(np.genfromtxt(str(prefix)+'.fit_change_trace.csv',names=True,delimiter=','));check(np.all(np.isfinite(change['accepted_change'])) and np.all(change['accepted_change'][1:]<=change['armijo_bound'][1:]),name+' all accepted changes meet Armijo')
check(Path(str(a.input/'science/stable-gd')+'.fit_coefficients.csv').read_bytes()==(root/'pm_optimizer_validation/science/value-gd.fit_coefficients.csv').read_bytes(),'stable GD 24-step result remains bitwise identical')
check(rows['stable-lbfgs']['status']==0 and rows['stable-lbfgs']['raw_gradient']<plan['gradient_tolerance'],'stable L-BFGS reaches unchanged gradient tolerance','convergence')
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),checks=checks,rows=rows,baseline=prior['rows'],frozen=dict(raw_accepts=int(sum(frozen['raw_accept'])),stable_accepts=int(sum(frozen['stable_accept'])),max_difference_error=float(np.max(abs(frozen['stable_delta']-old['stable_delta'])))),limits=['Same single fixed-parameter mock; broader refinement remains deferred.','Wall times are indicative under uncontrolled host activity.','Gaussian difference support and scalar priors are tested; custom likelihoods require their own difference method.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'comparison.csv').open('w') as f:
 writer=csv.writer(f,lineterminator='\n');writer.writerow(['method','iterations','status','raw_gradient','forwards','pullbacks','references','seconds'])
 for name in ['lbfgs','value-gd']:x=prior['rows'][name];writer.writerow([name,x['iterations'],x['status'],x['raw_gradient'],x['forwards'],x['pullbacks'],0,x['seconds']])
 for name,x in rows.items():writer.writerow([name,x['iterations'],x['status'],x['raw_gradient'],x['forwards'],x['pullbacks'],x['reference_evaluations'],x['seconds']])
fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
for name,label in [('stable-lbfgs','L-BFGS stable'),('stable-gd','GD stable')]:
 t=np.atleast_1d(np.genfromtxt(str(a.input/'science'/name)+'.fit_trace.csv',names=True,delimiter=','));axes[0].semilogy(t['iteration'],t['gradient_norm'],'-',label=label)
for name,label in [('lbfgs','L-BFGS ordinary'),('value-gd','GD ordinary')]:
 t=np.atleast_1d(np.genfromtxt(root/f'pm_optimizer_validation/science/{name}.fit_trace.csv',names=True,delimiter=','));axes[0].semilogy(t['iteration'],t['gradient_norm'],'--',label=label)
axes[0].axhline(plan['gradient_tolerance'],color='gray',ls=':');axes[0].set(xlabel='Accepted iterations',ylabel='Raw max gradient',title='Same gradient tolerance');axes[0].legend(fontsize=7)
labels=['L-BFGS ordinary','L-BFGS stable','GD ordinary','GD stable'];data=[prior['rows']['lbfgs'],rows['stable-lbfgs'],prior['rows']['value-gd'],rows['stable-gd']];x=np.arange(4)
axes[1].bar(x-.18,[v['forwards'] for v in data],.36,label='Predictions');axes[1].bar(x+.18,[v['pullbacks'] for v in data],.36,label='Pullbacks');axes[1].set(title='Actual model calls',xticks=x,xticklabels=labels);axes[1].tick_params(axis='x',rotation=22);axes[1].legend(fontsize=7)
axes[2].bar(x,[v['seconds'] for v in data]);axes[2].set(title='Indicative MAP wall time [s]',xticks=x,xticklabels=labels);axes[2].tick_params(axis='x',rotation=22)
fig.suptitle('Unchanged N32/M16 reconstruction, initialization, nuisance parameters and data')
fig.savefig(a.output/'comparison.png',dpi=170);fig.savefig(a.output/'comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.plot(frozen['alpha'],frozen['raw_delta'],'o-',ms=3,label='Ordinary subtraction');ax.plot(frozen['alpha'],frozen['stable_delta'],'o-',ms=3,label='New stable change API');ax.plot(frozen['alpha'],frozen['armijo_delta'],'k:',label='Armijo change bound');ax.axhline(0,color='gray',lw=.7);ax.set_xscale('log');ax.invert_xaxis();ax.set_yscale('symlog',linthresh=1e-13);ax.set(xlabel='Frozen search step alpha',ylabel='Target change',title='The diagnosed false rejection is removed');ax.legend(fontsize=8);fig.savefig(a.output/'frozen.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],failed=[c['check'] for c in checks if not c['passed']],rows=rows),indent=2))
if not summary['passed']:raise SystemExit(1)
