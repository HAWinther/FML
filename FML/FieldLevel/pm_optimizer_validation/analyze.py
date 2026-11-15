"""Compare optimizer work and recovery using the frozen reconstruction diagnostics."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);plan=json.loads((a.input/'plan.json').read_text());checks=[]
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def metadata(prefix,suffix):
    with Path(str(prefix)+'.'+suffix+'.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
rows={};summaries={}
base=Path(__file__).resolve().parent.parent/'pm_reconstruction_validation/analyze.py'
for spec in plan['runs']:
    name=spec['name'];budget=spec['iterations'];source=a.input/'science'/name
    mapped=a.output/(name+'-input');(mapped/'science').mkdir(parents=True,exist_ok=True)
    subplan=dict(plan,budgets=[budget]);(mapped/'plan.json').write_text(json.dumps(subplan,indent=2)+'\n')
    for path in (a.input/'science').glob('reference.*.csv'):
        (mapped/'science'/path.name).write_bytes(path.read_bytes())
    for path in (a.input/'science').glob(name+'.*.csv'):
        (mapped/'science'/('fit-'+str(budget)+path.name[len(name):])).write_bytes(path.read_bytes())
    report=a.output/name
    with (a.output/(name+'-analysis.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(base),'--input',str(mapped),'--output',str(report)],stdout=log,stderr=subprocess.STDOUT)
    check(result.returncode in [0,1] and (report/'summary.json').exists(),name+' analyzer completed')
    measured=json.loads((report/'summary.json').read_text());summaries[name]=measured
    check(measured['consistency_passed'],name+' unchanged-data reconstruction consistency')
    check(measured['recovery_passed'],name+' reconstruction recovery','recovery')
    row=measured['rows'][-1];row.update(metadata(source,'fit_optimizer'));rows[name]=row
    check(int(row['optimizer'])==spec['optimizer'] and int(row['history'])==spec['history'],name+' optimizer identity')
    check(row['pullbacks']==row['gradient_evaluations'],name+' reported gradients match actual pullbacks')
    check(row['forwards']==row['value_evaluations']+row['gradient_evaluations'],name+' reported work matches actual predictions')
    check(row['peak_history_pairs']<=spec['history'],name+' bounded history')
    if spec['optimizer']:
        trace=np.atleast_1d(np.genfromtxt(str(source)+'.fit_trace.csv',delimiter=',',names=True))
        check(np.all(np.isfinite(trace['potential'])) and np.all(np.diff(trace['potential'])<=1e-10),name+' finite monotone trace')
        check(trace[-1]['gradient_norm']==row['raw_gradient'] and int(trace[-1]['iteration'])==row['iterations'],name+' trace matches endpoint')
legacy=a.input/'science/legacy';old=a.input/'science/fit-24'
for suffix in ['fit_coefficients','fit']:
    check(Path(str(legacy)+'.'+suffix+'.csv').read_bytes()==Path(str(old)+'.'+suffix+'.csv').read_bytes(),'legacy reproduces frozen '+suffix+' bitwise')
check(Path(str(a.input/'science/value-gd')+'.fit_coefficients.csv').read_bytes()==Path(str(legacy)+'.fit_coefficients.csv').read_bytes(),'value-only GD reproduces legacy coefficients bitwise')
check(rows['value-gd']['pullbacks']==rows['value-gd']['iterations']+1,'value-only GD requests gradients only at accepted points')
check(rows['value-gd']['pullbacks']<rows['legacy']['pullbacks'],'value-only line search reduces pullbacks','performance')
check(rows['lbfgs']['status']==0 and rows['lbfgs']['raw_gradient']<plan['gradient_tolerance'],'L-BFGS reaches unchanged tolerance','convergence')
# Wall time is reported, not gated: concurrent host activity affects it.
check(rows['lbfgs']['pullbacks']<rows['legacy']['pullbacks'],'L-BFGS uses fewer pullbacks than legacy budget','performance')
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),checks=checks,rows=rows,
    limits=['One frozen compensated mock; broader refinement deferred.','Timing is indicative; same launcher/ranks/compiler, uncontrolled host activity.',
            'sum_rank_peak_rss_bytes sums process high-water marks, not simultaneous cluster RSS. History bytes count retained vector payload only.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['algorithm',*rows['legacy']],lineterminator='\n');writer.writeheader()
    for name,row in rows.items():writer.writerow(dict(algorithm=name,**row))
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for name,label in [('lbfgs','L-BFGS'),('value-gd','Value-only GD')]:
    t=np.atleast_1d(np.genfromtxt(str(a.input/'science'/name)+'.fit_trace.csv',delimiter=',',names=True))
    axes[0].semilogy(t['iteration'],t['gradient_norm'],'o-',ms=3,label=label)
    axes[1].plot(t['iteration'],t['potential'],'o-',ms=3,label=label)
old_x=[];old_g=[];old_u=[]
for n in [1,4,12,24]:
    m=metadata(a.input/f'science/fit-{n}','fit');old_x.append(m['iterations']);old_g.append(m['gradient_norm']);old_u.append(m['potential'])
axes[0].semilogy(old_x,old_g,'k--x',label='Legacy endpoints');axes[1].plot(old_x,old_u,'k--x',label='Legacy endpoints')
axes[0].axhline(plan['gradient_tolerance'],color='gray',ls=':');axes[0].set(xlabel='Accepted iterations',ylabel='Raw cell max gradient',title='Unchanged convergence criterion')
axes[1].set(xlabel='Accepted iterations',ylabel='Posterior U',title='Objective')
names=['legacy','value-gd','lbfgs'];x=np.arange(3)
axes[2].bar(x-.18,[rows[k]['forwards'] for k in names],.36,label='Predictions');axes[2].bar(x+.18,[rows[k]['pullbacks'] for k in names],.36,label='Pullbacks');axes[2].set(xticks=x,xticklabels=['Legacy GD','Value-only GD','L-BFGS'],ylabel='Actual calls',title='Measured optimizer work')
for ax in axes:ax.legend(fontsize=8)
fig.suptitle('Same frozen data, N32/M16/16-step fit, fixed nuisance parameters')
fig.savefig(a.output/'optimization.png',dpi=170);fig.savefig(a.output/'optimization.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for ax,key,title,scale in [(axes[0],'seconds','MAP wall time [s]',1),(axes[1],'total_history_bytes','Retained history payload [MiB]',2**20),(axes[2],'sum_rank_peak_rss_bytes','Sum of rank peak RSS [MiB]',2**20)]:
    ax.bar(x,[rows[k][key]/scale for k in names]);ax.set(xticks=x,xticklabels=['Legacy','Value GD','L-BFGS'],title=title)
fig.suptitle('Four-rank run; RSS includes the native process lifetime, not just MAP')
fig.savefig(a.output/'resources.png',dpi=170);fig.savefig(a.output/'resources.pdf');plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],failed=[c['check'] for c in checks if not c['passed']],rows=rows),indent=2))
if not summary['passed']:raise SystemExit(1)
