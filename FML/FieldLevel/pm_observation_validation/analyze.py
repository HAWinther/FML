"""Actual PM posterior refinement in a common observation space."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent
plan=json.loads((a.input/'plan.json').read_text());checks=[]
def check(ok,label,kind='consistency'):checks.append(dict(check=label,passed=bool(ok),kind=kind))
def load(prefix,suffix,skip=0):return np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=skip)
def modes(prefix):
    v=load(prefix,'modes').reshape(16,16,9,2);return v[...,0]+1j*v[...,1]
source=(here.parent/'pm_force_reference_validation/reference.data.csv').read_bytes();data=(a.input/'reference.data.csv').read_bytes()
check(hashlib.sha256(source).hexdigest()==plan['source_data_sha256'],'source frozen data hash')
check(hashlib.sha256(data).hexdigest()==plan['data_sha256'],'tagged frozen data hash')
check(source.split(b'\n',1)[1]==data.split(b'\n',1)[1] and data.split(b'\n',1)[0]==source.split(b'\n',1)[0]+b',observation_cic,32','tag changes declared window only; data values unchanged')
initial=np.loadtxt(here.parent/'pm_force_reference_validation/reference.initial.csv',delimiter=',')
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16;x,y,zz=np.meshgrid(f,f,z,indexing='ij');q=x*x+y*y+zz*zz
weights=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),q.shape);mask=(q>0)&(q<=16);dof=float(weights[mask].sum())
check(dof==256,'independent observed degrees')
def norm(v):return float(np.sqrt(np.sum(weights[mask]*abs(v[mask])**2)))
cases={};metadata={}
for w in [0,1]:
    for forward,configs in [(True,plan['forward']+(plan['extra_forward'] if w else [])),(False,plan['gradients'])]:
        for c in configs:
            n,m,s=c['n'],c['m'],c['steps'];key=f"{'forward' if forward else 'gradient'}-w{w}-{n}-{m}-{s}";prefix=a.input/'science'/key
            v=modes(prefix);physical=load(prefix,'initial')
            with Path(str(prefix)+('.forward_meta.csv' if forward else '.meta.csv')).open() as file:mt={k:float(v) for k,v in next(csv.DictReader(file)).items()}
            check(physical.shape==(122,) and np.max(abs(physical-initial))<2e-12,key+' matched physical IC')
            check(np.all(np.isfinite(v)) and np.max(abs(v[~mask]))==0 and all(np.isfinite(x) for x in mt.values()),key+' finite supported output')
            check(mt['n']==n and mt['m']==m and mt['steps']==s and mt['seed']==91 and mt['power']==.002 and mt['final']==1. and mt['active']==122,key+' fixture metadata')
            check(int(load(prefix,'window',1))==w and int(load(prefix,'observation',1))==32,key+' fixed force/observation conventions')
            check(Path(str(prefix)+'.data.csv').read_bytes()==data,key+' same tagged data')
            if forward:
                check(mt['diagnostic_nodes']==s+1 and mt['max_mass_error']<1e-10 and mt['max_mean_momentum']<1e-10,key+' mass/momentum and complete nodes')
            else:
                g=load(prefix,'gradient');check(g.shape==(124,) and np.all(np.isfinite(g)),key+' finite posterior gradient')
            cases[key]=dict(prediction=v,gradient=None if forward else g);metadata[key]=mt
convergence={}
for w in [0,1]:
    coarse=cases[f'forward-w{w}-128-32-32']['prediction'];fine=cases[f'forward-w{w}-128-64-32']['prediction'];delta=coarse-fine
    ga,gb=[cases[f'gradient-w{w}-64-{m}-32']['gradient'] for m in [32,64]]
    row=dict(field_error=norm(delta)/norm(fine),noise_rms_units=norm(delta)/np.sqrt(1e-4*dof),gradient_error=float(np.linalg.norm(ga-gb)/np.linalg.norm(gb)),
             field_gradient_error=float(np.linalg.norm(ga[:122]-gb[:122])/np.linalg.norm(gb[:122])),scalar_gradients=[ga[-2:].tolist(),gb[-2:].tolist()])
    for metric,bound in [('field_error',.05),('noise_rms_units',.1),('gradient_error',.15)]:check(row[metric]<bound,f'w{w} common-observation {metric} < {bound}','accuracy')
    convergence[str(w)]=row
extra={}
for family,keys in [('particle',['forward-w1-96-64-32','forward-w1-128-64-32']),('time',['forward-w1-128-64-32','forward-w1-128-64-64'])]:
    coarse,fine=[cases[k]['prediction'] for k in keys];delta=coarse-fine
    extra[family]=dict(field_error=norm(delta)/norm(fine),noise_rms_units=norm(delta)/np.sqrt(1e-4*dof))
    check(extra[family]['field_error']<.05 and extra[family]['noise_rms_units']<.1,f'compensated observed {family} refinement','accuracy')
fd={};prefix=a.input/'science/directional-w1';r=np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv',delimiter=',',names=True))
for probe in [0,1]:
    v=r[r['probe']==probe];err=abs(v['numeric']-v['analytic'])/np.maximum(1.,abs(v['analytic']))
    check(len(v)==8 and np.min(err[-4:])<2e-5 and np.sum(err[-4:]<1e-4)>=2,f'observed compensated FD {probe}')
    fd[str(probe)]=dict(h=v['h'].tolist(),error=err.tolist())
check(int(load(prefix,'observation',1))==32 and int(load(prefix,'window',1))==1,'FD uses nontrivial M64 to M32 response and compensated force')
shells=[s for s in range(1,17) if np.any(q==s)];power={}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for w,label in [(0,'Uncompensated force'),(1,'CIC-compensated force')]:
    ca=cases[f'forward-w{w}-128-32-32']['prediction'];cb=cases[f'forward-w{w}-128-64-32']['prediction']
    values=[float(np.sum(weights[q==s]*abs(ca[q==s])**2)/np.sum(weights[q==s]*abs(cb[q==s])**2)-1) for s in shells]
    power[str(w)]=values;axes[0].plot(2*np.pi*np.sqrt(shells),100*np.array(values),'o-',label=label)
    axes[1].bar(w,100*convergence[str(w)]['field_error'],label=label)
    axes[2].bar(w,100*convergence[str(w)]['gradient_error'],label=label)
axes[0].axhline(0,color='gray',lw=.7);axes[0].set(title='Observed power: M32 / M64 - 1',xlabel='k [1/L]',ylabel='Power difference [%]');axes[0].legend(fontsize=7)
for ax,metric,title,threshold in [(axes[1],'field_error','Observed density difference',5),(axes[2],'gradient_error','Full projected posterior gradient',15)]:
    ax.set_xticks([0,1],['Uncompensated','Compensated']);ax.set(title=title,ylabel='Relative difference [%]');ax.axhline(threshold,color='gray',ls=':');ax.grid(alpha=.2,axis='y')
fig.suptitle('Actual PM likelihood: fixed M32 CIC observation window; M32/M64 force comparison')
fig.savefig(a.output/'observation.png',dpi=170);fig.savefig(a.output/'observation.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
for probe,v in fd.items():ax.loglog(v['h'],np.maximum(v['error'],1e-16),'o-',label='field' if probe=='0' else 'field + scalars')
ax.axhline(2e-5,color='gray',ls=':');ax.set(title='Nontrivial observation-response adjoint',xlabel='Central-difference h',ylabel='Scaled directional error');ax.legend();fig.savefig(a.output/'gradients.png',dpi=170);plt.close(fig)
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),
             compensated_accuracy_passed=all(c['passed'] for c in checks if c['kind']=='accuracy' and not c['check'].startswith('w0')),
             checks=checks,convergence=convergence,extra=extra,finite_differences=fd,power=dict(squared_wavenumber=shells,fractional_difference=power),metadata=metadata,
             limits=['One strong realization, fixed nuisance values and finite references; no coverage or global convergence proof.',
                     'Forward density uses N128, projected-gradient comparison uses finite N64. Further particle/gradient calibration remains useful.',
                     'No new MAP/HMC scientific chains, PM survey/tracer/RSD integration or cosmological derivatives.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['force_window','field_error','noise_rms_units','gradient_error','field_gradient_error'])
    for w,row in convergence.items():writer.writerow([w,row['field_error'],row['noise_rms_units'],row['gradient_error'],row['field_gradient_error']])
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],compensated_accuracy_passed=summary['compensated_accuracy_passed'],failed=[c['check'] for c in checks if not c['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
