"""Field-only finite-reference checks; no posterior-gradient accuracy claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent;plan=json.loads((a.input/'plan.json').read_text());checks=[]
def check(ok,label,kind='consistency'): checks.append(dict(check=label,passed=bool(ok),kind=kind))
def load(prefix,suffix,skip=0):return np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=skip)
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16;x,y,zz=np.meshgrid(f,f,z,indexing='ij');q=x*x+y*y+zz*zz
w=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),q.shape);mask=(q>0)&(q<=16);dof=float(w[mask].sum())
check(dof==plan['observed'],'independent observed degrees')
def norm(v):return float(np.sqrt(np.sum(w[mask]*abs(v[mask])**2)))
for name,expected in plan['inputs'].items():check(hashlib.sha256((here/name).read_bytes()).hexdigest()==expected,'frozen input '+name)
reference=np.loadtxt(here/'reference.initial.csv',delimiter=',');data=(here/'reference.data.csv').read_bytes();cases={}
for config in plan['cases']:
    n,m,s=config['n'],config['m'],config['steps'];key=f'{n}-{m}-{s}';prefix=a.input/'science'/('case-'+key)
    v=load(prefix,'modes').reshape(16,16,9,2);density=v[...,0]+1j*v[...,1];initial=load(prefix,'initial')
    with Path(str(prefix)+'.forward_meta.csv').open() as file:mt={k:float(v) for k,v in next(csv.DictReader(file)).items()}
    check(initial.shape==(122,) and np.max(np.abs(initial-reference))<2e-12,'matched physical initial '+key)
    check(np.all(np.isfinite(density)) and np.max(np.abs(density[~mask]))==0,'finite supported density '+key)
    check(all(np.isfinite(v) for v in mt.values()) and mt['n']==n and mt['m']==m and mt['steps']==s and mt['active']==122 and
          mt['seed']==91 and mt['power']==.002 and mt['final']==1. and mt['diagnostic_nodes']==s+1,'finite complete metadata '+key)
    check(mt['max_mass_error']<1e-10 and mt['max_mean_momentum']<1e-10,'mass/momentum conservation '+key)
    check(Path(str(prefix)+'.data.csv').read_bytes()==data,'same frozen data '+key)
    check(not Path(str(prefix)+'.gradient.csv').exists(),'no gradient output in forward-only mode '+key)
    cases[key]=dict(prediction=density,metadata=mt)
families={'particles-M32':['64-32-32','96-32-32','128-32-32'],
          'particles-M64':['64-64-32','96-64-32','128-64-32'],
          'force-N128':['128-32-32','128-64-32'],
          'time-N128-M64':['128-64-32','128-64-64']}
convergence={}
for name,keys in families.items():
    fine=cases[keys[-1]];rows=[]
    for key in keys:
        c=cases[key];delta=c['prediction']-fine['prediction'];n,m,s=map(int,key.split('-'))
        rows.append(dict(case=key,resolution=n if name.startswith('particles') else (m if name.startswith('force') else s),
                         field_error=norm(delta)/max(norm(fine['prediction']),1e-30),noise_rms_units=norm(delta)/np.sqrt(plan['noise_power']*dof)))
    last=rows[-2]
    check(last['field_error']<plan['accuracy']['field'],name+' last field <5%','accuracy')
    check(last['noise_rms_units']<plan['accuracy']['noise'],name+' last discrepancy <0.1 noise SD','accuracy')
    if len(rows)>2:check(last['field_error']<rows[0]['field_error'],name+' improving particle correction','accuracy')
    convergence[name]=dict(reference=keys[-1],rows=rows)
shells=[s for s in range(1,17) if np.any(q==s)];power={}
for n in [64,96,128]:
    coarse=cases[f'{n}-32-32']['prediction'];fine=cases[f'{n}-64-32']['prediction']
    power[str(n)]=[float(np.sum(w[q==s]*abs(coarse[q==s])**2)/np.sum(w[q==s]*abs(fine[q==s])**2)-1) for s in shells]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for name,style in [('particles-M32','o-'),('particles-M64','s-')]:
    rows=convergence[name]['rows'][:-1];axes[0].semilogy([r['resolution'] for r in rows],[r['field_error'] for r in rows],style,label=name)
axes[0].axhline(.05,color='gray',ls=':');axes[0].set(title='Particle corrections to N128',xlabel='Particle/IC N',ylabel='Relative observed-band density difference');axes[0].set_xticks([64,96]);axes[0].xaxis.set_minor_locator(NullLocator());axes[0].legend(fontsize=8)
for n in [64,96,128]:axes[1].plot(2*np.pi*np.sqrt(shells),100*np.array(power[str(n)]),'o-',label=f'N={n}')
axes[1].axhline(0,color='gray',lw=.6);axes[1].set(title='Force-grid shell-power response',xlabel='k [1/L]',ylabel='M32 power / M64 power - 1 [%]');axes[1].legend(fontsize=8)
for m,style in [(32,'o-'),(64,'s-')]:
    axes[2].plot([64,96,128],[cases[f'{n}-{m}-32']['metadata']['density_rms'] for n in [64,96,128]],style,label=f'M={m}')
axes[2].set(title='Final full-mesh density RMS',xlabel='Particle/IC N',ylabel='RMS (includes unobserved modes)');axes[2].set_xticks([64,96,128]);axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Strong91, 122 initial directions, 32 steps; unchanged CIC/spectral force')
fig.savefig(a.output/'reference.png',dpi=170);fig.savefig(a.output/'reference.pdf');plt.close(fig)
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),
             checks=checks,convergence=convergence,force_power=dict(squared_wavenumber=shells,fractional_difference=power),
             metadata={key:c['metadata'] for key,c in cases.items()},
             limits=['One strong initial-field realization, fixed IC/observed support and finite references.',
                     'Field-only test; no new adjoint/gradient, optimization, posterior or coverage validation.',
                     'Finite particle/force/time corrections do not identify an exact continuum solution or optimal numerical law.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['family','case','field_error','noise_rms_units'])
    for name,c in convergence.items():
        for row in c['rows']:writer.writerow([name,row['case'],row['field_error'],row['noise_rms_units']])
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],failed=[c['check'] for c in checks if not c['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
