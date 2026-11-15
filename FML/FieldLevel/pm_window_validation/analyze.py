"""Force-window comparison with explicit raw and common-observation-window results."""
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
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def load(prefix,suffix,skip=0):return np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=skip)
def modes(prefix):
    v=load(prefix,'modes').reshape(16,16,9,2);return v[...,0]+1j*v[...,1]
def meta(prefix,forward):
    with Path(str(prefix)+('.forward_meta.csv' if forward else '.meta.csv')).open() as file:
        return {k:float(v) for k,v in next(csv.DictReader(file)).items()}
data=here.parent/'pm_force_reference_validation/reference.data.csv';initial=np.loadtxt(here.parent/'pm_force_reference_validation/reference.initial.csv',delimiter=',')
check(hashlib.sha256(data.read_bytes()).hexdigest()==plan['data_sha256'],'frozen raw-data hash')
check(hashlib.sha256((here/'linear.initial.csv').read_bytes()).hexdigest()==plan['linear_sha256'],'linear-control initial hash')
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16;x,y,zz=np.meshgrid(f,f,z,indexing='ij');q=x*x+y*y+zz*zz
weights=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),q.shape);mask=(q>0)&(q<=16);dof=float(weights[mask].sum())
check(dof==256,'independent observed degrees')
def norm(v):return float(np.sqrt(np.sum(weights[mask]*abs(v[mask])**2)))
def window(m):return np.sinc(x/m)**2*np.sinc(y/m)**2*np.sinc(zz/m)**2
common=window(32)
cases={};metadata={};fd={}
for w in [0,1]:
    for forward,configs in [(True,plan['forward']),(False,plan['gradients'])]:
        for c in configs:
            n,m,s=c['n'],c['m'],c['steps'];key=f"{'forward' if forward else 'gradient'}-w{w}-{n}-{m}-{s}";prefix=a.input/'science'/key
            v=modes(prefix);mt=meta(prefix,forward);physical=load(prefix,'initial')
            check(physical.shape==(122,) and np.max(abs(physical-initial))<2e-12,key+' matched physical ICs')
            check(np.all(np.isfinite(v)) and np.max(abs(v[~mask]))==0 and all(np.isfinite(v) for v in mt.values()),key+' finite supported output')
            check(Path(str(prefix)+'.data.csv').read_bytes()==data.read_bytes(),key+' same frozen raw data')
            check(int(load(prefix,'window',1))==w,key+' explicit window metadata')
            check(mt['n']==n and mt['m']==m and mt['steps']==s and mt['active']==122,key+' configuration metadata')
            if forward:
                check(mt['diagnostic_nodes']==s+1 and mt['max_mass_error']<1e-10 and mt['max_mean_momentum']<1e-10,key+' diagnostic/conservation checks')
            else:
                g=load(prefix,'gradient');check(g.shape==(124,) and np.all(np.isfinite(g)),key+' finite projected posterior gradient')
            cases[key]=dict(raw=v,matched=v*common/window(m),gradient=None if forward else g);metadata[key]=mt
convergence={}
for w in [0,1]:
    convergence[str(w)]={}
    for family,keys in [('force',[f'forward-w{w}-128-32-32',f'forward-w{w}-128-64-32']),
                        ('particle',[f'forward-w{w}-96-64-32',f'forward-w{w}-128-64-32']),
                        ('time',[f'forward-w{w}-128-64-32',f'forward-w{w}-128-64-64'])]:
        coarse,fine=[cases[k] for k in keys];row={}
        for representation in ['raw','matched']:
            delta=coarse[representation]-fine[representation]
            row[representation]=dict(field_error=norm(delta)/max(norm(fine[representation]),1e-30),noise_rms_units=norm(delta)/np.sqrt(1e-4*dof))
        check(row['raw']['field_error']<.05,f'w{w} {family} raw field <5%','accuracy')
        check(row['raw']['noise_rms_units']<.1,f'w{w} {family} raw discrepancy <0.1 noise SD','accuracy')
        row['cases']=keys
        if family=='force':
            ga,gb=[cases[f'gradient-w{w}-64-{m}-32']['gradient'] for m in [32,64]]
            row['raw_gradient_error']=float(np.linalg.norm(ga-gb)/np.linalg.norm(gb))
            row['field_gradient_error']=float(np.linalg.norm(ga[:122]-gb[:122])/np.linalg.norm(gb[:122]))
            row['scalar_gradients']=[ga[-2:].tolist(),gb[-2:].tolist()]
            check(row['raw_gradient_error']<.15,f'w{w} raw projected gradient <15%','accuracy')
        convergence[str(w)][family]=row
prefix=a.input/'science/directional-w1';rows=np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv',delimiter=',',names=True))
for probe in [0,1]:
    r=rows[rows['probe']==probe];error=abs(r['numeric']-r['analytic'])/np.maximum(1,abs(r['analytic']))
    check(len(r)==8 and np.min(error[-4:])<2e-5 and np.sum(error[-4:]<1e-4)>=2,f'compensated refined FD {probe}')
    fd[str(probe)]=dict(h=r['h'].tolist(),error=error.tolist())
check(int(load(prefix,'window',1))==1,'FD fixture uses compensation')
linear={}
for w in [0,1]:
    source=json.loads((a.input/f'force-analysis-w{w}/force_summary.json').read_text())
    # Keep the underresolved coarse compensated step visible. Acceptance uses
    # both refined kicks, with the same 2e-5 tolerance and explicit step agreement.
    cap=plan['probe_acceptance_max_delta_a'][str(w)]
    accepted=[r for r in source['comparisons'] if r['delta_a']<=cap]
    native=[r for r in source['native_rows'] if r['delta_a']<=cap]
    fine_error=max(r['max_error'] for r in accepted)
    check(fine_error<2e-5,f'w{w} independent native/NumPy force checks at accepted kicks')
    for n,m in sorted(set((r['n'],r['m']) for r in native)):
        rows=[r for r in native if r['n']==n and r['m']==m]
        check(len(rows)==2 and np.ptp([r['longitudinal_response'] for r in rows])<2e-5,f'w{w} refined shrinking kick {int(n)}/{int(m)}')
    linear[str(w)]=dict(max_native_numpy_difference=source['max_native_numpy_difference'],max_accepted_difference=fine_error,
                       acceptance_max_delta_a=cap,all_step_consistency_passed=source['code_consistency_passed'],
                       all_step_checks=source['checks'],comparisons=source['comparisons'],rows=source['native_rows'])
for m in [16,32]:
    values=[next(r['longitudinal_response'] for r in linear[str(w)]['rows'] if r['n']==64 and r['m']==m and r['delta_a']==1e-9) for w in [0,1]]
    check(abs(values[1]-1)<abs(values[0]-1) and abs(values[1]-1)<.01,f'dense N64/M{m} linear response improves and is within 1%','linear_accuracy')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
shells=[s for s in range(1,17) if np.any(q==s)];power={}
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for w,label in [(0,'Uncompensated'),(1,'CIC-compensated')]:
    for representation,style in [('raw','o-'),('matched','s--')]:
        aa=cases[f'forward-w{w}-128-32-32'][representation];bb=cases[f'forward-w{w}-128-64-32'][representation]
        values=[float(np.sum(weights[q==s]*abs(aa[q==s])**2)/np.sum(weights[q==s]*abs(bb[q==s])**2)-1) for s in shells]
        power[f'{w}-{representation}']=values;axes[0].plot(2*np.pi*np.sqrt(shells),100*np.array(values),style,label=label+' / '+representation)
    response=[next(r['longitudinal_response'] for r in linear[str(w)]['rows'] if r['n']==64 and r['m']==m and r['delta_a']==1e-9) for m in [8,16,32]]
    axes[1].plot([8,16,32],response,'o-',label=label)
    axes[2].bar(np.array([0,1])+w*.32,[100*convergence[str(w)]['force'][k]['field_error'] for k in ['raw','matched']],width=.32,label=label)
axes[0].axhline(0,color='gray',lw=.6);axes[0].set(title='M32 / M64 power, N128, 32 steps',xlabel='k [1/L]',ylabel='Power difference [%]');axes[0].legend(fontsize=7)
axes[1].axhline(1,color='gray',ls=':');axes[1].set(title='Independent tiny-kick control: N64',xlabel='Force mesh M',ylabel='Longitudinal force response');axes[1].set_xticks([8,16,32]);axes[1].legend(fontsize=8)
axes[2].set_xticks([.16,1.16],['Raw density','Common-window density']);axes[2].set(title='Finite force-grid field difference',ylabel='Relative density difference [%]');axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2,axis='y')
fig.suptitle('Explicit force compensation; final CIC density is unchanged; matched curves are postprocessing')
fig.savefig(a.output/'window_comparison.png',dpi=170);fig.savefig(a.output/'window_comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
for probe,r in fd.items():ax.loglog(r['h'],np.maximum(r['error'],1e-16),'o-',label='field' if probe=='0' else 'field + scalars')
ax.axhline(2e-5,color='gray',ls=':');ax.set(title='Compensated full-posterior directional checks',xlabel='Central-difference h',ylabel='Scaled error');ax.legend();fig.savefig(a.output/'gradients.png',dpi=170);plt.close(fig)
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),checks=checks,
             convergence=convergence,finite_differences=fd,linear=linear,power=dict(squared_wavenumber=shells,fractional_difference=power),metadata=metadata,
             limits=['One strong realization and a separate small linear-force control; finite references, no coverage/scaling claim.',
                     'Common-window density is postprocessing, not the raw-density likelihood or a new posterior-gradient validation.',
                     'CIC compensation does not remove aliasing or replace particle/time/cutoff checks; sparse-load behavior may be poor.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['window','family','representation','field_error','noise_rms_units'])
    for w,families in convergence.items():
        for family,r in families.items():
            for k in ['raw','matched']:writer.writerow([w,family,k,r[k]['field_error'],r[k]['noise_rms_units']])
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],failed=[c['check'] for c in checks if not c['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
