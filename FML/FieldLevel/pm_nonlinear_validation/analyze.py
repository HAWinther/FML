"""Many-mode finite-reference accuracy and refined-data MAP diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads((a.input/'plan.json').read_text()); gates=[]
def check(ok,text,kind='consistency'):
    gates.append(dict(check=text,passed=bool(ok),kind=kind))
def load(prefix,suffix,skip=0):
    return np.atleast_1d(np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=skip))
def meta(prefix,suffix='meta'):
    with Path(str(prefix)+'.'+suffix+'.csv').open() as f:
        return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
def modes(prefix,suffix='modes'):
    x=load(prefix,suffix).reshape(16,16,9,2);return x[...,0]+1j*x[...,1]
f=np.fft.fftfreq(16)*16; z=np.fft.rfftfreq(16)*16
x,y,zz=np.meshgrid(f,f,z,indexing='ij');q=x*x+y*y+zz*zz
weight=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),q.shape)
mask=(q>0)&(q<=16); dof=float(weight[mask].sum())
check(dof==plan['observed'],'independent observed degrees')
def norm(x):return float(np.sqrt(np.sum(weight[mask]*np.abs(x[mask])**2)))
def case_prefix(name,n,m,s):
    return a.input/'science'/(name+'-reference' if (n,m,s)==(64,32,32) else f'{name}-{n}-{m}-{s}')
cases={};convergence={};fd={};fits={}
for fixture in plan['fixtures']:
    name=fixture['name']; reference=a.input/'science'/(name+'-reference')
    initial_ref=load(reference,'initial');data_bytes=Path(str(reference)+'.data.csv').read_bytes()
    check(initial_ref.shape==(122,),'122 initial coordinates '+name)
    for c in plan['cases']:
        n,m,s=c['n'],c['m'],c['steps'];key=f'{name}-{n}-{m}-{s}';prefix=case_prefix(name,n,m,s)
        initial=load(prefix,'initial');g=load(prefix,'gradient');density=modes(prefix);mt=meta(prefix)
        check(initial.shape==(122,) and np.max(np.abs(initial-initial_ref))<2e-12,'matched initial '+key)
        check(g.shape==(124,) and np.all(np.isfinite(g)) and np.all(np.isfinite(density)) and np.max(np.abs(density[~mask]))==0,'finite supported outputs '+key)
        check(Path(str(prefix)+'.data.csv').read_bytes()==data_bytes,'identical refined data '+key)
        check(mt['active']==122 and all(mt[k]==fixture[k] for k in ['seed','power','final']),'fixture metadata '+key)
        cases[key]=dict(prediction=density,gradient=g,metadata=mt)
        if Path(str(prefix)+'.fd.csv').exists():
            r=np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv',delimiter=',',names=True));fd[key]={}
            for probe in [0,1]:
                v=r[r['probe']==probe];error=np.abs(v['numeric']-v['analytic'])/np.maximum(1.,np.abs(v['analytic']))
                check(len(v)==8 and np.all(np.isfinite(error)) and np.min(error[-4:])<2e-5 and np.sum(error[-4:]<1e-4)>=2,key+f' refined FD {probe}')
                fd[key][str(probe)]=dict(h=v['h'].tolist(),error=error.tolist())
    convergence[name]={}
    for family,configs in [('particle',[(n,16,16) for n in [32,48,64]]),('force',[(64,m,32) for m in [32,64]]),('time',[(64,32,s) for s in [32,64]])]:
        reference_key=f'{name}-'+ '-'.join(map(str,configs[-1]));fine=cases[reference_key];rows=[]
        for n,m,s in configs:
            key=f'{name}-{n}-{m}-{s}';c=cases[key];delta=c['prediction']-fine['prediction']
            rows.append(dict(key=key,resolution={'particle':n,'force':m,'time':s}[family],field_error=norm(delta)/max(norm(fine['prediction']),1e-30),
                             gradient_error=float(np.linalg.norm(c['gradient']-fine['gradient'])/max(np.linalg.norm(fine['gradient']),1e-30)),
                             field_gradient_error=float(np.linalg.norm(c['gradient'][:122]-fine['gradient'][:122])/max(np.linalg.norm(fine['gradient'][:122]),1e-30)),
                             scalar_gradients=c['gradient'][-2:].tolist(),noise_rms_units=norm(delta)/np.sqrt(1e-4*dof)))
        last=rows[-2];limits=plan['accuracy']
        for metric,bound in [('field_error',limits['field']),('gradient_error',limits['gradient']),('noise_rms_units',limits['noise'])]:
            check(last[metric]<bound,f'{name} {family}: {metric} < {bound}', 'accuracy')
        if len(rows)>2:
            check(last['field_error']<rows[0]['field_error'] and last['gradient_error']<rows[0]['gradient_error'],name+' particle refinement improves','accuracy')
        convergence[name][family]=dict(reference=reference_key,rows=rows)
check(len(fd)==2,'two field/scalar derivative fixtures')
for spec in plan['fits']:
    key=f"{spec['fixture']}-{spec['n']}";prefix=a.input/'science'/('fit-'+key)
    c=load(prefix,'fit_coefficients');r=meta(prefix,'fit');truth,start,fit,gradient=c
    check(c.shape==(4,122) and np.all(np.isfinite(c)),'finite fitted coordinates '+key)
    power=next(f['power'] for f in plan['fixtures'] if f['name']==spec['fixture'])
    check(np.max(np.abs(load(prefix,'initial')-np.sqrt(power)*truth))<2e-12,'fit physical initial normalization '+key)
    ref=a.input/'science'/(spec['fixture']+'-reference')
    check(Path(str(prefix)+'.data.csv').read_bytes()==Path(str(ref)+'.data.csv').read_bytes(),'fit uses refined data '+key)
    density=modes(prefix,'fit_modes')
    raw=load(prefix,'data',1).reshape(16,16,9,2);data=raw[...,0]+1j*raw[...,1]
    error=float(np.linalg.norm(fit-truth)/np.linalg.norm(truth));start_error=float(np.linalg.norm(start-truth)/np.linalg.norm(truth));correlation=float(np.dot(truth,fit)/(np.linalg.norm(truth)*np.linalg.norm(fit)))
    check(np.isfinite(r['potential']) and r['potential']<r['initial_potential'],'MAP lowers objective '+key)
    check(r['status']==0,'MAP converges '+key,'optimization')
    check(error<.8 and correlation>.7,'MAP initial error <0.8 baseline and field correlation >0.7 '+key,'recovery')
    fits[key]=dict(r,relative_initial_error=error,start_error=start_error,initial_field_correlation=correlation,
                   residual_noise_units=norm(density-data)/np.sqrt(1e-4*dof),truth=truth.tolist(),fit=fit.tolist())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
for row,(name,families) in enumerate(convergence.items()):
    for col,(family,c) in enumerate(families.items()):
        r=c['rows'][:-1];xx=[v['resolution'] for v in r];ax=axes[row,col]
        ax.semilogy(xx,[v['field_error'] for v in r],'o-',label='density')
        ax.semilogy(xx,[v['gradient_error'] for v in r],'s-',label='projected gradient')
        ax.axhline(.05,color='C0',ls=':',alpha=.6);ax.axhline(.15,color='C1',ls=':',alpha=.6)
        ax.set(title=name+' / '+family,xlabel={'particle':'Particle/IC N (reference 64)','force':'Force M (reference 64)','time':'Steps (reference 64)'}[family],ylabel='Relative difference from finite reference')
        ax.set_xticks(xx,[str(v) for v in xx]);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('122 initial directions; refined common data; dotted lines: 5% field / 15% gradient criteria')
fig.savefig(a.output/'convergence.png',dpi=170);fig.savefig(a.output/'convergence.pdf');plt.close(fig)
# Force-grid power response at identical particles/time nodes, shell by shell.
force_power={}
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
shells=[s for s in range(1,17) if np.any(q==s)]
for ax,fixture in zip(axes,plan['fixtures']):
    name=fixture['name'];coarse=cases[name+'-64-32-32']['prediction'];fine=cases[name+'-64-64-32']['prediction']
    differences=[]
    for shell in shells:
        pick=q==shell
        differences.append(float(np.sum(weight[pick]*abs(coarse[pick])**2)/np.sum(weight[pick]*abs(fine[pick])**2)-1))
    force_power[name]=dict(squared_wavenumber=shells,fractional_power_difference=differences)
    ax.plot(2*np.pi*np.sqrt(shells),100*np.array(differences),'o-');ax.axhline(0,color='k',lw=.7)
    ax.set(title=name,xlabel='k [1/L]',ylabel='M32 power / M64 power - 1 [%]');ax.grid(alpha=.2)
fig.suptitle('Finite force-grid response: N=64 particles, 32 steps, identical physical ICs')
fig.savefig(a.output/'force_power.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,(key,probes) in zip(axes,fd.items()):
    for probe,v in probes.items():ax.loglog(v['h'],np.maximum(v['error'],1e-16),'o-',label='field' if probe=='0' else 'field + scalars')
    ax.axhline(2e-5,color='gray',ls=':');ax.set(title=key,xlabel='Central difference h',ylabel='Scaled directional error');ax.legend()
fig.savefig(a.output/'gradients.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,len(fits),figsize=(12,4),layout='constrained')
for ax,(key,r) in zip(np.atleast_1d(axes),fits.items()):
    ax.scatter(r['truth'],r['fit'],s=12,alpha=.6);lim=max(np.max(np.abs(r['truth'])),np.max(np.abs(r['fit'])))
    ax.plot([-lim,lim],[-lim,lim],'k:',lw=1);ax.set(title=f"{key}: status {int(r['status'])}\nrelative error {r['relative_initial_error']:.3f}",xlabel='True initial white coefficient',ylabel='Fitted coefficient')
fig.suptitle('Fixed-parameter MAP using N64/M32/32-step data; status 0 = converged')
fig.savefig(a.output/'reconstruction.png',dpi=170);fig.savefig(a.output/'reconstruction.pdf');plt.close(fig)
# Render all reconstructions on a common physical slice, independent of fit N.
waves=[(x,y,z) for x in range(-3,4) for y in range(-3,4) for z in range(-3,4)
       if 0<x*x+y*y+z*z<=9 and (x>0 or (x==0 and y>0) or (x==0 and y==0 and z>0))]
xx,yy=np.meshgrid(np.arange(32)/32,np.arange(32)/32,indexing='ij')
def slice_field(coefficients,power):
    result=np.zeros((32,32))
    for j,(x,y,z) in enumerate(waves):
        phase=2*np.pi*(x*xx+y*yy)
        result+=np.sqrt(2*power)*(coefficients[2*j]*np.cos(phase)+coefficients[2*j+1]*np.sin(phase))
    return result
fig,axes=plt.subplots(len(fits),3,figsize=(11,3.1*len(fits)),layout='constrained',squeeze=False)
for row,(key,r) in enumerate(fits.items()):
    name=key.rsplit('-',1)[0];power=next(f['power'] for f in plan['fixtures'] if f['name']==name)
    truth=slice_field(r['truth'],power);estimate=slice_field(r['fit'],power);residual=estimate-truth
    limit=max(np.max(abs(truth)),np.max(abs(estimate)))
    for col,img in enumerate([truth,estimate,residual]):
        ax=axes[row,col];scale=limit if col<2 else np.max(abs(residual))
        im=ax.imshow(img.T,origin='lower',extent=(0,1,0,1),vmin=-scale,vmax=scale,cmap='RdBu_r')
        ax.set(title=['Initial truth',f"MAP iterate (status {int(r['status'])})",'Iterate - truth'][col],xlabel='x/L',ylabel=key+'\ny/L' if col==0 else 'y/L')
        fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Initial density extrapolated to a=1; z=0; common 32x32 slice')
fig.savefig(a.output/'field_reconstruction.png',dpi=170);plt.close(fig)
summary=dict(passed=all(g['passed'] for g in gates),consistency_passed=all(g['passed'] for g in gates if g['kind']=='consistency'),gates=gates,
             convergence=convergence,force_power=force_power,finite_differences=fd,fits=fits,metadata={k:c['metadata'] for k,c in cases.items()},
             limits=['Three prescribed fixtures, finite numerical references and fixed nuisance parameters.',
                     'Refined-data MAP is an optimization/recovery diagnostic, not posterior coverage or a global-optimum proof.',
                     'No external gravity reference, cosmological derivatives, tracer/RSD/survey PM, or production scaling claim.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['fixture','family','case','field_error','gradient_error','noise_rms_units'])
    for name,families in convergence.items():
        for family,c in families.items():
            for r in c['rows']:writer.writerow([name,family,r['key'],r['field_error'],r['gradient_error'],r['noise_rms_units']])
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],failed=[g['check'] for g in gates if not g['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
