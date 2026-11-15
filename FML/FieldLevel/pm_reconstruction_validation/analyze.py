"""Refined-data recovery and explicit MAP endpoint diagnostics; retain failures."""
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
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads((a.input/'plan.json').read_text());checks=[]
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def values(prefix,suffix,skip=0):return np.atleast_1d(np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=skip))
def metadata(prefix,suffix):
    with Path(str(prefix)+'.'+suffix+'.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
def modes(prefix,suffix='modes',skip=0):
    v=values(prefix,suffix,skip).reshape(16,16,9,2);return v[...,0]+1j*v[...,1]
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16
x,y,zz=np.meshgrid(f,f,z,indexing='ij');q=x*x+y*y+zz*zz;mask=(q>0)&(q<=16)
weight=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),q.shape);dof=float(weight[mask].sum())
def norm(v):return float(np.sqrt(np.sum(weight[mask]*abs(v[mask])**2)))
noise=np.sqrt(plan['noise_power']*dof);ref=a.input/'science/reference'
data_path=Path(str(ref)+'.data.csv');data=modes(ref,'data',1);signal=modes(ref);ref_meta=metadata(ref,'forward_meta')
check(hashlib.sha256(data_path.read_bytes()).hexdigest()==plan['data_sha256'],'frozen refined-data hash')
check(dof==plan['observed'],'independent observed degrees')
check(ref_meta['n']==128 and ref_meta['m']==64 and ref_meta['steps']==32,'reference resolution')
check(ref_meta['max_mass_error']<1e-10 and ref_meta['max_mean_momentum']<1e-10,'reference conservation')
check(data_path.read_text().splitlines()[0].endswith(',observation_cic,16'),'new mock declared observation space')
check(np.all(np.isfinite(signal)) and np.max(abs(signal[~mask]))==0,'finite supported reference signal')
rows=[];details={};prior_truth=None;prior_start=None;start_modes=None
for budget in plan['budgets']:
    prefix=a.input/f'science/fit-{budget}';co=values(prefix,'fit_coefficients');truth,start,fit,g=co
    mt=metadata(prefix,'meta');ft=metadata(prefix,'fit');pred=modes(prefix,'fit_modes');sm=modes(prefix,'fit_start_modes')
    check(co.shape==(4,122) and np.all(np.isfinite(co)) and np.all(np.isfinite(pred)) and all(np.isfinite(v) for v in ft.values()),f'finite fit {budget}')
    check(Path(str(prefix)+'.data.csv').read_bytes()==data_path.read_bytes(),f'identical refined data {budget}')
    check(all(mt[k]==plan['fit'][k] for k in ['n','m','steps']),f'lower-resolution fit {budget}')
    check(np.max(abs(values(prefix,'initial')-values(ref,'initial')))<2e-12,f'matched physical ICs {budget}')
    check(np.max(abs(values(prefix,'initial')-np.sqrt(plan['power'])*truth))<2e-12,f'independent initial normalization {budget}')
    for owner in [ref,prefix]:
        check(values(owner,'window',1)[0]==1 and values(owner,'observation',1)[0]==16,f'compensated force and common window {owner.name}')
    if prior_truth is not None:
        check(np.array_equal(truth,prior_truth) and np.array_equal(start,prior_start) and np.array_equal(sm,start_modes),f'identical starts {budget}')
    prior_truth,prior_start,start_modes=truth,start,sm
    relative=float(np.linalg.norm(fit-truth)/np.linalg.norm(truth));correlation=float(np.dot(truth,fit)/np.linalg.norm(truth)/np.linalg.norm(fit))
    row=dict(budget=budget,iterations=int(ft['iterations']),status=int(ft['status']),potential=ft['potential'],initial_potential=ft['initial_potential'],
             raw_gradient=ft['gradient_norm'],projected_gradient=ft['projected_gradient_norm'],relative_initial_error=relative,correlation=correlation,
             residual_noise_units=norm(pred-data)/noise,initial_residual_noise_units=norm(sm-data)/noise,
             signal_error_noise_units=norm(pred-signal)/noise,truth_discretization_noise_units=norm(modes(prefix)-signal)/noise)
    check(row['potential']<=row['initial_potential'],f'objective decreases {budget}')
    check(0<=row['iterations']<=budget and row['status'] in [0,1,2],f'valid optimizer status {budget}')
    check(abs(np.linalg.norm(g)-row['projected_gradient'])<1e-10,f'projected-gradient metadata {budget}')
    if rows:check(row['potential']<=rows[-1]['potential']+1e-10,f'objective across nested budgets {budget}')
    rows.append(row);details[str(budget)]=dict(truth=truth.tolist(),start=start.tolist(),fit=fit.tolist(),gradient=g.tolist())
final=rows[-1];check(final['relative_initial_error']<plan['recovery']['relative_error'],'final coefficient recovery','recovery')
check(final['correlation']>plan['recovery']['correlation'],'final initial-field correlation','recovery')
check(final['status']==0 and final['raw_gradient']<plan['gradient_tolerance'],'final optimizer convergence','convergence')
summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),
             recovery_passed=all(c['passed'] for c in checks if c['kind']=='recovery'),checks=checks,rows=rows,coefficients=details,
             reference_metadata=ref_meta,noise_realization_rms_units=norm(data-signal)/noise,
             limits=['One compensated mock and fixed nuisance parameters; broader refinement checks deferred.',
                     'Finite numerical reference, not continuum or independent physics.',
                     'Iteration-budget endpoints, not uncertainty or posterior coverage; raw grid gradient is resolution dependent.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'recovery.csv').open('w') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
# Endpoints of independent deterministic restarts, not all accepted steps.
it=[r['iterations'] for r in rows];fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
axes[0].plot(it,[r['potential']-rows[0]['initial_potential'] for r in rows],'o-');axes[0].set(ylabel='U - U(start)',xlabel='Accepted iterations',title='Objective change')
for label,key in [('raw cell maximum','raw_gradient'),('projected L2','projected_gradient')]:axes[1].semilogy(it,[r[key] for r in rows],'o-',label=label)
axes[1].axhline(plan['gradient_tolerance'],ls=':',color='gray',label='raw tolerance');axes[1].set(xlabel='Accepted iterations',title='Gradient diagnostics');axes[1].legend(fontsize=8)
axes[2].plot(it,[r['relative_initial_error'] for r in rows],'o-',label='Initial coefficient error');axes[2].plot(it,[r['residual_noise_units'] for r in rows],'s-',label='Data residual / noise');axes[2].set(xlabel='Accepted iterations',title='Recovery and residual');axes[2].legend(fontsize=8)
fig.suptitle('N128/M64/32-step data → N32/M16/16-step fit; independent budget endpoints')
fig.savefig(a.output/'optimization.png',dpi=170);fig.savefig(a.output/'optimization.pdf');plt.close(fig)
last=details[str(plan['budgets'][-1])];truth=np.array(last['truth']);fit=np.array(last['fit'])
waves=[(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if 0<i*i+j*j+k*k<=9 and (i>0 or (i==0 and j>0) or (i==0 and j==0 and k>0))]
xx,yy=np.meshgrid(np.arange(64)/64,np.arange(64)/64,indexing='ij')
def initial_slice(c):
    v=np.zeros((64,64))
    for j,(i,k,l) in enumerate(waves):
        ph=2*np.pi*(i*xx+k*yy);v+=np.sqrt(2*plan['power'])*(c[2*j]*np.cos(ph)+c[2*j+1]*np.sin(ph))
    return v
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained');t=initial_slice(truth);est=initial_slice(fit);lim=max(np.max(abs(t)),np.max(abs(est)))
for ax,v,title in zip(axes,[t,est,est-t],['Initial truth',f"Reconstruction ({final['iterations']} iterations)",'Reconstruction - truth']):
    im=ax.imshow(v.T,origin='lower',extent=(0,1,0,1),vmin=-lim,vmax=lim,cmap='RdBu_r');ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Initial density extrapolated to a=1; z=0; common color scale')
fig.savefig(a.output/'initial_fields.png',dpi=170);fig.savefig(a.output/'initial_fields.pdf');plt.close(fig)
# Retained observed signal/data rendered on the common output mesh, not native density.
final_modes=modes(a.input/f"science/fit-{plan['budgets'][-1]}",'fit_modes')
def observed_slice(c):return (np.fft.irfftn(c,s=(16,16,16),axes=(0,1,2))*16**3)[:,:,0]
images=[observed_slice(signal),observed_slice(data),observed_slice(final_modes),observed_slice(final_modes-data)]
fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout='constrained');lim=max(np.max(abs(v)) for v in images[:3])
for ax,v,title in zip(axes,images,['Refined noiseless signal','Fixed noisy data','Fitted signal','Fit - data']):
    im=ax.imshow(v.T,origin='lower',extent=(0,1,0,1),vmin=-lim,vmax=lim,cmap='RdBu_r');ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Observed-band final density; fixed M16 CIC response; z=0; common color scale')
fig.savefig(a.output/'observed_fields.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(4.7,4.2),layout='constrained');ax.scatter(truth,fit,s=15,alpha=.7);lim=max(np.max(abs(truth)),np.max(abs(fit)));ax.plot([-lim,lim],[-lim,lim],'k:');ax.set(xlabel='True initial white coefficient',ylabel='Fitted coefficient',title=f"Error {final['relative_initial_error']:.3f}; r={final['correlation']:.3f}");fig.savefig(a.output/'coefficients.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],recovery_passed=summary['recovery_passed'],final=final,failed=[c['check'] for c in checks if not c['passed']]),indent=2))
if not summary['passed']:raise SystemExit(1)
