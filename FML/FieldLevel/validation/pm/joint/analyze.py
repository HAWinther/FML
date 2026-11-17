"""Joint TSC MAP checks, physical-field recovery and parameterization diagnostics."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent.parent;plan=json.loads((a.input/'plan.json').read_text());prefix=a.input/'science/joint';checks=[]
def check(ok,text,kind='consistency'):checks.append(dict(check=text,passed=bool(ok),kind=kind))
def table(suffix):return np.atleast_1d(np.genfromtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
def metadata(suffix):
 with Path(str(prefix)+'.'+suffix+'.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
def modes(path,skip=0):
 v=np.loadtxt(path,delimiter=',',skiprows=skip).reshape(16,16,9,2);return v[...,0]+1j*v[...,1]
fit=metadata('fit');control=metadata('controls');t=table('trace');ids=table('identities');fd=table('fd');physical=np.loadtxt(str(prefix)+'.physical.csv',delimiter=',');co=np.loadtxt(str(prefix)+'.coefficients.csv',delimiter=',')
check(hashlib.sha256(Path(str(prefix)+'.data.csv').read_bytes()).hexdigest()==plan['data_sha256'],'frozen matched TSC data unchanged')
check(all(control[k]==plan['fit'][k] for k in ['n','m','steps']) and control['max_iterations']==128 and control['vary']==1 and control['fd']==1,'declared joint protocol')
check(control['amplitude_lower']==-4 and control['amplitude_upper']==4 and control['amplitude_mean']==0 and control['amplitude_sigma']==.5 and control['noise_sigma']==.5,'declared Gaussian priors and widened amplitude support')
for rows in [t,ids,fd]:
 for name in rows.dtype.names:
  if rows.dtype[name].kind in 'iuf':check(np.all(np.isfinite(rows[name])),'finite '+name)
check(len(ids)==3 and set(ids['point'])=={'truth','start','endpoint'},'complete analytic identity states')
check(max(np.max(abs(ids['amplitude_error'])),np.max(abs(ids['noise_error'])))<2e-9,'independent amplitude and normalized-noise gradient identities')
check(fit['forwards']+fit['unsupported_trials']==fit['value_evaluations']+fit['gradient_evaluations']+fit['reference_evaluations'] and fit['pullbacks']==fit['gradient_evaluations'],'work counts include support rejections without forward predictions')
check(fit['unsupported_trials']>=0 and fit['peak_history_pairs']<=5,'bounded trials/history')
check(len(t)==fit['iterations']+1 and t[-1]['potential']==fit['potential'] and t[-1]['log_amplitude']==fit['log_amplitude'] and t[-1]['log_noise_power']==fit['log_noise_power'] and t[-1]['gradient_norm']==fit['joint_gradient_norm'],'accepted parameter trace matches endpoint')
check(np.all(t['accepted_change'][1:]<=t['armijo_bound'][1:]),'all accepted stable changes satisfy Armijo')
check(fit['potential']<fit['initial_potential'] and fit['log_amplitude']!=0 and fit['log_noise_power']!=control['noise_mean'],'field and both scalar parameters updated')
e=ids[ids['point']=='endpoint'][0];check(abs(max(e['field_max'],e['scalar_scaled_max'])-fit['joint_gradient_norm'])<1e-12,'fresh endpoint reproduces joint stopping norm')
check(fit['status']==0 and fit['joint_gradient_norm']<2e-5,'joint MAP converges at unchanged numerical tolerance','convergence')
check(-3.9<fit['log_amplitude']<3.9 and -15.9<fit['log_noise_power']<-4.1,'solution stays inside declared support','convergence')
check(physical.shape==(3,122) and co.shape==(3,122) and np.all(np.isfinite(physical)),'finite physical and white initial coefficients')
check(np.max(abs(physical[0]-np.sqrt(plan['power'])*co[0]))<2e-12 and np.max(abs(physical[2]-np.sqrt(plan['power']*np.exp(fit['log_amplitude']))*co[2]))<2e-12,'physical field uses the fitted spectrum amplitude')
relative=float(np.linalg.norm(physical[2]-physical[0])/np.linalg.norm(physical[0]));correlation=float(np.dot(physical[0],physical[2])/np.linalg.norm(physical[0])/np.linalg.norm(physical[2]));check(relative<.8 and correlation>.7,'physical initial-field recovery','recovery')
fd_rows=[];check(len(fd)==42,'complete two-state three-direction derivative scan')
for point in ['start','endpoint']:
 for direction in [0,1,2]:
  v=fd[(fd['point']==point)&(fd['direction']==direction)];err=abs(v['numeric']-v['analytic'])/np.maximum(1.,abs(v['analytic']));tail=err[-4:];gate=len(v)==7 and np.min(tail)<2e-5 and np.sum(tail<1e-4)>=2;check(gate,point+' direction '+str(direction)+' refined derivative','derivative');fd_rows.append(dict(point=point,direction=direction,best_error=float(np.min(tail))))
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16;x,y,zz=np.meshgrid(f,f,z,indexing='ij');mask=(x*x+y*y+zz*zz>0)&(x*x+y*y+zz*zz<=16);weights=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),mask.shape)
noise=np.sqrt(plan['noise_power']*weights[mask].sum());data=modes(str(prefix)+'.data.csv',1);pred=modes(str(prefix)+'.fit_modes.csv');residual=float(np.sqrt(np.sum(weights[mask]*abs(pred[mask]-data[mask])**2))/noise)
fixed=json.loads((root/'pm_tsc_validation/summary.json').read_text())['row'];summary=dict(passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),checks=checks,fit=fit,physical_recovery=dict(relative_error=relative,correlation=correlation,residual_true_noise_units=residual),identities=[{k:(str(v[k]) if ids.dtype[k].kind in 'US' else float(v[k])) for k in ids.dtype.names} for v in ids],derivatives=fd_rows,
 parameterization=dict(amplitude_multiplier=float(np.exp(fit['log_amplitude'])),noise_power=float(np.exp(fit['log_noise_power'])),noise_power_ratio=float(np.exp(fit['log_noise_power'])/plan['noise_power']),amplitude_stationary_rhs=float(e['white_norm2']/8),noise_chi2=float(e['chi2'])),fixed_baseline=fixed,
 limits=['Joint MAP is a density mode in whitened coordinates, not a marginalized parameter estimate.','Gaussian amplitude mean/scale are unchanged, but support is explicitly widened from [-1,1] to [-4,4].','One fixed realization; broader refinement, posterior sampling and coverage remain deferred.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');axes[0].plot(t['iteration'],t['log_amplitude'],label='Joint MAP path');axes[0].axhline(0,color='black',ls=':',label='Injected / prior mean');axes[0].set(xlabel='Accepted iterations',ylabel='Log power amplitude A',title='Amplitude in whitened coordinates');axes[0].legend(fontsize=7)
axes[1].plot(t['iteration'],np.exp(t['log_noise_power'])/plan['noise_power']);axes[1].axhline(1,color='black',ls=':');axes[1].set_yscale('log');axes[1].set(xlabel='Accepted iterations',ylabel='Noise power / injected noise power',title='Inferred noise');axes[2].semilogy(t['iteration'],t['gradient_norm']);axes[2].axhline(2e-5,color='gray',ls=':');axes[2].set(xlabel='Accepted iterations',ylabel='max(raw field, scaled scalar gradient)',title='Joint stopping criterion');fig.suptitle('Joint TSC MAP; parameters are modes, not posterior means');fig.savefig(a.output/'parameters.png',dpi=170);fig.savefig(a.output/'parameters.pdf');plt.close(fig)
waves=[(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if 0<i*i+j*j+k*k<=9 and (i>0 or (i==0 and j>0) or (i==0 and j==0 and k>0))];xx,yy=np.meshgrid(np.arange(64)/64,np.arange(64)/64,indexing='ij')
def initial_slice(c):
 v=np.zeros((64,64))
 for j,(i,k,l) in enumerate(waves):
  phase=2*np.pi*(i*xx+k*yy);v+=np.sqrt(2)*(c[2*j]*np.cos(phase)+c[2*j+1]*np.sin(phase))
 return v
truth=initial_slice(physical[0]);estimate=initial_slice(physical[2]);lim=max(np.max(abs(truth)),np.max(abs(estimate)));fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for ax,v,title in zip(axes,[truth,estimate,estimate-truth],['Initial truth','Joint MAP physical initial field','Reconstruction - truth']):
 im=ax.imshow(v.T,origin='lower',extent=(0,1,0,1),vmin=-lim,vmax=lim,cmap='RdBu_r');ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle(f'Physical initial density at a=1; z=0; error {relative:.3f}, r={correlation:.3f}');fig.savefig(a.output/'initial_fields.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained');images=[(np.fft.irfftn(v,s=(16,16,16),axes=(0,1,2))*16**3)[:,:,0] for v in [data,pred,pred-data]];lim=max(np.max(abs(v)) for v in images)
for ax,v,title in zip(axes,images,['Frozen noisy data','Joint MAP signal','Fit - data']):
 im=ax.imshow(v.T,origin='lower',extent=(0,1,0,1),vmin=-lim,vmax=lim,cmap='RdBu_r');ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Common M16 observation window; z=0; common color scale');fig.savefig(a.output/'residuals.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,point in zip(axes,['start','endpoint']):
 for direction,label in [(0,'White'),(1,'Scalars'),(2,'Joint')]:
  v=fd[(fd['point']==point)&(fd['direction']==direction)];ax.loglog(v['h'],np.maximum(abs(v['numeric']-v['analytic'])/np.maximum(1.,abs(v['analytic'])),1e-16),'o-',label=label)
 ax.axhline(2e-5,color='gray',ls=':');ax.invert_xaxis();ax.set(title=point,xlabel='Perturbation h',ylabel='Scaled derivative error');ax.legend(fontsize=7)
fig.suptitle('Stable central differences for the full joint posterior');fig.savefig(a.output/'derivatives.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],failed=[c['check'] for c in checks if not c['passed']],fit=fit,recovery=summary['physical_recovery'],parameters=summary['parameterization']),indent=2))
if not summary['passed']:raise SystemExit(1)
