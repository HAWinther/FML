"""N16 active-marginal TSC pilot; retain strict mixing failures."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent.parent;plan=json.loads((a.input/'plan.json').read_text());science=a.input/'science';checks=[]
def check(ok,name,kind='consistency'):checks.append(dict(check=name,passed=bool(ok),kind=kind))
def meta(path):
 with path.open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
def table(path):return np.atleast_1d(np.genfromtxt(path,names=True,delimiter=',',dtype=None,encoding='utf8'))
reference=root/'pm_tsc_validation/science/reference.data.csv';check(hashlib.sha256(reference.read_bytes()).hexdigest()==plan['data_sha256'],'unchanged frozen TSC data')
check(hashlib.sha256((root/'pm_tsc_joint_validation/science/joint.physical.csv').read_bytes()).hexdigest()==plan['map_seed_sha256'],'recorded physical initialization seed')
check(plan['n']==16 and plan['m']==16 and plan['steps']==16,'user-selected N16 pilot discretization')
target_check=table(science/'target.checks.csv');check(len(target_check)==3 and max(np.max(abs(target_check[k])) for k in ['value_error','gradient_error','basis_error','roundtrip_error'])<2e-9 and np.max(target_check['fd_error'])<2e-5,'full pilot-target density, Jacobian, basis and gradient checks')
chains=[];metadata=[];predictions=[]
for i in range(4):
 prefix=science/('chain'+str(i));states=table(Path(str(prefix)+'.states.csv'));m=meta(Path(str(prefix)+'.meta.csv'));pred=table(Path(str(prefix)+'.predictive.csv'));metadata.append(m);predictions.append(pred)
 q=np.column_stack([states['q'+str(j)] for j in range(122)]);z=np.column_stack([q,states['A'],states['N']]);chains.append(z)
 check(len(states)==plan['draws'] and np.array_equal(states['draw'],np.arange(plan['draws'])),'chain '+str(i)+' complete retained draws')
 check(np.all(np.isfinite(z)) and np.all(np.isfinite(states['potential'])),'chain '+str(i)+' finite states')
 check(all(m[k]==plan[k] for k in ['n','m','steps','warmup','draws','leapfrog']) and m['active']==122 and m['id']==i and m['seed']==12000+i and m['adapt']==1,'chain '+str(i)+' protocol and seed')
 check(np.all((z[:,-2]>=-4)&(z[:,-2]<=4)&(z[:,-1]>=-16)&(z[:,-1]<=-4)),'chain '+str(i)+' prior support')
 check(m['acceptance']>.5,'chain '+str(i)+' mean acceptance','mixing');check(m['divergences']==0,'chain '+str(i)+' no retained divergences','mixing')
 check(np.array_equal(pred['draw'],np.arange(0,plan['draws'],plan['predictive_stride'])),'chain '+str(i)+' predictive stride')
chains=np.asarray(chains);diagnostics=table(science/'diagnostics.csv');check(len(diagnostics)==248,'all active coordinates/squares/scalars/energy/norm monitored')
check(np.all(diagnostics['status']=='ok'),'finite valid rank diagnostics','mixing')
check(np.all(diagnostics['rhat']<1.05),'all rank split/folded Rhat < 1.05','mixing');check(np.all(diagnostics['ess_bulk']>400),'all bulk ESS > 400','mixing');check(np.all(diagnostics['ess_tail']>200),'all tail ESS > 200','mixing')
posterior={}
for name,index in [('A',122),('N',123)]:
 v=chains[:,:,index].ravel();d=diagnostics[diagnostics['feature']==name][0];posterior[name]=dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),quantile_025=float(np.quantile(v,.025)),median=float(np.median(v)),quantile_975=float(np.quantile(v,.975)),rhat=float(d['rhat']),ess_bulk=float(d['ess_bulk']),ess_tail=float(d['ess_tail']))
noise=np.exp(chains[:,:,123].ravel());posterior['noise_power']=dict(mean=float(np.mean(noise)),quantile_025=float(np.quantile(noise,.025)),median=float(np.median(noise)),quantile_975=float(np.quantile(noise,.975)))
physical=np.sqrt(.002)*chains[:,:,:122].reshape(-1,122);truth=np.loadtxt(root/'pm_tsc_joint_validation/science/joint.physical.csv',delimiter=',')[0];mean=np.mean(physical,axis=0);relative=float(np.linalg.norm(mean-truth)/np.linalg.norm(truth));corr=float(np.dot(mean,truth)/np.linalg.norm(mean)/np.linalg.norm(truth))
f=np.fft.fftfreq(16)*16;z=np.fft.rfftfreq(16)*16;x,y,zz=np.meshgrid(f,f,z,indexing='ij');mask=(x*x+y*y+zz*zz>0)&(x*x+y*y+zz*zz<=16);weight=np.broadcast_to(np.where((z==0)|(z==8),1.,2.),mask.shape);v=np.loadtxt(reference,delimiter=',',skiprows=1).reshape(16,16,9,2);data=v[...,0]+1j*v[...,1]
rng=np.random.default_rng(19001);signals=[];replicas=[];obs_chi=[];rep_chi=[]
for p in predictions:
 for row in p:
  raw=np.array([row['m'+str(j)] for j in range(4608)]).reshape(16,16,9,2);signal=raw[...,0]+1j*raw[...,1];power=float(np.exp(row['N']));n=np.fft.rfftn(rng.normal(size=(16,16,16)))*np.sqrt(power/16**3);n[~mask]=0;replica=signal+n
  signals.append(signal);replicas.append(replica);obs_chi.append(float(np.sum(weight[mask]*abs(data[mask]-signal[mask])**2)/power));rep_chi.append(float(np.sum(weight[mask]*abs(n[mask])**2)/power))
signals=np.asarray(signals);replicas=np.asarray(replicas);check(np.all(np.isfinite(signals)) and np.all(np.isfinite(replicas)),'finite latent and noisy predictions');ppp=float(np.mean(np.asarray(rep_chi)>=obs_chi))
map_control=meta(science/'map.controls.csv');check(all(map_control[k]==plan[k] for k in ['n','m','steps']) and map_control['vary']==1,'same-target N16 MAP baseline');check(hashlib.sha256((science/'map.data.csv').read_bytes()).hexdigest()==plan['data_sha256'],'MAP and chains use identical data');map_result=meta(science/'map.fit.csv')
summary=dict(map_baseline=map_result,passed=all(c['passed'] for c in checks),consistency_passed=all(c['passed'] for c in checks if c['kind']=='consistency'),mixing_passed=all(c['passed'] for c in checks if c['kind']=='mixing'),checks=checks,posterior=posterior,metadata=metadata,
 diagnostic_extrema=dict(max_rhat=float(np.max(diagnostics['rhat'])),min_bulk_ess=float(np.min(diagnostics['ess_bulk'])),min_tail_ess=float(np.min(diagnostics['ess_tail']))),mean_physical_field=dict(relative_error=relative,correlation=corr),predictive=dict(draws=len(signals),observed_chi2_mean=float(np.mean(obs_chi)),replicated_chi2_mean=float(np.mean(rep_chi)),tail_fraction=ppp),
 limits=['N16 particle pilot changes the N32 MAP discretization; N32 posterior mixing remains unvalidated.','Active posterior sampled in physical coordinates with the 122*A/2 Jacobian; inactive Gaussian white directions integrated exactly.','One finite reference and realization; posterior intervals are exploratory if mixing gates fail, not repeated-mock coverage.','No post-warmup thinning; predictive saves only use a stride.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'parameter_intervals.csv').open('w') as f:
 writer=csv.writer(f,lineterminator='\n');writer.writerow(['parameter','mean','lower_025','median','upper_975'])
 for key,v in posterior.items():writer.writerow([key,v['mean'],v['quantile_025'],v['median'],v['quantile_975']])
fig,axes=plt.subplots(2,2,figsize=(11,6.6),layout='constrained');joint=meta(science/'map.fit.csv')
for i in range(4):axes[0,0].plot(chains[i,:,122],lw=.8,label='Chain '+str(i));axes[0,1].plot(np.exp(chains[i,:,123])/1e-4,lw=.8)
axes[0,0].axhline(0,color='black',ls=':');axes[0,0].set(title='Log-amplitude traces',xlabel='Retained draw',ylabel='A');axes[0,0].legend(fontsize=7);axes[0,1].axhline(1,color='black',ls=':');axes[0,1].set(title='Noise-power traces',xlabel='Retained draw',ylabel='Noise power / injected')
axes[1,0].hist(chains[:,:,122].ravel(),bins=24,density=True,alpha=.6,label='Pilot marginal');grid=np.linspace(-1.3,2.2,400);axes[1,0].plot(grid,np.exp(-.5*(grid/.5)**2)/(.5*np.sqrt(2*np.pi)),label='Prior');axes[1,0].axvline(joint['log_amplitude'],color='tab:red',ls='--',label='N16 whitened joint MAP');axes[1,0].axvline(0,color='black',ls=':');axes[1,0].set(title='Amplitude marginal vs coordinate MAP',xlabel='A',ylabel='Density');axes[1,0].legend(fontsize=7)
axes[1,1].hist(noise/1e-4,bins=24,density=True,alpha=.6);axes[1,1].axvline(1,color='black',ls=':',label='Injected');axes[1,1].axvline(np.exp(joint['log_noise_power'])/1e-4,color='tab:red',ls='--',label='N16 whitened joint MAP');axes[1,1].set(title='Noise-power marginal',xlabel='Noise power / injected',ylabel='Density');axes[1,1].legend(fontsize=7)
fig.suptitle('N16 bounded pilot; strict mixing gates '+('passed' if summary['mixing_passed'] else 'NOT passed'));fig.savefig(a.output/'parameters.png',dpi=170);fig.savefig(a.output/'parameters.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained');axes[0].hist(diagnostics['rhat'],bins=25);axes[0].axvline(1.05,color='red',ls=':');axes[0].set(title='Rank split/folded Rhat',xlabel='Rhat',ylabel='Features');axes[1].hist(diagnostics['ess_bulk'],bins=25);axes[1].axvline(400,color='red',ls=':');axes[1].set(title='Bulk ESS',xlabel='ESS');axes[2].hist(diagnostics['ess_tail'],bins=25);axes[2].axvline(200,color='red',ls=':');axes[2].set(title='Tail ESS',xlabel='ESS');fig.suptitle('All 248 monitored features; unchanged finite-chain criteria');fig.savefig(a.output/'mixing.png',dpi=170);plt.close(fig)
waves=[(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if 0<i*i+j*j+k*k<=9 and (i>0 or(i==0 and j>0)or(i==0 and j==0 and k>0))];xx,yy=np.meshgrid(np.arange(64)/64,np.arange(64)/64,indexing='ij');basis=[]
for i,k,l in waves:
 phase=2*np.pi*(i*xx+k*yy);basis.extend([np.sqrt(2)*np.cos(phase),np.sqrt(2)*np.sin(phase)])
basis=np.asarray(basis).reshape(122,-1);fields=physical@basis;truth_field=(truth@basis).reshape(64,64);field_mean=np.mean(fields,axis=0).reshape(64,64);field_sd=np.std(fields,axis=0,ddof=1).reshape(64,64);fig,axes=plt.subplots(1,4,figsize=(14,3.4),layout='constrained');lim=max(np.max(abs(truth_field)),np.max(abs(field_mean)))
for ax,image,title in zip(axes,[truth_field,field_mean,field_mean-truth_field,field_sd],['Initial truth','Pilot mean','Mean - truth','Pointwise latent SD']):
 sd=title.endswith('SD');im=ax.imshow(image.T,origin='lower',extent=(0,1,0,1),cmap='viridis' if sd else 'RdBu_r',vmin=0 if sd else -lim,vmax=float(np.max(field_sd)) if sd else lim);ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Physical initial density at a=1, z=0; pilot summaries require mixing checks');fig.savefig(a.output/'initial_fields.png',dpi=170);plt.close(fig)
def real_slice(m):return (np.fft.irfftn(m,s=(16,16,16),axes=(0,1,2))*16**3)[:,:,0]
latent=np.asarray([real_slice(m) for m in signals]);noisy=np.asarray([real_slice(m) for m in replicas]);obs=real_slice(data);fig,axes=plt.subplots(1,4,figsize=(14,3.4),layout='constrained');mean=np.mean(latent,axis=0);images=[obs,mean,np.std(latent,axis=0,ddof=1),np.std(noisy,axis=0,ddof=1)];lim=max(np.max(abs(obs)),np.max(abs(mean)))
for j,(ax,image,title) in enumerate(zip(axes,images,['Frozen observed data','Pilot signal mean','Latent signal SD','Noisy predictive SD'])):
 im=ax.imshow(image.T,origin='lower',extent=(0,1,0,1),cmap='viridis' if j>=2 else 'RdBu_r',vmin=0 if j>=2 else -lim,vmax=None if j>=2 else lim);ax.set(title=title,xlabel='x/L',ylabel='y/L');fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('M16 observed band; latent and measurement-noise uncertainty separated');fig.savefig(a.output/'predictive.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained');ax.hist(obs_chi,bins=20,alpha=.6,label='Observed-data discrepancy');ax.hist(rep_chi,bins=20,alpha=.6,label='Replicated-data discrepancy');ax.set(xlabel='Full-Hermitian chi-square at each sampled noise power',ylabel='Saved draws',title=f'Exploratory predictive tail fraction {ppp:.3f}');ax.legend(fontsize=8);fig.savefig(a.output/'predictive_check.png',dpi=170);plt.close(fig)
print(json.dumps(dict(passed=summary['passed'],consistency_passed=summary['consistency_passed'],mixing_passed=summary['mixing_passed'],failed=[c['check'] for c in checks if not c['passed']],posterior=posterior,diagnostics=summary['diagnostic_extrema']),indent=2))
if not summary['passed']:raise SystemExit(1)
