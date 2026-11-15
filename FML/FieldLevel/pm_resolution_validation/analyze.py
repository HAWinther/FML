"""Matched-field PM discretization checks and larger-mode posterior diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from prepare import basis
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
gates=[]
def check(valid,text):gates.append(dict(check=text,passed=bool(valid)))
def load(prefix,suffix,header=False):return np.atleast_2d(np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=int(header)))
def meta(prefix):
    with Path(str(prefix)+'.meta.csv').open() as file:return {k:float(v) for k,v in next(csv.DictReader(file)).items()}
def modes(prefix,suffix='modes'):
    x=load(prefix,suffix).reshape(8,8,5,2);return x[...,0]+1j*x[...,1]
def geometry(n):
    f=np.fft.fftfreq(n)*n;z=np.fft.rfftfreq(n)*n;xx,yy,zz=np.meshgrid(f,f,z,indexing='ij');q=xx*xx+yy*yy+zz*zz
    weight=np.broadcast_to(np.where((z==0)|(z==n//2),1.,2.),q.shape);return q,weight
q,w=geometry(8);mask=(q>0)&(q<=5);shells=np.arange(1,6);dof=float(w[mask].sum())
def norm(x):return float(np.sqrt(np.sum(w[mask]*np.abs(x[mask])**2)))
def power(fk):return np.stack([np.sum(np.abs(fk[...,q==s])**2*w[q==s],axis=-1)/w[q==s].sum() for s in shells],axis=-1)
plan=json.loads((a.input/'plan.json').read_text());cases={};families={};reference=load(a.input/'science/data-reference','initial').ravel();fd_results={}
for case in plan['cases']:
    n,m,s=case['n'],case['m'],case['steps'];prefix=a.input/f'science/case-{n}-{m}-{s}';key=f'{n}-{m}-{s}'
    initial=load(prefix,'initial').ravel();gradient=load(prefix,'gradient').ravel();prediction=modes(prefix)
    check(initial.shape==(18,) and gradient.shape==(20,) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(prediction)),key+': finite common coordinates')
    check(np.max(np.abs(initial-reference))<2e-12,key+': matched physical initial Fourier field')
    check(np.max(np.abs(prediction[~mask]))==0,key+': common observed Fourier support')
    cases[key]=dict(n=n,m=m,steps=s,prediction=prediction,gradient=gradient,metadata=meta(prefix))
    for family in case['families']:families.setdefault(family,[]).append(key)
    if Path(str(prefix)+'.fd.csv').exists():
        rows=np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv',delimiter=',',names=True));fd_results[key]={}
        for probe in [0,1]:
            r=rows[rows['probe']==probe];err=np.abs(r['finite_difference']-r['analytic'])/np.maximum(1,np.abs(r['analytic']))
            check(np.all(np.isfinite(err)) and len(r)==8,key+f': finite FD probe {probe}')
            check(np.min(err[-4:])<2e-5 and np.sum(err[-4:]<1e-4)>=2,key+f': stable refined FD probe {probe}')
            fd_results[key][str(probe)]=dict(h=r['h'].tolist(),relative_error=err.tolist(),analytic=float(r['analytic'][0]))
check(len(fd_results)==2,'two independent resolution FD checks')
convergence={}
for family,keys in families.items():
    variable={'force':'m','time':'steps','particle':'n'}[family];keys.sort(key=lambda k:cases[k][variable]);fine=cases[keys[-1]];rows=[]
    for key in keys:
        c=cases[key];delta=c['prediction']-fine['prediction'];field_error=norm(delta)/max(norm(fine['prediction']),1e-30)
        gradient_error=float(np.linalg.norm(c['gradient']-fine['gradient'])/max(np.linalg.norm(fine['gradient']),1e-30))
        sigma_units=norm(delta)/np.sqrt(.0001*dof)
        rows.append(dict(key=key,resolution=c[variable],field_error=field_error,gradient_error=gradient_error,noise_rms_units=sigma_units,
                         power_fraction=(power(c['prediction'])/power(fine['prediction'])-1).tolist()))
    # These are bounded error checks for this one field, not a claimed universal order.
    check(rows[-2]['field_error']<rows[0]['field_error'],family+': finest correction improves common-band field')
    check(rows[-2]['gradient_error']<rows[0]['gradient_error'],family+': finest correction improves projected posterior gradient')
    check(rows[-2]['field_error']<.05 and rows[-2]['gradient_error']<.15,family+': bounded finest field/gradient correction')
    check(rows[-2]['noise_rms_units']<.1,family+': finest field correction below 0.1 noise SD per observed degree')
    convergence[family]=dict(variable=variable,reference=keys[-1],rows=rows)
# Every retained coordinate is monitored; exact Gaussian null moments are a reference.
with (a.input/'science/diagnostics.csv').open() as file:diag={r['feature']:{k:float(v) if k not in ['feature','status'] else v for k,v in r.items()} for r in csv.DictReader(file)}
prefixes=[a.input/f'science/chain{i}' for i in range(4)];metadata=[meta(p) for p in prefixes];n=int(metadata[0]['n']);m=int(metadata[0]['m']);cells=n**3
states=[load(p,'states',True) for p in prefixes];pred=[load(p,'predictive',True) for p in prefixes];pot=[load(p,'potential',True) for p in prefixes]
for i,(s,d,u,mt) in enumerate(zip(states,pred,pot,metadata)):
    check(s.shape==(int(mt['draws']),cells+3) and np.all(np.isfinite(s)) and np.array_equal(s[:,0],np.arange(len(s))),f'chain{i}: full finite consecutive white/scalar states')
    check(np.array_equal(u[:,0],s[:,0]) and np.all(np.isfinite(u)),f'chain{i}: complete potential records')
    check(np.array_equal(d[:,0],np.arange(0,len(s),int(mt['stride']))) and d.shape[1]==1+cells+m**3 and np.all(np.isfinite(d)),f'chain{i}: predictive schedule')
    check(mt['divergences']==0 and .6<mt['acceptance']<=1,f'chain{i}: post-warmup divergences/acceptance')
    check(np.all(np.abs(s[:,-2])<=1) and np.all((s[:,-1]>=-16)&(s[:,-1]<=-4)),f'chain{i}: scalar support')
for feature,d in diag.items():check(d['status']=='ok' and d['rhat']<1.05 and d['ess_bulk']>200 and d['ess_tail']>100,'mixing '+feature)
active,null=basis(n);total=sum(map(len,states));null_errors=[]
for i in range(cells-18):
    for name,expected,var in [(f'null{i}',0.,1.),(f'null_square{i}',1.,2.)]:
        d=diag[name];se=np.sqrt(var/max(1.,min(total,d['ess_bulk'])));z=abs(d['mean']-expected)/se;null_errors.append(z);check(z<5,'analytic moment '+name)
d=diag['null_norm'];check(abs(d['mean']-(cells-18))<5*np.sqrt(2*(cells-18)/max(1,min(total,d['ess_bulk']))),'analytic null squared norm')
truth=[np.fromstring(x,sep=',') for x in Path(str(prefixes[0])+'.truth.csv').read_text().splitlines()];tw,ti,tm=truth
check(tw.shape==(cells,) and ti.shape==(cells,) and tm.shape==(m**3,),'canonical larger-grid truth')
for prefix in prefixes:check(Path(str(prefix)+'.truth.csv').read_bytes()==Path(str(prefixes[0])+'.truth.csv').read_bytes() and Path(str(prefix)+'.data.csv').read_bytes()==Path(str(prefixes[0])+'.data.csv').read_bytes(),'same mock '+prefix.name)
samples=np.stack(states);white=samples[:,:,1:1+cells];z=white@active;density=np.concatenate(pred);parameters=np.concatenate([s[d[:,0].astype(int),-2:] for s,d in zip(states,pred)])
initial=density[:,1:1+cells];matter=density[:,1+cells:];selected_white=np.concatenate([s[d[:,0].astype(int),1:1+cells] for s,d in zip(states,pred)])
def fft(x,mesh):return np.fft.rfftn(x.reshape((-1,mesh,mesh,mesh)),axes=(-3,-2,-1),norm='forward')
qi,wi=geometry(n);expected=fft(selected_white,n)*np.sqrt(cells*.002*np.exp(parameters[:,0]))[:,None,None,None]*((qi==1)|(qi==2))
dft_error=float(np.max(np.abs(fft(initial,n)-expected)));check(dft_error<2e-12,'larger-grid direct DFT initial normalization')
qm,wm=geometry(m);observed=(qm>0)&(qm<=5)
def band(x):return np.fft.irfftn(fft(x,m)*observed,s=(m,m,m),axes=(-3,-2,-1),norm='forward').reshape(-1,m**3)
banded=band(matter);true_band=band(tm)[0]
common=modes(prefixes[0],'data');data=np.zeros((m,m,m//2+1),complex)
for x in range(-2,3):
    for y in range(-2,3):
        for zz in range(3):
            if 0<x*x+y*y+zz*zz<=5:data[x%m,y%m,zz]=common[x%8,y%8,zz]
rng=np.random.default_rng(551);noise=rng.normal(size=(len(matter),m,m,m))*np.sqrt(np.exp(parameters[:,1])*m**3)[:,None,None,None]
noise_fft=np.fft.rfftn(noise,axes=(-3,-2,-1),norm='forward');prediction=fft(matter,m);variance=np.exp(parameters[:,1])
chi_obs=np.sum(wm[observed]*np.abs(prediction[:,observed]-data[observed])**2,axis=1)/variance
chi_rep=np.sum(wm[observed]*np.abs(noise_fft[:,observed])**2,axis=1)/variance;ppc=float(np.mean(chi_rep>=chi_obs))
check(np.all(np.isfinite(chi_obs)) and 0<ppc<1,'nondegenerate larger-grid posterior predictive discrepancy')
check(abs(chi_rep.mean()-dof)<5*np.sqrt(2*dof/len(matter)),'replicated noise normalization')
inference=dict(n=n,m=m,active=18,null=cells-18,metadata=metadata,features=len(diag),max_rhat=max(x['rhat'] for x in diag.values()),
    min_bulk_ess=min(x['ess_bulk'] for x in diag.values()),min_tail_ess=min(x['ess_tail'] for x in diag.values()),
    max_null_moment_standard_errors=max(null_errors),null_norm_mean=diag['null_norm']['mean'],initial_dft_error=dft_error,
    initial_correlation=float(np.corrcoef(initial.mean(0),ti)[0,1]),final_band_correlation=float(np.corrcoef(banded.mean(0),true_band)[0,1]),
    posterior_predictive_tail_fraction=ppc,chi_observed_mean=float(chi_obs.mean()),chi_replicate_mean=float(chi_rep.mean()),
    scalar_summary={name:dict(mean=float(samples[:,:,index].mean()),interval95=np.quantile(samples[:,:,index],[.025,.975]).tolist()) for name,index in [('log_amplitude',-2),('log_noise',-1)]})
# Optional short engineering replays are separate from the scientific chains.
replay={};base=a.input/'replay'
for name in ['mpi1','mpi2','mpi4','pencil4','pencil4-thread1','sanitizer']:
    if not (base/f'{name}-case.meta.csv').exists():continue
    err=[]
    for suffix in ['modes','initial','gradient']:
        x=load(base/'serial-case',suffix);y=load(base/(name+'-case'),suffix);check(x.shape==y.shape,name+' case shape '+suffix);err.append(float(np.max(np.abs(x-y))))
    source='serial8-chain' if name=='mpi4' else 'serial-chain'
    for suffix in ['states','potential','predictive']:
        x=load(base/source,suffix,True);y=load(base/(name+'-chain'),suffix,True);check(x.shape==y.shape,name+' chain shape '+suffix);err.append(float(np.max(np.abs(x-y))))
    source_meta=meta(base/source);target_meta=meta(base/(name+'-chain'))
    check(source_meta['divergences']==target_meta['divergences'] and abs(source_meta['epsilon']-target_meta['epsilon'])<2e-7 and abs(source_meta['acceptance']-target_meta['acceptance'])<2e-7,name+' adaptive summary replay')
    replay[name]=max(err);check(replay[name]<2e-7,name+' canonical adaptive state/predictive replay')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for j,family in enumerate(['force','time','particle']):
    r=convergence[family]['rows'][:-1];x=[v['resolution'] for v in r]
    for row,key,label in [(0,'field_error','Relative observed-band density error'),(1,'gradient_error','Relative projected posterior gradient error')]:
        titles={'force':'Force refinement\nN=8, 16 steps; reference M=64','time':'Time refinement\nN=8, M=32; reference 64 steps','particle':'Particle refinement\nM=32, 16 steps; reference N=32'}
        axes[row,j].loglog(x,[v[key] for v in r],'o-');axes[row,j].set(xlabel={'force':'Force mesh M','time':'Time steps','particle':'Particle lattice N'}[family],ylabel=label,title=titles[family]);axes[row,j].grid(alpha=.2)
        axes[row,j].set_xticks(x);axes[row,j].set_xticklabels([str(v) for v in x]);axes[row,j].xaxis.set_minor_locator(NullLocator())
fig.suptitle('Matched Fourier ICs; one factor varied at a time; each family uses its own finest reference')
fig.savefig(a.output/'convergence.png',dpi=170);fig.savefig(a.output/'convergence.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,family in zip(axes,['force','time','particle']):
    for r in convergence[family]['rows'][:-1]:ax.plot(2*np.pi*np.sqrt(shells),100*np.array(r['power_fraction']),'o-',label=str(r['resolution']))
    ax.axhline(0,color='k',lw=.6);ax.set(title=family.capitalize()+' refinement',xlabel='k [1/L]',ylabel='Power difference from finest [%]');ax.legend(title=convergence[family]['variable'],fontsize=8)
fig.savefig(a.output/'resolution_power.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,(key,result) in zip(axes,fd_results.items()):
    for probe,r in result.items():ax.loglog(r['h'],np.maximum(r['relative_error'],1e-16),'o-',label='field' if probe=='0' else 'field + scalars')
    ax.axhline(2e-5,color='gray',ls='--');ax.set(title='N/M/steps = '+key,xlabel='central-difference h',ylabel='Scaled directional error');ax.legend()
fig.savefig(a.output/'gradients.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,4,figsize=(13,6.2),layout='constrained')
for row,mesh,draws,t in [(0,n,initial,ti),(1,m,banded,true_band)]:
    cubes=draws.reshape(-1,mesh,mesh,mesh);mean=cubes.mean(0);sd=cubes.std(0);true=t.reshape(mesh,mesh,mesh)
    residual=true-mean if row==0 else np.fft.irfftn(data,s=(m,m,m),norm='forward')-mean
    images=[true[:,:,0],mean[:,:,0],sd[:,:,0],residual[:,:,0]];lim=max(np.abs(images[0]).max(),np.abs(images[1]).max())
    for j,(ax,img) in enumerate(zip(axes[row],images)):
        scale=np.abs(img).max() if j==3 else lim
        im=ax.imshow(img.T,origin='lower',extent=(0,1,0,1),cmap='viridis' if j==2 else 'RdBu_r',vmin=0 if j==2 else -scale,vmax=sd.max() if j==2 else scale)
        ax.set_title(['Mock truth','Posterior mean','Latent posterior SD','Truth − mean' if row==0 else 'Observed band data − mean'][j]);ax.set_xlabel('x / L');fig.colorbar(im,ax=ax,shrink=.75)
    axes[row,0].set_ylabel(('Initial density at a=1' if row==0 else 'Final observed-band density')+'\ny / L')
fig.suptitle(f'Joint posterior: {cells} white coordinates, 18 active IC modes; z=0 slices')
fig.savefig(a.output/'reconstruction.png',dpi=170);fig.savefig(a.output/'reconstruction.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,mesh,draws,t in [(axes[0],n,initial,ti),(axes[1],m,matter,tm)]:
    qq,ww=geometry(mesh);ss=[1,2] if mesh==n else shells;fk=fft(draws,mesh);tk=fft(t,mesh)
    def ps(f):return np.stack([np.sum(np.abs(f[...,qq==s])**2*ww[qq==s],axis=-1)/ww[qq==s].sum() for s in ss],axis=-1)
    lo,med,hi=np.quantile(ps(fk),[.025,.5,.975],axis=0);x=2*np.pi*np.sqrt(ss)
    ax.fill_between(x,lo,hi,alpha=.3,label='latent posterior 95%');ax.plot(x,med,'o-');ax.plot(x,ps(tk)[0],'k*--',label='mock truth')
    if mesh==m:
        low,high=np.quantile(ps(fk+noise_fft*observed),[.025,.975],axis=0);ax.fill_between(x,low,high,color='gray',alpha=.18,label='noisy prediction 95%');ax.plot(x,ps(data[None,...])[0],'kx',label='observed data')
    ax.set(title='Initial (two supported shells)' if mesh==n else 'Final observed shells',xlabel='k [1/L]',ylabel='P(k) [L³]',yscale='log');ax.legend(fontsize=8)
fig.savefig(a.output/'inference_power.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for i in range(4):
    axes[0].plot(z[i,:,0],lw=.5,alpha=.6);axes[1].plot(samples[i,:,-2],lw=.5,alpha=.6);axes[2].plot(samples[i,:,-1],lw=.5,alpha=.6)
for ax,title in zip(axes,['Active mode 0','log amplitude','log noise power']):ax.set(title=title,xlabel='retained draw')
fig.savefig(a.output/'traces.png',dpi=170);plt.close(fig)
summary=dict(passed=all(x['passed'] for x in gates),gates=gates,convergence=convergence,finite_differences=fd_results,inference=inference,replay_errors=replay,
    limits=['One weakly nonlinear Fourier realization; separate finest references are not an exact continuum solution.',
            'Larger inference is a separate same-model mock, not resolution-bias calibration or repeated-mock coverage.',
            'CIC is piecewise differentiable; no universal convergence order is assumed.',
            'Only observed k<15 and IC k<10 are tested; no high-cutoff/shell-crossing or production scaling claim.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer=csv.writer(file,lineterminator="\n");writer.writerow(['family','case','field_error','gradient_error','noise_rms_units'])
    for family,r in convergence.items():
        for row in r['rows']:writer.writerow([family,row['key'],row['field_error'],row['gradient_error'],row['noise_rms_units']])
print(json.dumps(dict(passed=summary['passed'],failed=[g['check'] for g in gates if not g['passed']],inference=inference,replay_errors=replay),indent=2))
if not summary['passed']:raise SystemExit(1)
