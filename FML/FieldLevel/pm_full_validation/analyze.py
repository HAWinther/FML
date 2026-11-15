"""Diagnostics and posterior predictions for a bounded full-field PM mock."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import basis

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ensemble',choices=['both','fixed'],default='both');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);gates=[]
def check(condition,label):gates.append(dict(check=label,passed=bool(condition)))
def table(prefix,suffix):return np.atleast_2d(np.loadtxt(str(prefix)+'.'+suffix+'.csv',delimiter=',',skiprows=1))
def meta(prefix):
    with Path(str(prefix)+'.meta.csv').open() as f:return {k:float(v) for k,v in next(csv.DictReader(f)).items()}
def diagnostics(path):
    with path.open() as f:return {r['feature']:{k:float(v) if k not in ['feature','status'] else v for k,v in r.items()} for r in csv.DictReader(f)}
def truth(prefix):return [np.fromstring(x,sep=',') for x in Path(str(prefix)+'.truth.csv').read_text().splitlines()]
# Canonical x,y,z grids; numpy's forward-normalized rFFT matches the model convention.
freq=np.fft.fftfreq(8)*8;fz=np.fft.rfftfreq(8)*8
nx,ny,nz=np.meshgrid(freq,freq,fz,indexing='ij');q=nx*nx+ny*ny+nz*nz
mask=(q>0)&(2*np.pi*np.sqrt(q)<15);weight=np.broadcast_to(np.where((fz==0)|(fz==4),1.,2.),q.shape)
shells=np.unique(q[mask]).astype(int)
def fft(x,n):return np.fft.rfftn(x.reshape((-1,n,n,n)),axes=(-3,-2,-1),norm='forward')
def band(x):return np.fft.irfftn(fft(x,8)*mask,s=(8,8,8),axes=(-3,-2,-1),norm='forward').reshape(-1,512)
def power(x,n):
    fk=fft(x,n);f=np.fft.fftfreq(n)*n;z=np.fft.rfftfreq(n)*n
    xx,yy,zz=np.meshgrid(f,f,z,indexing='ij');qq=xx*xx+yy*yy+zz*zz;w=np.broadcast_to(np.where((z==0)|(z==n//2),1.,2.),qq.shape)
    valid=shells if n==8 else np.array([1])
    return np.stack([np.sum(np.abs(fk[...,qq==s])**2*w[qq==s],axis=-1)/w[qq==s].sum() for s in valid],axis=-1)
active,null=basis();results={};cache={}
families=['fixed','joint'] if a.ensemble=='both' else ['fixed']
for family in families:
    joint=family=='joint'
    prefixes=[a.input/'science'/f'{family}{i}' for i in range(4)]
    states=[table(x,'states') for x in prefixes];potentials=[table(x,'potential') for x in prefixes]
    pred=[table(x,'predictive') for x in prefixes];metadata=[meta(x) for x in prefixes];diag=diagnostics(a.input/f'science/{family}.diagnostics.csv')
    lengths=[len(x) for x in states];check(len(set(lengths))==1 and min(lengths)>=128,f'{family}: equal complete chains')
    for i,(s,u,d,m) in enumerate(zip(states,potentials,pred,metadata)):
        check(s.shape[1]==67 and np.all(np.isfinite(s)) and np.array_equal(s[:,0],np.arange(len(s))),f'{family}{i}: finite consecutive 64-white/two-scalar states')
        check(len(s)==int(m['draws']) and np.array_equal(s[:,0],u[:,0]) and np.all(np.isfinite(u)),f'{family}{i}: potential and draw counts')
        check(np.array_equal(d[:,0],np.arange(0,len(s),int(m['stride']))) and d.shape[1]==577 and np.all(np.isfinite(d)),f'{family}{i}: complete predictive schedule')
        check(m['divergences']==0 and .5<m['acceptance']<=1 and m['epsilon']>0,f'{family}{i}: no post-warmup divergences and usable acceptance')
        check(np.all(np.abs(s[:,65])<=1) and np.all((s[:,66]>=-14)&(s[:,66]<=-3)),f'{family}{i}: scalar support')
        if not joint:check(np.max(np.abs(s[:,65]))==0 and np.max(np.abs(s[:,66]-np.log(.001)))<1e-14,f'{family}{i}: fixed scalar parameters')
    for name,d in diag.items():
        check(d['status']=='ok' and d['rhat']<1.05 and d['ess_bulk']>400 and d['ess_tail']>200,f'{family}: mixing {name}')
    total=sum(lengths);null_checks=[]
    for j in range(58):
        for name,expected,variance in [(f'null{j}',0.,1.),(f'null_square{j}',1.,2.)]:
            d=diag[name];se=np.sqrt(variance/max(1.,min(d['ess_bulk'],total)))
            null_checks.append(abs(d['mean']-expected)/se);check(abs(d['mean']-expected)<5*se,f'{family}: analytic prior moment {name}')
    d=diag['null_norm'];norm_se=np.sqrt(116/max(1.,min(d['ess_bulk'],total)))
    check(abs(d['mean']-58)<5*norm_se,f'{family}: chi-square-58 mean')
    tw,ti,tm=truth(prefixes[0]);check(tw.shape==(64,) and ti.shape==(64,) and tm.shape==(512,),f'{family}: canonical truth sizes')
    check(np.std(tw@active)>.1,f'{family}: mock excites multiple active directions')
    for prefix in prefixes:
        check(all(np.array_equal(x,y) for x,y in zip(truth(prefix),[tw,ti,tm])),f'{family}: identical mock truth {prefix.name}')
    states=np.stack(states);white=states[:,:,1:65];projections=white@active
    density=np.concatenate(pred);params=np.concatenate([s[d[:,0].astype(int),65:67] for s,d in zip(states,pred)])
    initial=density[:,1:65];matter=density[:,65:];matter_band=band(matter);truth_band=band(tm)[0]
    # A direct DFT checks whitening/power normalization and the six-direction support
    # against the saved public reconstruction for every predictive draw.
    selected_white=np.concatenate([s[d[:,0].astype(int),1:65] for s,d in zip(states,pred)])
    k4=np.fft.fftfreq(4)*4;z4=np.fft.rfftfreq(4)*4
    aa,bb,cc=np.meshgrid(k4,k4,z4,indexing='ij');supported=(aa*aa+bb*bb+cc*cc)==1
    expected=fft(selected_white,4)*np.sqrt(64*.02*np.exp(params[:,0]))[:,None,None,None]*supported
    dft_error=float(np.max(np.abs(fft(initial,4)-expected)))
    check(dft_error<2e-12,f'{family}: direct-DFT initial whitening and six-direction support')
    data_raw=np.fromstring(Path(str(prefixes[0])+'.data.csv').read_text(),sep=',').reshape(-1,2)
    data=(data_raw[:,0]+1j*data_raw[:,1]).reshape(8,8,5)
    check(np.max(np.abs(data[~mask]))==0,f'{family}: observed data support')
    check(all(Path(str(prefix)+'.data.csv').read_bytes()==Path(str(prefixes[0])+'.data.csv').read_bytes() for prefix in prefixes),f'{family}: identical observed data across chains')
    # Real-space mock noise -> Fourier noise variance exp(log_noise)/V per complex mode.
    rng=np.random.default_rng(551);noise=rng.normal(size=(len(matter),8,8,8))*np.sqrt(np.exp(params[:,1])*512)[:,None,None,None]
    noise_fourier=np.fft.rfftn(noise,axes=(-3,-2,-1),norm='forward')
    prediction=fft(matter,8);variance=np.exp(params[:,1])
    chi_observed=np.sum(weight[mask]*np.abs(prediction[:,mask]-data[mask])**2,axis=1)/variance
    chi_replicate=np.sum(weight[mask]*np.abs(noise_fourier[:,mask])**2,axis=1)/variance
    ppc=float(np.mean(chi_replicate>=chi_observed));dof=float(np.sum(weight[mask]))
    check(np.all(np.isfinite(chi_observed)) and 0<ppc<1,f'{family}: nondegenerate posterior predictive chi-square check')
    check(abs(chi_replicate.mean()-dof)<5*np.sqrt(2*dof/len(matter)),f'{family}: replicated noise normalization')
    # Latent density uncertainty is distinct from a noisy posterior prediction.
    initial_error=float(np.mean((initial.mean(0)-ti)**2));matter_error=float(np.mean((matter_band.mean(0)-truth_band)**2))
    def correlation(x,y):return float(np.corrcoef(x,y)[0,1])
    repeat=sum(int(np.sum(np.all(np.diff(s[:,1:],axis=0)==0,axis=1))) for s in states)
    results[family]=dict(draws_per_chain=lengths[0],chains=4,max_rhat=max(d['rhat'] for d in diag.values()),
        min_bulk_ess=min(d['ess_bulk'] for d in diag.values()),min_tail_ess=min(d['ess_tail'] for d in diag.values()),
        max_white_rhat=max(diag[f'w{i}']['rhat'] for i in range(64)),metadata=metadata,
        max_null_moment_standard_errors=max(null_checks),null_norm_mean=diag['null_norm']['mean'],
        posterior_predictive_p=ppc,observed_degrees_of_freedom=dof,mean_chi_observed=float(chi_observed.mean()),mean_chi_replicate=float(chi_replicate.mean()),
        initial_correlation=correlation(initial.mean(0),ti),final_band_correlation=correlation(matter_band.mean(0),truth_band),
        initial_mse=initial_error,final_band_mse=matter_error,repeated_states=repeat,initial_dft_error=dft_error,
        scalar_summary={name:dict(mean=float(states[:,:,index].mean()),interval95=np.quantile(states[:,:,index],[.025,.975]).tolist()) for name,index in [('log_amplitude',65),('log_noise',66)]})
    cache[family]=dict(states=states,projections=projections,initial=initial,matter=matter,matter_band=matter_band,
        truth_initial=ti,truth_matter=tm,truth_band=truth_band,data=data,noise=noise_fourier,params=params,chi_observed=chi_observed,chi_replicate=chi_replicate)
# Export pointwise intervals, without treating correlated cells as independent mocks.
with (a.output/'density_intervals.csv').open('w') as file:
    writer=csv.writer(file);writer.writerow(['ensemble','mesh','cell','truth','mean','sd','q025','q975'])
    for family in families:
        for n,key,tkey in [(4,'initial','truth_initial'),(8,'matter_band','truth_band')]:
            samples=cache[family][key];lo,hi=np.quantile(samples,[.025,.975],axis=0)
            for i,(t,m,s,l,h) in enumerate(zip(cache[family][tkey],samples.mean(0),samples.std(0),lo,hi)):
                writer.writerow([family,n,i,t,m,s,l,h])
# All optional variants compare the complete retained traces, predictions and truth to serial.
replay={}
base=a.input/'replay'
for name in ['mpi1','mpi2','mpi4','pencil4','pencil4-thread1','sanitizer']:
    if not (base/f'{name}-0.meta.csv').exists():continue
    errors=[]
    for joint in [0,1]:
        source=base/f'serial-{joint}';target=base/f'{name}-{joint}'
        for suffix in ['states','potential','predictive']:
            x=table(source,suffix);y=table(target,suffix);check(x.shape==y.shape,f'{name}/{joint}: {suffix} shape')
            errors.append(float(np.max(np.abs(x-y))))
        errors.extend(float(np.max(np.abs(x-y))) for x,y in zip(truth(source),truth(target)))
        m=meta(target);check(m['divergences']==0 and m['acceptance']>.5,f'{name}/{joint}: sampler summary')
    replay[name]=max(errors);check(replay[name]<2e-8,f'{name}: canonical retained-state/predictive replay')
# Figures: show joint latent reconstruction, sampled power and discrepancies.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
display='joint' if 'joint' in cache else 'fixed';c=cache[display];fig,axes=plt.subplots(2,4,figsize=(13,6.2),layout='constrained')
for row,n,key,tkey in [(0,4,'initial','truth_initial'),(1,8,'matter_band','truth_band')]:
    draws=c[key].reshape(-1,n,n,n);mean=draws.mean(0);sd=draws.std(0);truth_field=c[tkey].reshape(n,n,n)
    residual=truth_field-mean if row==0 else np.fft.irfftn(c['data'],s=(8,8,8),norm='forward')-mean
    images=[truth_field[:,:,0],mean[:,:,0],sd[:,:,0],residual[:,:,0]];v=max(np.abs(images[0]).max(),np.abs(images[1]).max())
    titles=['Mock truth','Posterior mean','Latent posterior SD','Truth − mean' if row==0 else 'Band-limited data − mean']
    for j,(ax,img,title) in enumerate(zip(axes[row],images,titles)):
        lim=max(np.abs(img).max(),1e-12) if j==3 else v
        im=ax.imshow(img.T,origin='lower',extent=(0,1,0,1),cmap='viridis' if j==2 else 'RdBu_r',vmin=0 if j==2 else -lim,vmax=sd.max() if j==2 else lim)
        ax.set_title(title);ax.set_xlabel('x / L');fig.colorbar(im,ax=ax,shrink=.75)
    axes[row,0].set_ylabel(('Initial linear density' if row==0 else 'Final observed-band density')+'\ny / L')
fig.suptitle(display.capitalize()+' full-field PM posterior — z = 0 slices; latent uncertainty excludes measurement noise')
fig.savefig(a.output/'density.png',dpi=170);fig.savefig(a.output/'density.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for ax,n,key,tkey in [(axes[0],4,'initial','truth_initial'),(axes[1],8,'matter','truth_matter')]:
    x=np.array([2*np.pi]) if n==4 else 2*np.pi*np.sqrt(shells)
    for family,color in [(name,color) for name,color in [('fixed','C0'),('joint','C1')] if name in cache]:
        powers=power(cache[family][key],n);low,median,high=np.quantile(powers,[.025,.5,.975],axis=0)
        if n==4:ax.errorbar(x,median,yerr=np.stack([median-low,high-median]),fmt='o',capsize=5,color=color,label=family+' latent 95%')
        else:
            ax.fill_between(x,low,high,color=color,alpha=.22);ax.plot(x,median,'o-',color=color,label=family+' latent 95%')
    ax.plot(x,power(c[tkey],n)[0],'k*--',label='mock truth')
    if n==8:
        noisy=c['data'][None,...]+np.zeros_like(c['noise']);rep=fft(c['matter'],8)+c['noise']*mask
        def mode_power(fk):return np.stack([np.sum(np.abs(fk[:,q==s])**2*weight[q==s],axis=-1)/weight[q==s].sum() for s in shells],axis=-1)
        lo,hi=np.quantile(mode_power(rep),[.025,.975],axis=0);ax.fill_between(x,lo,hi,color='gray',alpha=.15,label=display+' noisy prediction 95%')
        ax.plot(x,mode_power(noisy[:1])[0],'kx',label='noisy observed data')
    ax.set(xlabel='k [1/L]',ylabel='shell-averaged |δ(k)|² L³',yscale='log',title='Initial density (one supported shell)' if n==4 else 'Final density in observed shells');ax.legend(fontsize=8)
fig.savefig(a.output/'power.png',dpi=170);fig.savefig(a.output/'power.pdf');plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
for j,ax in enumerate(axes.flat):
    values=c['projections'][:,:,2*j] if j<3 else (c['states'][:,:,1:65]@null)[:,:,0] if j==3 else c['states'][:,:,65+j-4]
    for i in range(4):ax.plot(values[i],alpha=.65,lw=.55,label=f'chain {i}')
    ax.set_title(['Active x cosine','Active y cosine','Active z cosine','Null coordinate 0 (exact N(0,1))','log amplitude','log noise power'][j]);ax.set_xlabel('retained draw')
    if j==0:ax.legend(ncol=4,fontsize=8)
fig.suptitle(display.capitalize()+' PM chains — warmup excluded, every retained draw shown');fig.savefig(a.output/'traces.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for family,ax in zip(families,axes):
    d=cache[family];ax.hist(d['chi_replicate'],bins=30,density=True,alpha=.5,label='replicated data vs latent draw')
    ax.hist(d['chi_observed'],bins=30,density=True,alpha=.5,label='observed data vs latent draw')
    ax.set(title=f'{family}: predictive tail fraction = {results[family]["posterior_predictive_p"]:.3f}',xlabel='χ² discrepancy',ylabel='density');ax.legend(fontsize=8)
if len(families)==1:axes[1].set_visible(False)
fig.savefig(a.output/'predictive.png',dpi=170);plt.close(fig)
if 'joint' in cache:
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,index,label,mean,limits in [(axes[0],65,'log amplitude',0.,(-1,1)),(axes[1],66,'log noise power',np.log(.001),(-9,-5))]:
        values=c['states'][:,:,index].ravel();ax.hist(values,bins=35,density=True,alpha=.55,label='joint posterior')
        grid=np.linspace(-1,1,600) if index==65 else np.linspace(-14,-3,1200)
        prior=np.exp(-.5*((grid-mean)/.5)**2);prior/=np.trapz(prior,grid)
        ax.plot(grid,prior,'k--',label='truncated Gaussian prior');ax.axvline(mean,color='C3',label='injected value')
        ax.set(xlabel=label,ylabel='density',xlim=limits);ax.legend(fontsize=8)
    fig.savefig(a.output/'parameters.png',dpi=170);plt.close(fig)
summary=dict(passed=all(x['passed'] for x in gates),gates=gates,results=results,replay_errors=replay,
    limits=['One same-model mock, N=4/M=8; six active IC directions and 58 exact-prior null directions.',
            'Diagnostics are finite-chain checks, not convergence proofs or repeated-mock coverage.',
            'Predictive draws use every eighth retained sample; diagnostics use every retained sample.',
            'Latent density and noisy posterior predictive intervals are distinct; final density comparisons use observed Fourier support.'])
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(passed=summary['passed'],failed=[g['check'] for g in gates if not g['passed']],results=results,replay_errors=replay),indent=2))
if not summary['passed']:raise SystemExit(1)
