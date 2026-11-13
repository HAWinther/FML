#!/usr/bin/env python3
"""Independent exact Sod solution, convergence/ABI replay checks, and scientific figures."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def metadata(path):
    return {k:float(v) for k,v in (line.split() for line in (path/'meta.tsv').read_text().splitlines())}


def load(path):
    meta=metadata(path)
    rows=np.concatenate([np.loadtxt(p,ndmin=2) for p in sorted(path.glob('particles.rank*.tsv'))])
    rows=rows[np.argsort(rows[:,0])]
    expected=int(meta['particles'])
    if rows.shape!=(expected,2*int(meta['dimension'])+11) or not np.array_equal(rows[:,0],np.arange(expected)):
        raise ValueError(f'Incomplete particle catalogue: {path}')
    D=int(meta['dimension']);u=2*D+2;rho=2*D+4
    if not np.isfinite(rows).all() or np.any(rows[:,u]<=0) or np.any(rows[:,rho]<=0):
        raise ValueError('Nonphysical final gas state')
    return meta,rows


def sod(x,t,gamma=1.4):
    """Exact Euler Riemann solution; independent of SPH/kernel/deposition code."""
    left=(1.,0.,1.);right=(.125,0.,.1)
    def wave(p,state):
        rho,v,pk=state;c=np.sqrt(gamma*pk/rho)
        if p>pk:return (p-pk)*np.sqrt(2/((gamma+1)*rho)/(p+(gamma-1)/(gamma+1)*pk))
        return 2*c/(gamma-1)*((p/pk)**((gamma-1)/(2*gamma))-1)
    lo,hi=1e-12,10.
    for _ in range(100):
        p=(lo+hi)/2
        if wave(p,left)+wave(p,right)>0:hi=p
        else:lo=p
    pstar=(lo+hi)/2;ustar=.5*(wave(pstar,right)-wave(pstar,left))
    # The standard Sod states have a left rarefaction, contact and right shock.
    if not (.1<pstar<1):raise ValueError('Unexpected Riemann topology')
    cl=np.sqrt(gamma);cr=np.sqrt(gamma*.1/.125)
    rl=pstar**(1/gamma);rr=.125*(pstar/.1+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*pstar/.1+1)
    cstar=cl*pstar**((gamma-1)/(2*gamma))
    shock=cr*np.sqrt((gamma+1)/(2*gamma)*pstar/.1+(gamma-1)/(2*gamma))
    xi=(np.asarray(x)-.5)/t
    rho=np.empty_like(xi);v=np.empty_like(xi);pressure=np.empty_like(xi)
    for i,s in enumerate(xi):
        if s< -cl:rho[i],v[i],pressure[i]=left
        elif s<ustar-cstar:
            vel=2/(gamma+1)*(cl+s);sound=2/(gamma+1)*(cl-(gamma-1)*s/2)
            rho[i]=(sound/cl)**(2/(gamma-1));v[i]=vel;pressure[i]=(sound/cl)**(2*gamma/(gamma-1))
        elif s<ustar:rho[i],v[i],pressure[i]=rl,ustar,pstar
        elif s<shock:rho[i],v[i],pressure[i]=rr,ustar,pstar
        else:rho[i],v[i],pressure[i]=right
    return rho,v,pressure


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    report={'sound':{},'sod':{},'mpi_replay':{}}
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for name,ns in [('sound1d',(64,128,256)),('sound3d',(8,16))]:
        cases=[load(args.input/f'{name}-{n}') for n in ns]
        e=np.array([m['velocity_l1_over_amplitude'] for m,_ in cases]);er=np.array([m['density_l1_over_amplitude'] for m,_ in cases]);
        if not np.all(np.diff(e)<0) or not np.all(np.diff(er)<0):raise ValueError('Sound-wave refinement did not improve both observables')
        report['sound'][name]={'n':list(ns),'velocity_l1_over_amplitude':e.tolist(),'density_l1_over_amplitude':er.tolist(),'energy_error':[m['relative_energy_error'] for m,_ in cases]}
        axes[0].loglog(ns,e,'o-',label=name+' velocity');axes[0].loglog(ns,er,'s--',label=name+' density')
    axes[0].set(xlabel='Particles per axis N',ylabel='Mean absolute error / wave amplitude',title='Traveling sound waves, amplitude 10⁻⁴');axes[0].legend(fontsize=8)
    shockcases=[]
    for n in (225,450,900):
        m,p=load(args.input/f'sod-{n}');sel=(p[:,1]>.25)&(p[:,1]<.75);q=p[sel];exact=sod(q[:,1],m['time'],m['gamma']);w=q[:,3]/q[:,6];err=[float(np.sum(w*np.abs(q[:,j]-v))/w.sum()) for j,v in zip((6,2,7),exact)]
        report['sod'][str(n)]={'volume_weighted_l1_density_velocity_pressure':err,'relative_energy_error':m['relative_energy_error']};shockcases.append((m,p))
        axes[1].loglog(n,err[0],'o',color='#2463a6');axes[1].loglog(n,err[1],'s',color='#c56b24');axes[1].loglog(n,err[2],'^',color='#32855b')
    errors=np.array([report['sod'][str(n)]['volume_weighted_l1_density_velocity_pressure'] for n in (225,450,900)])
    if not np.all(np.diff(errors,axis=0)<0):raise ValueError('Sod refinement did not improve density, velocity and pressure')
    for i,label,color,marker in [(0,'Density','#2463a6','o'),(1,'Velocity','#c56b24','s'),(2,'Pressure','#32855b','^')]:axes[1].loglog([225,450,900],errors[:,i],marker+'-',color=color,label=label)
    axes[1].set(xlabel='Total particles',ylabel='Volume-weighted absolute L1 error',title='Sod vs independent exact Euler solution');axes[1].legend()
    for name in ('sod-900','blast3d-16'):
        history=np.loadtxt(args.input/name/'history.tsv');error=np.abs(history[:,4]/history[0,4]-1)
        axes[2].plot(history[:,0],error,label=name)
        if error.max()>1e-4:raise ValueError('Excessive bounded-fixture energy drift')
    axes[2].set(xlabel='Time',ylabel='Absolute relative gas-energy drift',title='Nonlinear conservation');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(args.output/'convergence.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'convergence.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');x=np.linspace(.25,.75,1200);exact=sod(x,.1)
    for col,(j,label) in enumerate([(6,'Density'),(2,'Velocity'),(7,'Pressure')]):
        axes[col].plot(x,exact[col],'k-',label='Exact Euler Riemann solution')
        for (m,p),color in zip(shockcases,('#aaaaaa','#5893ba','#c16b2b')):
            order=np.argsort(p[:,1]);q=p[order];sel=(q[:,1]>.25)&(q[:,1]<.75)
            axes[col].plot(q[sel,1],q[sel,j],'.-',ms=2,lw=.7,color=color,label=f'SPH N={int(m["n"])}')
        axes[col].set(xlim=(.25,.75),xlabel='x',ylabel=label,title=label);axes[col].grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Periodic double-Sod problem: interior window before boundary waves arrive, t=0.1')
    fig.savefig(args.output/'sod.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'sod.pdf',bbox_inches='tight');plt.close(fig)
    m,p=load(args.input/'blast3d-16');r=np.linalg.norm(p[:,1:4]-.5,axis=1);vr=np.sum(p[:,4:7]*(p[:,1:4]-.5),axis=1)/r
    history=np.loadtxt(args.input/'blast3d-16/history.tsv');report['blast']={'relative_energy_error':m['relative_energy_error'],'momentum_norm':m['final_momentum_norm'],'max_radial_velocity':float(vr.max()),'min_energy':float(p[:,8].min())}
    if not vr.max()>0:raise ValueError('Blast did not expand')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    bins=np.linspace(0,.5,21);radius=(bins[:-1]+bins[1:])/2
    for ax,values,label in zip(axes,(p[:,10],vr,p[:,8]),('Density','Radial velocity','Specific internal energy')):
        mean=np.array([values[(r>=a)&(r<b)].mean() if np.any((r>=a)&(r<b)) else np.nan for a,b in zip(bins[:-1],bins[1:])]);ax.plot(r,values,'.',ms=2,alpha=.15);ax.plot(radius,mean,'o-',ms=3);ax.set(xlim=(0,.5),xlabel='Radius',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle('Bounded 16³-particle thermal blast — conservation/positivity check, not Sedov calibration')
    fig.savefig(args.output/'blast.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'blast.pdf',bbox_inches='tight');plt.close(fig)
    for ranks in (1,2,4):
        path=args.input/f'mpi{ranks}-sound3d-8'
        if path.exists():
            m,p=load(path);_,ref=load(args.input/'sound3d-8');diff=float(np.max(np.abs(p-ref)));report['mpi_replay'][str(ranks)]=diff
            if diff>2e-9:raise ValueError('MPI/thread replay differs materially')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
