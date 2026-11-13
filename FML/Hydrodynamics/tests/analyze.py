#!/usr/bin/env python3
"""Independent exact Euler cell averages, multidimensional convergence and MPI replay."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def load(path):
    m={k:float(v) for k,v in (line.split() for line in (path/'meta.tsv').read_text().splitlines())}
    D=int(m['dimension']);pieces=[]
    for p in sorted(path.glob('cells.rank*.tsv')):
        with p.open() as f:has_data=any(line.strip() and not line.lstrip().startswith('#') for line in f)
        if has_data:pieces.append(np.loadtxt(p,ndmin=2))
    x=np.concatenate(pieces) if pieces else np.empty((0,3*D+6));x=x[np.argsort(x[:,0])]
    count=int(np.prod([m[f'shape{d}'] for d in range(D)]))
    if x.shape!=(count,3*D+6) or not np.array_equal(x[:,0],np.arange(count)):raise ValueError(f'Incomplete grid: {path}')
    if not np.isfinite(x).all() or np.any(x[:,D+1]<=0) or np.any(x[:,2*D+2]<=0):raise ValueError(f'Nonphysical gas: {path}')
    return m,x


class Exact:
    """Independent pressure bisection in linear p, piecewise exact ideal-gas fans/shocks."""
    def __init__(self,left,right,g=1.4):
        self.left=np.array(left,float);self.right=np.array(right,float);self.g=g
        self.cl=np.sqrt(g*left[2]/left[0]);self.cr=np.sqrt(g*right[2]/right[0])
        self.vacuum=right[1]-left[1]>=2*(self.cl+self.cr)/(g-1)
        if self.vacuum:
            self.breaks=[left[1]-self.cl,left[1]+2*self.cl/(g-1),right[1]-2*self.cr/(g-1),right[1]+self.cr];return
        def wave(p,q,c):
            rho,u,pk=q
            if p>pk:return (p-pk)*np.sqrt(2/((g+1)*rho)/(p+(g-1)/(g+1)*pk))
            return 2*c/(g-1)*((p/pk)**((g-1)/(2*g))-1)
        lo,hi=0.,max(left[2],right[2])
        while wave(hi,left,self.cl)+wave(hi,right,self.cr)+right[1]-left[1]<0:hi*=2
        for _ in range(180):
            p=(lo+hi)/2
            if wave(p,left,self.cl)+wave(p,right,self.cr)+right[1]-left[1]>0:hi=p
            else:lo=p
        self.p=(lo+hi)/2;self.u=(left[1]+right[1]+wave(self.p,right,self.cr)-wave(self.p,left,self.cl))/2
        self.breaks=[self.u]
        for q,c,sign in [(self.left,self.cl,-1),(self.right,self.cr,1)]:
            if self.p>q[2]:self.breaks.append(q[1]+sign*c*np.sqrt((g+1)/(2*g)*self.p/q[2]+(g-1)/(2*g)))
            else:self.breaks.extend([q[1]+sign*c,self.u+sign*c*(self.p/q[2])**((g-1)/(2*g))])
    def sample(self,xi):
        rho=np.zeros_like(xi);vel=np.zeros_like(xi);pressure=np.zeros_like(xi);g=self.g
        def fan(mask,q,c,sign):
            v=2/(g+1)*(sign*c+(g-1)*q[1]/2+xi[mask]);cs=2/(g+1)*(c+sign*(g-1)*(q[1]-xi[mask])/2)
            rho[mask]=q[0]*(np.maximum(cs,0)/c)**(2/(g-1));vel[mask]=v;pressure[mask]=q[2]*(np.maximum(cs,0)/c)**(2*g/(g-1))
        if self.vacuum:
            l=xi<self.left[1]-self.cl;r=xi>self.right[1]+self.cr
            rho[l],vel[l],pressure[l]=self.left;rho[r],vel[r],pressure[r]=self.right
            fan((~l)&(xi<self.left[1]+2*self.cl/(g-1)),self.left,self.cl,1)
            fan((~r)&(xi>self.right[1]-2*self.cr/(g-1)),self.right,self.cr,-1)
            return np.column_stack([rho,vel,pressure])
        for q,c,side in [(self.left,self.cl,-1),(self.right,self.cr,1)]:
            region=xi<self.u if side==-1 else xi>=self.u
            if self.p>q[2]:
                speed=q[1]+side*c*np.sqrt((g+1)/(2*g)*self.p/q[2]+(g-1)/(2*g));outside=region&(xi<speed if side==-1 else xi>speed);star=region&~outside
                rho[outside],vel[outside],pressure[outside]=q;ratio=self.p/q[2];rho[star]=q[0]*(ratio+(g-1)/(g+1))/((g-1)/(g+1)*ratio+1);vel[star]=self.u;pressure[star]=self.p
            else:
                cs=c*(self.p/q[2])**((g-1)/(2*g));head=q[1]+side*c;tail=self.u+side*cs
                outside=region&(xi<head if side==-1 else xi>head);star=region&(xi>tail if side==-1 else xi<tail);inside=region&~outside&~star
                rho[outside],vel[outside],pressure[outside]=q;rho[star]=q[0]*(self.p/q[2])**(1/g);vel[star]=self.u;pressure[star]=self.p;fan(inside,q,c,-side)
        return np.column_stack([rho,vel,pressure])
    def cell_average(self,centers,dx,t):
        nodes,weights=np.polynomial.legendre.leggauss(8);answer=[]
        for center in centers:
            low,high=center-dx/2,center+dx/2;cuts=sorted([low,high]+[.5+t*b for b in self.breaks if low<.5+t*b<high]);integral=np.zeros(3)
            for a,b in zip(cuts[:-1],cuts[1:]):
                xx=(a+b)/2+(b-a)/2*nodes;q=self.sample((xx-.5)/t);cons=np.column_stack([q[:,0],q[:,0]*q[:,1],q[:,2]/(self.g-1)+.5*q[:,0]*q[:,1]**2]);integral+=(b-a)/2*(weights@cons)
            avg=integral/dx
            answer.append([avg[0],avg[1]/avg[0] if avg[0]>0 else 0,(self.g-1)*(avg[2]-.5*avg[1]**2/avg[0]) if avg[0]>0 else 0])
        return np.array(answer)


STATES={'sod':((1,0,1),(.125,0,.1)),'lax':((.445,.698,3.528),(.5,0,.571)),'strong':((1,0,1000),(1,0,.01)),'rarefaction':((1,-2,.4),(1,2,.4)),'vacuum':((1,-8,1),(1,8,1))}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    report={'smooth':{},'riemann':{},'conservation':{},'replay':{}};cases={}
    for mode,ns in [('advect1d',(64,128,256)),('advect2d',(24,48)),('advect3d',(16,32)),('sound',(64,128))]:
        rows=[]
        for n in ns:
            m,x=load(args.input/f'{mode}-{n}');cases[f'{mode}-{n}']=(m,x);rows.append(m['density_l1_over_amplitude'])
        if not np.all(np.diff(rows)<0):raise ValueError(f'No smooth convergence: {mode}')
        report['smooth'][mode]={'n':list(ns),'density_l1_over_amplitude':rows,'observed_order':[float(np.log(rows[i]/rows[i+1])/np.log(ns[i+1]/ns[i])) for i in range(len(ns)-1)]}
    for mode,ns in [('sod',(128,256,512)),('lax',(256,)),('strong',(256,)),('rarefaction',(256,)),('vacuum',(256,))]:
        for n in ns:
            name=f'{mode}-{n}';m,x=load(args.input/name);cases[name]=(m,x);exact=Exact(*STATES[mode],m['gamma']);ref=exact.cell_average(x[:,1],1/n,m['time']);gas=ref[:,0]>1e-12
            errors=[float(np.mean(np.abs(x[:,2]-ref[:,0]))),float(np.mean(np.abs(x[gas,3]-ref[gas,1]))),float(np.mean(np.abs(x[:,4]-ref[:,2])))];report['riemann'][name]={'l1_density_velocity_pressure':errors,'minimum_density':m['minimum_density'],'minimum_pressure':m['minimum_pressure'],'riemann_fallbacks':m['riemann_fallbacks'],'low_order_stages':m['low_order_stages'],'retries':m['retries']}
    if not np.all(np.diff([report['riemann'][f'sod-{n}']['l1_density_velocity_pressure'] for n in (128,256,512)],axis=0)<0):raise ValueError('Sod did not refine in all three quantities')
    if report['riemann']['vacuum-256']['riemann_fallbacks']<=0:raise ValueError('Vacuum test missed guarded flux fallback')
    for mode,n in [('blast2d',64),('blast3d',16)]:cases[f'{mode}-{n}']=load(args.input/f'{mode}-{n}')
    for name,(m,x) in cases.items():
        if name.startswith(('advect','sound','blast')):
            de=abs(m['final_energy']/m['initial_energy']-1);dm=abs(m['final_mass']/m['initial_mass']-1)
            report['conservation'][name]={'mass_relative_error':dm,'energy_relative_error':de}
            if max(de,dm)>2e-11 or m['energy_repair']!=0:raise ValueError(f'Conservation failure: {name}')
    for name,reference in [('mpi1-advect2d-24','advect2d-24'),('mpi2-advect2d-24','advect2d-24'),('mpi4-advect2d-24','advect2d-24'),('mpi2-sod-128','sod-128'),('mpi2-blast3d-16','blast3d-16')]:
        if (args.input/name).exists():
            m,x=load(args.input/name);ref=cases[reference][1];err=float(np.max(np.abs(x-ref)));report['replay'][name]=err
            if err>2e-9:raise ValueError(f'MPI/thread replay mismatch: {name}')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');xx=np.linspace(0,1,1500);exact=Exact(*STATES['sod']);qq=exact.sample((xx-.5)/.2)
    for col,label in enumerate(('Density','Velocity','Pressure')):
        axes[col].plot(xx,qq[:,col],'k-',label='Exact Euler solution')
        for n,color in [(128,'#aaa'),(256,'#538bb3'),(512,'#c2732b')]:m,x=cases[f'sod-{n}'];axes[col].plot(x[:,1],x[:,2+col],'.-',lw=.7,ms=2,color=color,label=f'FV N={n}')
        axes[col].set(xlabel='x',ylabel=label,xlim=(.1,.95));axes[col].grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Sod shock tube: second-order HLLC at t=0.2; errors use exact conserved cell averages')
    fig.savefig(args.output/'sod.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'sod.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for mode,row in report['smooth'].items():axes[0].loglog(row['n'],row['density_l1_over_amplitude'],'o-',label=mode)
    axes[0].set_xticks([16,32,64,128,256],labels=['16','32','64','128','256']);axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set(xlabel='N in x (other axes follow declared shapes)',ylabel='Density L1 / perturbation amplitude',title='Smooth multidimensional refinement');axes[0].legend(fontsize=8)
    for j,label in enumerate(('Density','Velocity','Pressure')):axes[1].loglog([128,256,512],[report['riemann'][f'sod-{n}']['l1_density_velocity_pressure'][j] for n in (128,256,512)],'o-',label=label)
    axes[1].set_xticks([128,256,512],labels=['128','256','512']);axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[1].set(xlabel='Cells',ylabel='Absolute L1 error',title='Sod vs independent exact cell averages');axes[1].legend()
    for name in ('advect3d-32','blast2d-64','blast3d-16'):
        h=np.loadtxt(args.input/name/'history.tsv');axes[2].plot(h[:,0],np.abs(h[:,2]/h[0,2]-1),label=name)
    axes[2].set(xlabel='Time',ylabel='Relative total-energy drift',title='Conservative periodic evolution');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(args.output/'convergence.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'convergence.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,mode in zip(axes.ravel(),('lax','strong','rarefaction','vacuum')):
        m,x=cases[f'{mode}-256'];ex=Exact(*STATES[mode]);q=ex.sample((xx-.5)/m['time']);ax.plot(xx,q[:,0],'k-',label='Exact density');ax.plot(x[:,1],x[:,2],'.-',ms=2,lw=.7,label='FV density');ax.set(title=mode,xlabel='x',ylabel='Density');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Additional Riemann problems, including generated vacuum (no density/pressure clipping)')
    fig.savefig(args.output/'riemann.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'riemann.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout='constrained');m,x=cases['blast2d-64'];shape=tuple(int(m[f'shape{d}']) for d in range(2))
    for ax,values,title in zip(axes,(x[:,3],x[:,6],np.hypot(x[:,4],x[:,5])),('Density','Pressure','Speed')):
        im=ax.imshow(values.reshape(shape).T,origin='lower',extent=(0,1,0,1));ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('64×64 periodic thermal blast — finite injection region, not Sedov calibration')
    fig.savefig(args.output/'blast.png',dpi=170,bbox_inches='tight');fig.savefig(args.output/'blast.pdf',bbox_inches='tight');plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
