"""Independent NumPy CIC/FFT diagnostic, with dynamics and HMC removed."""
import argparse
import csv
from pathlib import Path
import numpy as np
from prepare import waves

def setup(n,coefficients):
    q=np.indices((n,n,n)).reshape(3,-1).T/n;psi=np.zeros_like(q)
    for j,k in enumerate(waves()):
        k=np.array(k);phase=2*np.pi*(q@k);amplitude=np.sqrt(2)*(-coefficients[2*j]*np.sin(phase)+coefficients[2*j+1]*np.cos(phase))
        psi+=amplitude[:,None]*k/(2*np.pi*(k@k))
    return q,psi

def nodes(x,m):
    u=(x%1)*m;floor=np.floor(u).astype(int);frac=u-floor
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                b=np.array([i,j,k]);c=(floor+b)%m;index=(c[:,0]*m+c[:,1])*m+c[:,2]
                w=np.prod(np.where(b,frac,1-frac),axis=1);yield index,w

def density(x,m):
    grid=np.zeros(m**3)
    for index,w in nodes(x,m):grid+=np.bincount(index,weights=w,minlength=m**3)
    return (grid*m**3/len(x)-1).reshape(m,m,m)

def force(x,m):
    rho=np.fft.fftn(density(x,m),norm='forward');f=np.fft.fftfreq(m)*m;ks=np.meshgrid(f,f,f,indexing='ij');k2=sum(k*k for k in ks);den=np.where(k2>0,k2,1)
    grid=[]
    for k in ks:
        kernel=1j*k/(2*np.pi*den);kernel[(k2==0)|(np.abs(k)==m//2)]=0
        grid.append(np.fft.ifftn(rho*kernel,norm='forward').real.ravel())
    out=np.zeros_like(x)
    for index,w in nodes(x,m):
        for d in range(3):out[:,d]+=w*grid[d][index]
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    coefficients=np.loadtxt(a.initial,delimiter=',');rows=[]
    for n,m in [(8,m) for m in [8,16,32,64]]+[(n,32) for n in [4,16,32,64]]+[(32,16),(64,16),(64,8)]:
        q,psi=setup(n,coefficients)
        for amplitude in [1e-3,1e-4]:
            f=force(q+amplitude*psi,m);response=float(np.sum(f*psi)/(amplitude*np.sum(psi*psi)))
            residual=float(np.linalg.norm(f-amplitude*psi)/np.linalg.norm(amplitude*psi));rows.append([n,m,amplitude,response,residual])
            print(n,m,amplitude,response,residual,flush=True)
    with (a.output/'linear_force.csv').open('w') as file:
        writer=csv.writer(file,lineterminator="\n");writer.writerow(['n','m','amplitude','longitudinal_response','relative_force_error']);writer.writerows(rows)
