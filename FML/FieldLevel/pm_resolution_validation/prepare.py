"""Diagnostics of every retained white coordinate and active/null projection."""
import argparse
from pathlib import Path
import numpy as np

def waves():
    return [(x,y,z) for x in range(-1,2) for y in range(-1,2) for z in range(-1,2)
            if 0<x*x+y*y+z*z<=2 and (x>0 or (x==0 and y>0) or (x==0 and y==0 and z>0))]

def basis(n):
    xyz=np.indices((n,n,n)).reshape(3,-1)
    active=np.stack([f(2*np.pi*np.dot(k,xyz)/n)*np.sqrt(2/n**3) for k in waves() for f in (np.cos,np.sin)],axis=1)
    assert active.shape==(n**3,18) and np.max(np.abs(active.T@active-np.eye(18)))<1e-13
    complete=np.linalg.qr(active,mode='complete')[0]
    assert np.max(np.abs(complete.T@complete-np.eye(n**3)))<1e-13
    return active,complete[:,18:]

def prepare(prefix):
    meta=np.genfromtxt(str(prefix)+'.meta.csv',delimiter=',',names=True);n=int(meta['n'])
    data=np.atleast_2d(np.loadtxt(str(prefix)+'.states.csv',delimiter=',',skiprows=1));potential=np.atleast_2d(np.loadtxt(str(prefix)+'.potential.csv',delimiter=',',skiprows=1))
    assert data.shape[1]==n**3+3 and len(data)==len(potential) and np.array_equal(data[:,0],np.arange(len(data))) and np.array_equal(potential[:,0],data[:,0])
    white=data[:,1:1+n**3];active,null=basis(n);z=white@active;u=white@null
    values=[white,z,z*z,u,u*u,np.sum(u*u,axis=1)[:,None],np.sum(white*white,axis=1)[:,None],potential[:,1:],data[:,-2:]]
    names=[*[f'w{i}' for i in range(n**3)],*[f'active{i}' for i in range(18)],*[f'active_square{i}' for i in range(18)],
           *[f'null{i}' for i in range(n**3-18)],*[f'null_square{i}' for i in range(n**3-18)],'null_norm','white_norm','potential','log_amplitude','log_noise']
    x=np.column_stack([np.arange(len(data)),*values]);assert np.all(np.isfinite(x))
    np.savetxt(str(prefix)+'.features.csv',x,delimiter=',',header=','.join(['draw',*names]),comments='',fmt='%.17g')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefix',nargs='+',type=Path);a=p.parse_args()
    for prefix in a.prefix:prepare(prefix)
