"""Create full-field diagnostics features; no samples are dropped or thinned here."""
import argparse
from pathlib import Path
import numpy as np

def basis():
    xyz=np.indices((4,4,4)).reshape(3,-1)
    active=np.stack([f(2*np.pi*xyz[d]/4)*np.sqrt(2/64) for d in range(3) for f in (np.cos,np.sin)],axis=1)
    assert np.max(np.abs(active.T@active-np.eye(6)))<1e-14
    complete=np.linalg.qr(active,mode='complete')[0]
    assert np.max(np.abs(complete.T@complete-np.eye(64)))<1e-14
    return active,complete[:,6:]

def prepare(prefix,joint):
    data=np.atleast_1d(np.genfromtxt(str(prefix)+'.states.csv',delimiter=',',names=True))
    potential=np.atleast_1d(np.genfromtxt(str(prefix)+'.potential.csv',delimiter=',',names=True))
    n=len(data);assert n==len(potential) and np.array_equal(data['draw'],np.arange(n)) and np.array_equal(potential['draw'],data['draw'])
    white=np.stack([data[f'w{i}'] for i in range(64)],axis=1);active,null=basis()
    z=white@active;u=white@null
    values=[white,z,z*z,u,u*u,np.sum(u*u,axis=1)[:,None],np.sum(white*white,axis=1)[:,None],potential['potential'][:,None]]
    names=[*[f'w{i}' for i in range(64)],*[f'active{i}' for i in range(6)],*[f'active_square{i}' for i in range(6)],
           *[f'null{i}' for i in range(58)],*[f'null_square{i}' for i in range(58)],'null_norm','white_norm','potential']
    if joint:
        values += [data['log_amplitude'][:,None],data['log_noise'][:,None]];names+=['log_amplitude','log_noise']
    x=np.column_stack([np.arange(n),*values]);assert np.all(np.isfinite(x))
    np.savetxt(str(prefix)+'.features.csv',x,delimiter=',',header=','.join(['draw',*names]),comments='',fmt='%.17g')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--joint',action='store_true');p.add_argument('prefix',nargs='+',type=Path);a=p.parse_args()
    for prefix in a.prefix:prepare(prefix,a.joint)
