#!/usr/bin/env python3
"""Independent Philox/prior/noise reference for the bounded native truth-control test."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

BOUNDS=np.array([[.5,1.5],[0,4],[-4,4],[-4,4],[-100,100]])

def words(seed, cell, purpose):
    # Standalone Random123 Philox4x32-10 definition; not an import of native code.
    mask=2**32-1
    c=[cell & mask,cell>>32,0,0]
    key=[(seed & mask)^purpose,seed>>32]
    for _ in range(10):
        a=0xD2511F53*c[0];b=0xCD9E8D57*c[2]
        c=[((b>>32)^c[1]^key[0])&mask,b&mask,((a>>32)^c[3]^key[1])&mask,a&mask]
        key=[(key[0]+0x9E3779B9)&mask,(key[1]+0xBB67AE85)&mask]
    return c

def uniform(a,b):
    return (((a<<20)|(b>>12))+.5)*2**-52

def key(mid,role,raw):
    return (0x54<<56)|(role<<48)|(mid<<32)|raw

def scalars(seed):
    return np.array([lo+(hi-lo)*uniform(*words(seed,j,0x53424353)[:2]) for j,(lo,hi) in enumerate(BOUNDS)])

def white(seed):
    out=[]
    for j in range(64):
        a,b,c,d=words(seed,j,0x4649454c)
        out.append(math.sqrt(-2*math.log(uniform(a,b)))*math.cos(2*math.pi*uniform(c,d)))
    return np.array(out)

def check_draws(path):
    data=np.loadtxt(path,delimiter=',',skiprows=1)
    assert data.shape==(1024,134) and np.isfinite(data).all()
    assert np.array_equal(data[:,0],np.arange(1024))
    error=0
    for k,row in enumerate(data):
        expected=np.r_[scalars(key(101,1,20007+8*k)),white(key(101,0,20001+8*k)),white(key(101,2,20002+8*k))]
        error=max(error,float(np.max(np.abs(row[1:]-expected)/np.maximum(1,np.abs(expected)))))
    assert error<1e-13
    u=(data[:,1:6]-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])
    assert np.all((u>0)&(u<1))
    # Fixed-fixture moment/CDF screens; not a formal multivariate randomness proof.
    ordered=np.sort(u,axis=0)
    ks=max(float(np.max(np.arange(1,1025)[:,None]/1024-ordered)),float(np.max(ordered-np.arange(1024)[:,None]/1024)))
    assert ks<.07
    assert np.max(np.abs(np.corrcoef(u,rowvar=False)-np.eye(5)))<.15
    w=data[:,6:70];noise=data[:,70:]
    max_cov=0
    for z in (w,noise):
        assert np.max(np.abs(z.mean(axis=0)))<.15
        max_cov=max(max_cov,float(np.max(np.abs(np.cov(z,rowvar=False)-np.eye(64)))))
        assert max_cov<.18
    assert abs(np.mean(w*noise))<.02
    transforms=np.fft.fftn(noise.reshape(-1,4,4,4),axes=(1,2,3),norm='ortho')
    power_error=float(np.max(np.abs(np.mean(np.abs(transforms)**2,axis=0)-1)))
    assert power_error<.18
    return dict(draws=1024,all_native_values_match_reference=True,max_scaled_reference_error=error,
                scalar_uniform_max_cdf_distance=ks,max_white_noise_covariance_error=max_cov,
                max_unit_noise_fourier_power_error=power_error)

def check_target(path):
    target=json.loads(path.read_text())
    assert target['truth_mode']=='prior_predictive_v1'
    m=target['mock'];mid=m['id']
    keys=[key(mid,0,m['truth_seed']),key(mid,1,target['scalar_truth_seed']),key(mid,2,m['noise_seed']),key(mid,3,m['initial_seed_base'])]
    assert [str(v) for v in keys]==target['generation_keys']
    assert np.array_equal(np.array([[p['lower'],p['upper']] for p in target['scalar_priors']]),BOUNDS)
    assert all(p['kind']=='uniform' for p in target['scalar_priors'])
    assert np.allclose(target['truth_parameters'],scalars(keys[1]),rtol=0,atol=1e-13)
    assert np.allclose(target['truth_white'],white(keys[0]),rtol=0,atol=1e-13)
    noise=white(keys[2])
    assert np.allclose(target['noise_white'],noise,rtol=0,atol=1e-13)
    fourier=np.fft.fftn(noise.reshape(4,4,4),norm='ortho')*np.sqrt(1000/100**3)
    residual=np.array([fourier[tuple(np.array(n)%4)] for n in target['mode_n']])
    z=lambda values:np.array(values)[:,0]+1j*np.array(values)[:,1]
    error=float(np.max(np.abs(z(target['data'])-z(target['truth_prediction'])-residual)))
    assert error<1e-12
    q=float(.5*np.sum(np.array(target['weights'])*np.abs(residual)**2)/(1000/100**3))
    assert abs(q-target['truth_quadratic'])<1e-10*max(1,q)
    return dict(target=str(path),reference_scalar_white_noise_match=True,max_fourier_noise_error=error,truth_quadratic=q)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('draws',type=Path)
    parser.add_argument('--targets',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    # Official Philox known-answer vector for zero counter and zero key.
    assert words(0,0,0)==[0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]
    print(json.dumps(dict(passed=True,draws=check_draws(args.draws),targets=[check_target(p) for p in args.targets]),indent=2))

if __name__=='__main__':main()
