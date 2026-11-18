#!/usr/bin/env python3
import numpy as np
from statistics import NormalDist
from weights import moments,normalized,shift,mcse,quantiles

def main():
    n=16384;x=np.array([NormalDist().inv_cdf((j+.5)/n) for j in range(n)]).reshape(4,-1,1);zero=np.zeros(x.shape[:2]);mu=.1;sd=1.05
    logw=-.5*((x[:,:,0]-mu)/sd)**2+.5*x[:,:,0]**2-np.log(sd)
    mean,s,weights,logz=moments(x,logw);assert abs(mean[0]-mu)<1e-4 and abs(s[0]-sd)<3e-4 and abs(logz)<1e-4
    assert np.allclose(normalized(logw+10000)[0],weights,rtol=1e-11,atol=1e-14)
    delta,error,a,b=shift(x,zero,zero);assert np.array_equal(delta,np.zeros(1)) and np.array_equal(error,np.zeros(1)) and np.array_equal(a,b)
    assert np.array_equal(quantiles(np.array([-100.,2.,100.]),np.array([0.,1.,0.])),np.array([2.,2.,2.]))
    rng=np.random.default_rng(811);rho=.8;ar=np.zeros((4,8192,1));noise=rng.normal(size=ar.shape)*np.sqrt(1-rho*rho)
    for j in range(1,8192):ar[:,j]=rho*ar[:,j-1]+noise[:,j]
    err=mcse(ar)[0];theory=np.sqrt((1+rho)/(1-rho)/ar.shape[0]/ar.shape[1]);iid=ar.std()/np.sqrt(ar.size)
    assert .5*theory<err<2*theory and err>2*iid
    try:normalized(np.array([np.nan]));raise AssertionError('Nonfinite weights accepted')
    except ValueError:pass
    print('PASS: analytic Gaussian shift/scale, stable weight normalization, identity/paired cancellation, autocorrelated batch MCSE and nonfinite rejection')
if __name__=='__main__':main()
