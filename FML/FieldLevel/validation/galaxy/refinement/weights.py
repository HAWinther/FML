"""Likelihood-ratio sensitivity with correlation-aware, paired shift errors."""
import numpy as np

def normalized(log_ratio):
    x=np.asarray(log_ratio,dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite log ratio')
    r=np.exp(x-x.max());w=r/r.sum()
    return w,float(x.max()+np.log(r.mean()))

def moments(values,log_ratio):
    x=np.asarray(values,dtype=float);w,logz=normalized(log_ratio)
    mean=np.sum(w[...,None]*x,axis=(0,1));var=np.sum(w[...,None]*(x-mean)**2,axis=(0,1))
    return mean,np.sqrt(var),w,logz

def quantiles(x,w,levels=(.05,.5,.95)):
    order=np.argsort(x,kind='stable');v=np.asarray(x)[order];q=np.asarray(w)[order];keep=q>0;v=v[keep];q=q[keep];q=q/q.sum();cdf=np.cumsum(q)-.5*q
    return np.interp(levels,cdf,v)

def mcse(influence,batch_lengths=(64,128)):
    v=np.asarray(influence,dtype=float);chains,n,dim=v.shape
    if chains<2:raise ValueError('Independent chains required')
    estimates=[]
    for b in batch_lengths:
        if n%b or n//b<2:raise ValueError('Batch length must divide chains with at least two batches')
        batches=v.reshape(chains,n//b,b,dim).mean(axis=2)
        estimates.append(np.sqrt(np.sum(np.var(batches,axis=1,ddof=1)/(n//b),axis=0))/chains)
    estimates.append(np.std(v.mean(axis=1),axis=0,ddof=1)/np.sqrt(chains))
    return np.max(estimates,axis=0)

def shift(values,loga,logb,batch_lengths=(64,128)):
    a,sa,wa,_=moments(values,loga);b,sb,wb,_=moments(values,logb)
    n=loga.size;influence=n*(wb[...,None]*(values-b)-wa[...,None]*(values-a))
    return b-a,mcse(influence,batch_lengths),sa,sb
