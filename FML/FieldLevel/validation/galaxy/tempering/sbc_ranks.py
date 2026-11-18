"""Coarse SBC ranks for independently generated posterior endpoints.

The caller must establish endpoint independence and posterior stationarity. This
function cannot convert correlated MCMC records into independent draws.
"""
import numpy as np

def endpoint_ranks(truth, endpoints, rng):
    truth=np.asarray(truth,dtype=float)
    endpoints=np.asarray(endpoints,dtype=float)
    if truth.ndim!=1 or endpoints.ndim!=2 or endpoints.shape[0]!=truth.size or endpoints.shape[1]<1:
        raise ValueError('One truth per dataset and a nonempty matrix of endpoints required')
    if not np.isfinite(truth).all() or not np.isfinite(endpoints).all():
        raise ValueError('Finite SBC quantities required')
    less=np.sum(endpoints<truth[:,None],axis=1)
    equal=np.sum(endpoints==truth[:,None],axis=1)
    # Randomize the truth's position among the tied values, including itself.
    ties=np.array([rng.integers(int(n)+1) for n in equal],dtype=int)
    return less+ties,equal
