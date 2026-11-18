#!/usr/bin/env python3
"""Independent normal-normal SBC control, including a data-ignoring counterexample."""
import json
import numpy as np
from sbc_ranks import endpoint_ranks

def main():
    seed=920401
    rng=lambda domain:np.random.default_rng([seed,domain])
    n=4096;l=4;sigma=.5
    truth=rng(0).normal(size=n)
    data=truth+sigma*rng(1).normal(size=n)
    # Exact conjugate posterior, calculated independently of any FML sampler.
    mean=data/(1+sigma*sigma)
    sd=sigma/np.sqrt(1+sigma*sigma)
    exact=mean[:,None]+sd*rng(2).normal(size=(n,l))
    ignored=rng(3).normal(size=(n,l))
    expected=n/(l+1)
    limit=6*np.sqrt(n/(l+1)*(1-1/(l+1)))
    histograms={}
    for name,draws in [('exact',exact),('ignores_data',ignored)]:
        quantities={'parameter':(truth,draws),'likelihood_Q':(.5*((data-truth)/sigma)**2,.5*((data[:,None]-draws)/sigma)**2)}
        for j,(quantity,(t,x)) in enumerate(quantities.items()):
            ranks,ties=endpoint_ranks(t,x,rng(10+2*(name=='ignores_data')+j))
            assert np.all((ranks>=0)&(ranks<=l)) and not ties.any()
            counts=np.bincount(ranks,minlength=l+1)
            uniform=bool(np.max(np.abs(counts-expected))<limit)
            histograms[name+'_'+quantity]=dict(counts=counts.tolist(),within_fixture_six_sigma_screen=uniform)
            if name=='exact' or quantity=='parameter':assert uniform
            else:assert not uniform and counts[0]>.5*n
    # Ties are randomized across all five valid positions, including both ends.
    ranks,ties=endpoint_ranks(np.zeros(n),np.zeros((n,l)),rng(20))
    assert np.all(ties==l) and set(ranks)==set(range(l+1))
    assert np.max(np.abs(np.bincount(ranks,minlength=l+1)-expected))<limit
    for t,x in [(np.zeros(2),np.zeros((1,4))),(np.array([np.nan]),np.zeros((1,4))), (np.zeros(2),np.empty((2,0)))]:
        try:endpoint_ranks(t,x,rng(21))
        except ValueError:pass
        else:raise AssertionError('Malformed rank inputs accepted')
    print(json.dumps(dict(passed=True,datasets=n,independent_posterior_endpoints=l,histograms=histograms,ties_and_invalid_inputs_checked=True,
                         scope='Analytic rank-control fixture only; not FML nonlinear posterior calibration'),indent=2))

if __name__=='__main__':main()
