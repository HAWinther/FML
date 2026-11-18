#!/usr/bin/env python3
"""Independent quadrature and analytic controls for prior-ratio sensitivity."""
from pathlib import Path
import json
import math
import tempfile
import numpy as np
from analyze import log_ratio, summarize, prepare, analyze, sha, WEIGHTS

def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    bounds = plan['baseline_scalar_bounds']
    nodes, quadrature = np.polynomial.legendre.leggauss(256)
    # Each one-dimensional ratio must integrate to unity under the baseline prior.
    for case in plan['cases']:
        for j in range(5):
            x = np.zeros((1, 256, 11))
            x[..., 0] = 1
            x[..., 1] = 2
            lo, hi = bounds[j]
            x[0, :, j] = (lo+hi)/2 + nodes*(hi-lo)/2
            center = x.copy()
            center[0, :, j] = (lo+hi)/2
            ratio = np.exp(log_ratio(x, case, bounds)[0]-log_ratio(center, case, bounds)[0])
            if j == 0 and case['log_amplitude']:
                density_at_center = (hi-lo)/(math.log(hi/lo)*((lo+hi)/2))
            elif j in (2, 3, 4) and case['nuisance_scales'] is not None:
                s = case['nuisance_scales'][j-2]
                mass = math.erf(hi/(s*math.sqrt(2)))
                density_at_center = (hi-lo)/(s*math.sqrt(2*math.pi)*mass)
            else:
                density_at_center = 1
            assert abs(np.sum(quadrature*ratio*density_at_center)/2-1) < 1e-12
    # A likelihood exp(2 alpha) on [0.5,1.5], importance-weighted uniform alpha
    # posterior -> log-uniform alpha posterior, against independent quadrature.
    n = 8192
    u = (np.arange(n)+.5)/n
    alpha = np.log(np.exp(1)+u*(np.exp(3)-np.exp(1)))/2
    rng = np.random.default_rng(78123)
    x = rng.normal(size=(4, 2048, 11))
    x[..., 0] = alpha.reshape(4, 2048)
    x[..., 1] = 2
    x[..., 2:5] = 0
    w = np.exp(log_ratio(x, plan['cases'][1], bounds))
    estimate = np.sum(w*x[..., 0])/np.sum(w)
    a = 1+nodes/2
    exact = np.sum(quadrature*np.exp(2*a))/np.sum(quadrature*np.exp(2*a)/a)
    assert abs(estimate-exact) < 1e-8
    # Identity and actual reweighting summaries on independent toy monitors.
    x = rng.normal(size=(4, 2048, 11))
    for j, (lo, hi) in enumerate(bounds):
        x[..., j] = rng.uniform(lo, hi, size=(4, 2048))
    result = summarize(x, plan, True)
    base = result['cases'][0]
    assert base['reliability_screen_passed']
    assert all(r['mean_shift_baseline_sd'] == r['paired_shift_mcse_baseline_sd'] == 0 for r in base['coordinates'])
    assert np.allclose(base['overlap_pooled_and_chains'], 1)
    assert result['cases'][1]['coordinates'][0]['mean_shift_baseline_sd'] < -.2
    assert not any(c['reliability_screen_passed'] for c in summarize(x, plan, False)['cases'])
    failed = json.loads(json.dumps(plan))
    failed['reliability_screen']['min_overlap_fraction_per_chain_and_pooled'] = 1.01
    assert not any(c['reliability_screen_passed'] for c in summarize(x, failed, True)['cases'])
    for value in (float('nan'), .1, 2):
        broken = x.copy()
        broken[0, 0, 0] = value
        try:
            log_ratio(broken, plan['cases'][0], bounds)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid source draw accepted')
    # Identity guards reject changed plan/analysis and repeated source datasets.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        planpath = root/'plan.json'
        planpath.write_text(json.dumps(plan))
        identity = dict(plan_sha256=sha(planpath), analysis_sha256=sha(Path(__file__).with_name('analyze.py')),
                        weights_sha256=sha(WEIGHTS))
        (root/'FROZEN.json').write_text(json.dumps(identity))
        planpath.write_text(json.dumps(plan)+'\n')
        try:
            analyze(root)
        except ValueError as exc:
            assert 'changed' in str(exc)
        else:
            raise AssertionError('Changed plan accepted')
    with tempfile.TemporaryDirectory() as tmp:
        try:
            prepare(Path(tmp), Path(__file__).with_name('plan.json'), [Path(tmp)]*3)
        except ValueError as exc:
            assert 'declared datasets' in str(exc)
        else:
            raise AssertionError('Repeated source accepted')
    print('PASS: normalized prior ratios by quadrature; nonuniform posterior tilt; identity/paired cancellation; source-mixing/overlap failures; invalid-draw rejection')

if __name__ == '__main__':
    main()
