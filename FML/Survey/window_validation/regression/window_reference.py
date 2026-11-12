#!/usr/bin/env python3
"""Independent window-matrix, IC and optional pypower reference experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.special import spherical_jn

ELLS = (0, 2, 4)
WELLS = (0, 2, 4, 6, 8)


def poly(ell, mu):
    c = np.zeros(ell + 1); c[-1] = 1
    return legval(mu, c)


def analytic_window(s):
    # Autocorrelation of an axisymmetric Gaussian selection: sigma_perp=12, sigma_parallel=20.
    mu, weight = leggauss(96)
    correlation = np.exp(-np.asarray(s)[:, None]**2 / 4 * ((1-mu**2)/12**2 + mu**2/20**2))
    return np.array([(2*ell+1)*.5*(correlation @ (weight*poly(ell, mu))) for ell in WELLS])


def table(ds=2., end=240.):
    edges = np.linspace(0, end, round(end/ds)+1)
    s = .5*(edges[1:]+edges[:-1])
    return dict(edges=edges, s=s, q=analytic_window(s), nvectors=np.ones(len(s)))


def theory(k):
    base = 1e4*np.exp(-400*np.asarray(k)**2)
    return np.array([base, .5*(k/.05)**2*base, .3*(k/.05)**4*base])


def write_matrix(path, w, kin, kout, output_edges):
    with path.open('w') as f:
        f.write(str(len(w['s']))+'\n')
        np.savetxt(f, w['edges'][None, :], fmt='%.17g')
        np.savetxt(f, np.column_stack([w['s'], w['nvectors'], w['q'].T]), fmt='%.17g')
        for v in (kin, kout, output_edges):
            f.write(str(len(v))+'\n'); np.savetxt(f, v[None, :], fmt='%.17g')
        np.savetxt(f, theory(.5*(kin[:-1]+kin[1:])), fmt='%.17g')


def read_matrix(path, nb, no):
    matrix = np.empty((3*no, 3*nb))
    for row in csv.DictReader(path.open()):
        i = ELLS.index(int(row['out_ell']))*no+int(row['out_bin'])
        j = ELLS.index(int(row['in_ell']))*nb+int(row['in_bin'])
        matrix[i, j] = float(row['value'])
    return matrix


def radial_nodes(w, order=24):
    used = w['nvectors'] > 0
    # Split at the actual interpolation knots as well as the integration-bin edges.
    bounds = np.unique(np.r_[w['edges'], w['s'][used]])
    x, weight = leggauss(order)
    r = (.5*(bounds[1:]+bounds[:-1])[:, None]+.5*np.diff(bounds)[:, None]*x).ravel()
    dr = (.5*np.diff(bounds)[:, None]*weight).ravel()
    q = np.array([np.interp(r, w['s'][used], row[used]) for row in w['q']])
    return r, dr, q


def independent_matrix(w, kin, output, binned=False, order=24, continuous=False):
    """Direct 3D angular projection + SciPy Bessel quadrature; returns no-IC and IC matrices."""
    if continuous:
        x, weight = leggauss(order)
        r = w['edges'][-1]*.5*(1+x)
        dr = w['edges'][-1]*.5*weight
        q = analytic_window(r)
    else:
        r, dr, q = radial_nodes(w, order)
    mu, mw = leggauss(48)
    angular_window = sum(row[:, None]*poly(ell, mu) for ell, row in zip(WELLS, q))
    coupling = np.array([[(2*lo+1)*.5*(angular_window @ (mw*poly(lo, mu)*poly(li, mu)))
                         for li in ELLS] for lo in ELLS])
    if binned:
        x, wt = leggauss(32)
        low, high = output[:-1], output[1:]
        kval = (low[:, None]+.5*(high-low)[:, None]*(1+x)).ravel()
        weights = (1.5*wt[None, :]*(kval.reshape(-1, 32)/high[:, None])**2 /
                   (1+low[:, None]/high[:, None]+(low[:, None]/high[:, None])**2))
        no = len(low)
    else:
        kval = output; no = len(output)
    nb = len(kin)-1
    matrices = [np.empty((3*no, 3*nb)) for _ in range(2)]
    gx, gw = leggauss(48)
    qvolume = np.dot(4*np.pi*r*r*dr, q[0])
    for il, ell in enumerate(ELLS):
        k = .5*(kin[1:]+kin[:-1])[:, None]+.5*np.diff(kin)[:, None]*gx
        kw = .5*np.diff(kin)[:, None]*gw*k*k
        xi = (-1)**(ell//2)/(2*np.pi**2)*np.einsum('bq,bqr->br', kw, spherical_jn(ell, k[:, :, None]*r))
        c = (xi*coupling[0, il]) @ (4*np.pi*r*r*dr)/qvolume
        for ol, outell in enumerate(ELLS):
            kernel = 4*np.pi*(-1)**(outell//2)*spherical_jn(outell, kval[:, None]*r)*(r*r*dr)[None, :]
            for ic in (0, 1):
                mixed = xi*coupling[ol, il]-ic*c[:, None]*q[ol]
                block = kernel @ mixed.T
                if binned:
                    block = np.einsum('bq,bqi->bi', weights, block.reshape(no, 32, nb))
                matrices[ic][ol*no:(ol+1)*no, il*nb:(il+1)*nb] = block
    return matrices


def smooth_reference(w, output, binned=False, ic=False):
    # Analytic Hankel transforms of the Gaussian k^ell input; independently project its angular product.
    r, dr, q = radial_nodes(w, 32)
    mu, mw = leggauss(48)
    xi = np.array([1e4*a*scale*np.power(r, ell)*np.exp(-r*r/1600) /
                   (2**(ell+3)*np.pi**1.5*400**(ell+1.5))
                   for ell, a, scale in [(0, 1, 1), (2, -.5, 1/.05**2), (4, .3, 1/.05**4)]])
    angle = sum(row[:, None]*poly(ell, mu) for ell, row in zip(ELLS, xi))
    angle *= sum(row[:, None]*poly(ell, mu) for ell, row in zip(WELLS, q))
    mixed = np.array([(2*ell+1)*.5*(angle @ (mw*poly(ell, mu))) for ell in ELLS])
    if ic:
        c = np.dot(mixed[0], r*r*dr)/np.dot(q[0], r*r*dr)
        mixed -= c*q[:3]
    if binned:
        x, wt = leggauss(32); low, high = output[:-1], output[1:]
        kval = low[:, None]+.5*(high-low)[:, None]*(1+x)
        weights = 1.5*wt[None, :]*(kval/high[:, None])**2/(1+low[:, None]/high[:, None]+(low[:, None]/high[:, None])**2)
    else:
        kval = output
    result = np.array([4*np.pi*(-1)**(ell//2)*(spherical_jn(ell, np.ravel(kval)[:, None]*r) @ (row*r*r*dr))
                       for ell, row in zip(ELLS, mixed)])
    if binned:
        result = np.einsum('lbq,bq->lb', result.reshape(3, len(output)-1, 32), weights)
    return result


def pypower_reference(w, kout, samples=4096):
    from pypower.smooth_window import CorrelationFunctionSmoothWindow, PowerSpectrumSmoothWindowMatrix
    sep = np.geomspace(.001, 1e5, samples)
    # Explicit zero tail; first-end extrapolation is the same constant convention as FML.
    qw = CorrelationFunctionSmoothWindow(np.r_[0., w['s'], w['edges'][-1]],
            np.column_stack([w['q'][:, 0], w['q'], np.zeros(5)]), [(ell, 0) for ell in WELLS])
    matrix = PowerSpectrumSmoothWindowMatrix(kout, projsin=[(ell, 0) for ell in ELLS],
        projsout=[(ell, 0) for ell in ELLS], k=1/sep[::-1], sep=sep,
        kin_lim=(1e-5, 2.), window=qw, sum_wa=False)
    return np.asarray(matrix.dot(theory(matrix.xin[0]), unpack=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pypower', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    kout = np.array([0., .01, .03, .06, .1, .15])
    output_edges = np.array([0., .02, .05, .08, .12, .16])
    base_kin = np.array([0., .015, .03, .05, .07, .1, .14, .2, .3, .4])
    summary = []; commands = []
    def run(label, w, kin):
        source = args.output/(label+'.txt'); prefix = args.output/label
        write_matrix(source, w, kin, kout, output_edges)
        cmd = [str(args.binary.resolve()), str(source.resolve()), str(prefix.resolve())]
        t = time.monotonic(); p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (args.output/(label+'.log')).write_text(p.stdout+p.stderr)
        commands.append(dict(command=cmd, returncode=p.returncode, seconds=time.monotonic()-t))
        (args.output/'commands.json').write_text(json.dumps(commands, indent=2))
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        return prefix
    w = table(8.)
    prefix = run('matrix-coarse', w, base_kin)
    for binned in (False, True):
        output = output_edges if binned else kout
        ref = independent_matrix(w, base_kin, output, binned)
        refined = independent_matrix(w, base_kin, output, binned, order=32)
        outputs = []
        for ic in (0, 1):
            kind = 'binned' if binned else 'point'
            actual = read_matrix(Path(str(prefix)+f'-{kind}-ic{ic}.csv'), len(base_kin)-1, len(output)-int(binned))
            scale = np.max(np.abs(refined[ic]))
            error = np.max(np.abs(actual-refined[ic]))/scale
            controls = np.max(np.abs(ref[ic]-refined[ic]))/scale
            np.savez(args.output/f'{kind}-ic{ic}-reference.npz', actual=actual, reference=refined[ic])
            summary.append(dict(case=f'{kind}-ic{ic}', max_matrix_scaled_error=float(error), reference_refinement=float(controls)))
            assert error < 1e-10 and controls < 1e-11, (kind, ic, error, controls)
            if not binned and ic:
                assert np.max(np.abs(actual[0])) < 1e-11*scale
            print(kind, ic, 'matrix error', error, 'reference refinement', controls, flush=True)
            outputs.append(actual)
        delta = outputs[0]-outputs[1]
        sv = np.linalg.svd(delta, compute_uv=False)
        assert sv[1]/sv[0] < 1e-12
        summary.append(dict(case=('binned' if binned else 'point')+'-IC-rank', second_to_first_singular=float(sv[1]/sv[0])))
    # Smooth-input bandpower convergence, compared to analytic input Hankel transforms, not a finer FML run.
    w = table(2.)
    smooth = smooth_reference(w, kout)
    input_errors=[]
    for step in (.02, .01, .005):
        kin = np.linspace(0, .4, round(.4/step)+1)
        prefix = run(f'input-dk{step:g}', w, kin)
        m = read_matrix(Path(str(prefix)+'-point-ic0.csv'), len(kin)-1, len(kout))
        estimate = (m @ theory(.5*(kin[1:]+kin[:-1])).ravel()).reshape(3, -1)
        err = np.max(np.abs(estimate-smooth))/np.max(np.abs(smooth))
        input_errors.append(err)
        summary.append(dict(case=f'input-dk{step:g}', smooth_scaled_error=float(err)))
        print('input dk', step, 'smooth error', err, flush=True)
    assert input_errors[2] < input_errors[1] < input_errors[0] and input_errors[2]<.001
    # Isolate separation-table interpolation using the SAME finite-band input on both sides.
    kin=np.linspace(0,.4,81); bandpower=theory(.5*(kin[:-1]+kin[1:])).ravel()
    exact=independent_matrix(table(2.),kin,kout,order=256,continuous=True)[0] @ bandpower
    control=independent_matrix(table(2.),kin,kout,order=192,continuous=True)[0] @ bandpower
    assert np.max(np.abs(exact-control))/np.max(np.abs(exact))<1e-10
    separation_errors=[]
    for ds in (8.,4.,2.,1.):
        qtable=table(ds);prefix=run(f'separation-ds{ds:g}',qtable,kin)
        actual=read_matrix(Path(str(prefix)+'-point-ic0.csv'),len(kin)-1,len(kout)) @ bandpower
        error=np.max(np.abs(actual-exact))/np.max(np.abs(exact));separation_errors.append(error)
        summary.append(dict(case=f'separation-ds{ds:g}', continuous_window_scaled_error=float(error)))
        print('separation ds',ds,'continuous-window error',error,flush=True)
    assert separation_errors[-1]<separation_errors[0] and separation_errors[-1]<.001
    # An input support ending near the measured range can miss window-mixed power.
    support_outputs=[]
    for maximum in (.08,.16,.24,.4):
        kin=np.linspace(0,maximum,round(maximum/.005)+1);prefix=run(f'support-kmax{maximum:g}',w,kin)
        value=read_matrix(Path(str(prefix)+'-point-ic0.csv'),len(kin)-1,len(kout)) @ theory(.5*(kin[:-1]+kin[1:])).ravel()
        support_outputs.append(value)
    scale=np.max(np.abs(support_outputs[-1]))
    for maximum,value in zip((.08,.16,.24,.4),support_outputs):
        error=np.max(np.abs(value-support_outputs[-1]))/scale
        summary.append(dict(case=f'support-kmax{maximum:g}', reference_full_support_difference=float(error)))
        print('input support',maximum,'difference',error,flush=True)
    assert np.max(np.abs(support_outputs[0]-support_outputs[-1]))/scale>.005
    assert np.max(np.abs(support_outputs[2]-support_outputs[-1]))/scale<1e-5
    if args.pypower:
        external = pypower_reference(w, kout,8192)
        external_fine = pypower_reference(w, kout,16384)
        np.savez(args.output/'pypower-reference.npz', coarse=external, refined=external_fine, direct=smooth)
        summary.append(dict(case='pypower', max_scaled_error=float(np.max(np.abs(external_fine-smooth))/np.max(np.abs(smooth))),
            refinement=float(np.max(np.abs(external-external_fine))/np.max(np.abs(smooth)))))
        assert summary[-1]['max_scaled_error']<5e-6 and summary[-1]['refinement']<3e-5
        print('pypower', summary[-1], flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(python=sys.version, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.txt')}), indent=2))
    print('Survey window matrix reference experiment: PASS',flush=True)


if __name__ == '__main__': main()
