#!/usr/bin/env python3
"""Continuous Gaussian-window mesh/padding and local-LOS checks (NumPy/SciPy)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.special import ndtri
from scipy.stats import qmc

ELLS = (0, 2, 4, 6, 8)
SIGMA = np.array([12., 12., 20.])
CENTER = np.array([0., 0., 200.])
I2 = 1/(8*np.pi**1.5*np.prod(SIGMA))


def gaussian_catalog():
    axes = [np.arange(-5*s, 5*s+1., 2.) for s in SIGMA]
    x = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    weights = 8*np.exp(-.5*np.sum((x/SIGMA)**2, axis=1))/((2*np.pi)**1.5*np.prod(SIGMA))
    return np.column_stack([x+CENTER, weights, np.zeros(len(x))])


def geometry(n=64, length=512., local=False):
    return dict(nmesh=n, boxsize=length, minimum=(CENTER-length/2).tolist(), observer=[0., 0., 0.],
                local=local, fixed_los=[0., 0., 1.])


def poly(ell, mu):
    c = np.zeros(ell+1); c[-1] = 1
    return legval(mu, c)


def fixed_reference(edges):
    rnode, rw = leggauss(48); mu, mw = leggauss(96)
    result = np.empty((5, len(edges)-1))
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        r = lo+.5*(hi-lo)*(1+rnode)
        weight = 1.5*rw*(r/hi)**2/(1+lo/hi+(lo/hi)**2)
        correlation = np.exp(-r[:, None]**2/4*((1-mu**2)/12**2+mu**2/20**2))
        for i, ell in enumerate(ELLS):
            result[i, b] = np.dot(weight, (2*ell+1)*.5*(correlation @ (mw*poly(ell, mu))))
    return result


def local_reference(edges, log2n=17, seed=2718):
    u = qmc.Sobol(6, scramble=True, seed=seed).random_base2(log2n)
    mu = 2*u[:, 1]-1; phi = 2*np.pi*u[:, 2]
    direction = np.column_stack([np.sqrt(1-mu**2)*np.cos(phi), np.sqrt(1-mu**2)*np.sin(phi), mu])
    normal = ndtri(u[:, 3:6])*SIGMA/np.sqrt(2)
    result = np.empty((5, len(edges)-1))
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        radius = (lo**3+u[:, 0]*(hi**3-lo**3))**(1/3)
        separation = radius[:, None]*direction
        origin = CENTER-.5*separation+normal
        angle = np.sum(direction*origin, axis=1)/np.linalg.norm(origin, axis=1)
        correlation = np.exp(-.25*np.sum((separation/SIGMA)**2, axis=1))
        for i, ell in enumerate(ELLS):
            result[i, b] = (2*ell+1)*np.mean(correlation*poly(ell, angle))
    return result


def write(path, points, g, edges, self=False):
    with path.open('w') as f:
        f.write(f"{g['nmesh']} {g['boxsize']:.17g}\n")
        for name in ['minimum', 'observer']:
            f.write(' '.join(format(x, '.17g') for x in g[name])+'\n')
        f.write(str(int(g['local']))+' '+' '.join(map(str, g['fixed_los']))+'\n')
        f.write(f'1 {I2:.17g} 8 {int(self)} {len(edges)}\n')
        f.write(' '.join(map(str, edges))+'\n'+str(len(points))+'\n')
        np.savetxt(f, points, fmt='%.17g')


def read(path):
    rows = list(csv.DictReader(path.open()))
    result = dict(q=np.array([[float(r['q']) for r in rows if int(r['ell']) == ell] for ell in ELLS]))
    for name in ['s', 'nvectors']:
        result[name] = np.array([float(r[name]) for r in rows if int(r['ell']) == 0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi-binary', type=Path)
    parser.add_argument('--launcher', default='mpirun')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    points = gaussian_catalog(); edges = np.arange(0., 193., 8.)
    references = [fixed_reference(edges), local_reference(edges)]
    local_fine = local_reference(edges, 19)
    np.savez(args.output/'continuum-reference.npz', fixed=references[0], local=references[1], local_refined=local_fine, edges=edges)
    summary = [dict(case='local-reference-refinement', max_absolute_difference=float(np.max(np.abs(references[1]-local_fine)))),
               dict(case='local-reference-monopole', max_absolute_difference=float(np.max(np.abs(local_fine[0]-references[0][0]))))]
    assert summary[0]['max_absolute_difference']<3e-4
    assert summary[1]['max_absolute_difference']<1e-7
    references[1] = local_fine
    commands = []
    def run(label, g, pts=points, ranks=1, expected_fail=False):
        path = args.output/(label+'.txt'); output = args.output/(label+'.csv')
        write(path, pts, g, edges)
        cmd = [str(args.binary.resolve()), str(path.resolve()), str(output.resolve())]
        if ranks > 1: cmd = shlex.split(args.launcher)+['-np', str(ranks), str(args.mpi_binary.resolve()), *cmd[1:]]
        start = time.monotonic(); p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        (args.output/(label+'.log')).write_text(p.stdout+p.stderr)
        commands.append(dict(command=cmd, returncode=p.returncode, expected_rejection=expected_fail, seconds=time.monotonic()-start))
        (args.output/'commands.json').write_text(json.dumps(commands, indent=2))
        if expected_fail:
            assert p.returncode and 'extent plus CIC' in p.stderr+p.stdout
            return
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        return read(output)
    for local in (False, True):
        outputs = []
        for n in (32, 64, 128):
            label = f"{'local' if local else 'fixed'}-n{n}"
            actual = run(label, geometry(n, local=local)); outputs.append(actual)
            nonempty=actual['nvectors']>0
            common=nonempty & (edges[:-1]>=48.)
            resolved=nonempty & (edges[:-1]>=3*512/n)
            difference=np.abs(actual['q']-references[int(local)])
            errors=np.max(difference[:,common],axis=1)
            summary.append(dict(case=label, common48_absolute_errors=errors.tolist(),
                all_nonempty_absolute_errors=np.max(difference[:,nonempty],axis=1).tolist(),
                resolved3cells_absolute_errors=np.max(difference[:,resolved],axis=1).tolist(),
                empty_bins=int(np.count_nonzero(~nonempty))))
            if n==128:assert np.max(errors)<.004
            print(label,'common s>=48 errors',errors,flush=True)
        g = geometry(96, 768., local)
        padding = run(f'padding-{int(local)}', g)
        error = np.max(np.abs(padding['q']-outputs[1]['q']))
        summary.append(dict(case=f'padding-{int(local)}', max_absolute_difference=float(error)))
        assert error<1e-10
        assert np.array_equal(padding['nvectors'], outputs[1]['nvectors'])
        print('padding', local, error, flush=True)
    run('underpadded', geometry(48, 384.), expected_fail=True)
    if args.mpi_binary:
        baseline = run('mpi-serial', geometry(64, local=True))
        for n in (2, 4):
            result = run(f'mpi-np{n}', geometry(64, local=True), ranks=n)
            err = np.max(np.abs(result['q']-baseline['q']))
            summary.append(dict(case=f'mpi-np{n}', max_absolute_difference=float(err)))
            assert err<1e-10
            print('MPI', n, err, flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(python=sys.version,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        quadrature_catalog_size=len(points), selection_sigma=SIGMA.tolist(), catalog_step=2., self_pairs_subtracted=False,
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.txt')}), indent=2))
    print('Survey window catalogue experiment: PASS',flush=True)


if __name__ == '__main__': main()
