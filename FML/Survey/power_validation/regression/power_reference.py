#!/usr/bin/env python3
"""Deterministic survey-power reference and mesh-convergence experiment (NumPy).
Optional --pypower executes the real installed pypower/pmesh implementation.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(seed=20261112):
    rng = np.random.default_rng(seed)
    def points(n, data):
        x = rng.uniform([-250., -200., 200.], [250., 200., 600.], (n, 3))
        w = rng.uniform(.7, 1.3, n)
        if data:
            w *= 1 + .35 * np.cos(2 * np.pi * x[:, 0] / 350.)
        nbar = 1e-4 * (1 + .25 * (x[:, 2] - 400.) / 200.)
        return np.column_stack([x, w, nbar])
    return points(768, True), points(2304, False)


def geometry(nmesh=64, local=True):
    return dict(nmesh=nmesh, boxsize=1200., minimum=[-500., -500., -100.],
                observer=[-50., 10., -150.], fixed_los=[1., 2., 3.], local=local)


def normalization(data, randoms, override):
    alpha = data[:, 3].sum() / randoms[:, 3].sum()
    return alpha, override if override > 0 else alpha * np.sum(randoms[:, 4] * randoms[:, 3]**2)


def write_input(path, data, randoms, g, edges, compensate, alias, i2):
    with path.open('w') as f:
        f.write(f"{g['nmesh']} {g['boxsize']:.17g}\n")
        for name in ['minimum', 'observer']:
            f.write(' '.join(format(x, '.17g') for x in g[name]) + '\n')
        f.write(str(int(g['local'])) + ' ' + ' '.join(map(str, g['fixed_los'])) + '\n')
        f.write(f'{i2:.17g} {int(compensate)} {int(alias)} {len(edges)}\n')
        f.write(' '.join(format(x, '.17g') for x in edges) + '\n')
        f.write(f'{len(data)} {len(randoms)}\n')
        np.savetxt(f, data, fmt='%.17g')
        np.savetxt(f, randoms, fmt='%.17g')


def read_output(path):
    rows = list(csv.DictReader(path.open()))
    result = {}
    for name in ['raw', 'shotnoise', 'power']:
        result[name] = np.array([[float(r[name]) for r in rows if int(r['ell']) == ell] for ell in [0, 2, 4]])
    first = [r for r in rows if int(r['ell']) == 0]
    for name in ['k', 'nmodes']:
        result[name] = np.array([float(r[name]) for r in first])
    for name in ['alpha', 'i2']:
        result[name] = float(first[0][name])
    return result


def modes(g, edges):
    n = int(np.ceil(edges[-1] * g['boxsize'] / (2 * np.pi)))
    axis = np.arange(-n, n + 1)
    k = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    k = k * (2 * np.pi / g['boxsize'])
    norm = np.linalg.norm(k, axis=1)
    b = np.searchsorted(edges, norm, side='right') - 1
    use = (norm > 0) & (b >= 0) & (b < len(edges) - 1)
    return k[use], norm[use], b[use]


def legendre(ell, mu):
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return .5 * (3 * mu**2 - 1)
    return .125 * (35 * mu**4 - 30 * mu**2 + 3)


def direct_catalog(data, randoms, g, edges, compensate, alias, i2):
    """Continuum catalogue DFT (no CIC) plus independent analytic painted-noise oracle."""
    alpha, norm = normalization(data, randoms, i2)
    x = np.concatenate([data[:, :3], randoms[:, :3]])
    w = np.concatenate([data[:, 3], -alpha * randoms[:, 3]])
    w2 = np.concatenate([data[:, 3]**2, alpha**2 * randoms[:, 3]**2])
    los = x - g['observer'] if g['local'] else np.broadcast_to(g['fixed_los'], x.shape)
    los = los / np.linalg.norm(los, axis=1)[:, None]
    k, kmag, bins = modes(g, edges)
    nb = len(edges) - 1
    counts = np.bincount(bins, minlength=nb)
    means = np.bincount(bins, weights=kmag, minlength=nb) / np.maximum(counts, 1)
    raw = np.zeros((3, nb)); shot = raw.copy()
    for start in range(0, len(k), 64):
        wave = k[start:start+64]; mag = kmag[start:start+64]; b = bins[start:start+64]
        khat = wave / mag[:, None]
        phase = np.exp(-1j * (wave @ x.T))
        a0 = phase @ w
        mu = khat @ los.T
        cell = g['boxsize'] / g['nmesh']
        window = np.prod(np.sinc(wave * cell / (2 * np.pi))**2, axis=1) if compensate else np.ones(len(wave))
        factor = np.prod(1 - 2./3. * np.sin(.5 * wave * cell)**2, axis=1) / window**2 if alias else np.ones(len(wave))
        for il, ell in enumerate([0, 2, 4]):
            poly = legendre(ell, mu)
            ae = (phase * poly) @ w
            value = (2 * ell + 1) * (a0 * ae.conj()).real / norm
            noise = (2 * ell + 1) * factor * (poly @ w2) / norm
            raw[il] += np.bincount(b, weights=value, minlength=nb)
            shot[il] += np.bincount(b, weights=noise, minlength=nb)
    raw /= np.maximum(counts, 1); shot /= np.maximum(counts, 1)
    return dict(raw=raw, shotnoise=shot, k=means, nmodes=counts)


def pypower_reference(data, randoms, g, edges, compensate, norm):
    from pypower import CatalogMesh, MeshFFTPower
    # pypower's observer is the origin; preserve physical mesh nodes by translating all coordinates.
    obs = np.asarray(g['observer'])
    center = np.asarray(g['minimum']) + .5 * g['boxsize'] - obs
    mesh = CatalogMesh(data_positions=data[:, :3] - obs, randoms_positions=randoms[:, :3] - obs,
                       data_weights=data[:, 3], randoms_weights=randoms[:, 3],
                       position_type='pos', nmesh=g['nmesh'], boxsize=g['boxsize'], boxcenter=center,
                       resampler='cic', interlacing=0, dtype='f8')
    # Use pypower's own CIC painting; supply an uncompensated RealField so the transfer convention
    # is explicit. Default noninterlaced CatalogMesh compensation is 'cic-sn', unlike FML's sinc^2.
    field = mesh.to_mesh(compensate=False)
    # Fixed-LOS pypower bins mu explicitly. For oblique LOS, exact parallel modes can
    # round a few ulps outside [-1,1]; expand only the floating endpoint guard.
    mu_guard = 16 * np.finfo(float).eps
    reference_edges = edges if g['local'] else (edges, np.array([-1-mu_guard, 1+mu_guard]))
    poles = MeshFFTPower(field, edges=reference_edges, ells=(0, 2, 4),
                        los='firstpoint' if g['local'] else g['fixed_los'], boxcenter=center,
                        compensations='cic' if compensate else None,
                        wnorm=norm, shotnoise=0., mode_oversampling=0).poles
    return dict(raw=poles.get_power(remove_shotnoise=False, complex=False),
                k=poles.k, nmodes=poles.nmodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pypower', action='store_true')
    parser.add_argument('--mpi-binary', type=Path)
    parser.add_argument('--launcher', default='mpirun')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    data, randoms = fixture()
    edges = np.linspace(.5 * 2 * np.pi / 1200., .045, 7)
    records = []; commands = []; hashes = {}
    def run(label, g, compensate=True, alias=True, override=0., d=data, r=randoms, ranks=1):
        path = args.output / (label + '.txt'); output = args.output / (label + '.csv')
        write_input(path, d, r, g, edges, compensate, alias, override)
        cmd = [str(binary), str(path.resolve()), str(output.resolve())]
        if ranks > 1:
            cmd = shlex.split(args.launcher) + ['-np', str(ranks), str(args.mpi_binary.resolve()), *cmd[1:]]
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (args.output / (label + '.log')).write_text(proc.stdout + proc.stderr)
        commands.append(dict(label=label, command=cmd, returncode=proc.returncode, seconds=time.monotonic()-start))
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2))
        if proc.returncode:
            raise RuntimeError(f'{label} failed: {proc.stdout[-1000:]} {proc.stderr[-1000:]}')
        hashes[path.name] = sha(path)
        actual = read_output(output)
        alpha, norm = normalization(d, r, override)
        assert np.isclose(actual['alpha'], alpha, rtol=1e-13) and np.isclose(actual['i2'], norm, rtol=1e-13)
        return actual
    for local in [False, True]:
        for nmesh in [32, 64]:
            for compensate in [False, True]:
                label = f"{'local' if local else 'fixed'}-n{nmesh}-c{int(compensate)}"
                g = geometry(nmesh, local)
                override = 3.125 if not compensate else 0.
                actual = run(label, g, compensate, override=override)
                oracle = direct_catalog(data, randoms, g, edges, compensate, True, override)
                assert np.array_equal(actual['nmodes'], oracle['nmodes'])
                assert np.allclose(actual['k'], oracle['k'], rtol=1e-12)
                assert np.allclose(actual['shotnoise'], oracle['shotnoise'], rtol=2e-10, atol=1e-8)
                reference_error = None
                if args.pypower:
                    ref = pypower_reference(data, randoms, g, edges, compensate, actual['i2'])
                    np.savez(args.output / (label + '-pypower.npz'), **ref)
                    assert np.array_equal(actual['nmodes'], ref['nmodes'])
                    assert np.allclose(actual['k'], ref['k'], rtol=1e-12)
                    # Scale tolerances by the monopole raw amplitude to handle zero higher multipoles.
                    scale = np.max(np.abs(ref['raw'][0]))
                    reference_error = np.max(np.abs(actual['raw'] - ref['raw'])) / scale
                    assert reference_error < 2e-10, (label, reference_error)
                records.append(dict(case=label, pypower_raw_max_scaled_error=reference_error,
                                    shot_max_abs_error=float(np.max(np.abs(actual['shotnoise']-oracle['shotnoise'])))))
                print(label, 'PASS', reference_error, flush=True)
    # Finite-CIC/LOS convergence against unpainted catalogue DFT, not against a finer FML mesh.
    continuum = direct_catalog(data, randoms, geometry(), edges, True, False, 0.)
    errors = []
    for n in [32, 64, 128]:
        actual = run(f'convergence-n{n}', geometry(n))
        err = np.sqrt(np.mean((actual['raw'] - continuum['raw'])**2, axis=1)) / np.sqrt(np.mean(continuum['raw'][0]**2))
        errors.append(err)
        records.append(dict(case=f'convergence-n{n}', raw_rms_errors=err.tolist()))
    assert np.max(errors[-1]) < .015 and np.max(errors[-1]) < np.max(errors[0]), errors
    # Common coordinate translation leaves observer-relative LOS, CIC nodes and units invariant.
    shift = np.array([120., -75., 230.]); g = geometry(); baseline = run('translation-base', g)
    moved_d = data.copy(); moved_r = randoms.copy(); moved_d[:, :3] += shift; moved_r[:, :3] += shift
    g['minimum'] = (np.asarray(g['minimum'])+shift).tolist(); g['observer'] = (np.asarray(g['observer'])+shift).tolist()
    moved = run('translation-shift', g, d=moved_d, r=moved_r)
    assert np.allclose(baseline['raw'], moved['raw'], rtol=1e-11, atol=1e-8)
    if args.mpi_binary:
        for ranks in [2, 4]:
            actual = run(f'mpi-np{ranks}', geometry(), ranks=ranks)
            for field in ['raw', 'shotnoise', 'power']:
                assert np.allclose(actual[field], baseline[field], rtol=1e-11, atol=1e-8)
            records.append(dict(case=f'mpi-np{ranks}', raw_max_abs_difference=float(np.max(np.abs(actual['raw']-baseline['raw'])))))
    versions = {}
    for package in ['numpy', 'scipy', 'pypower', 'pmesh', 'pfft-python', 'mpi4py']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = dict(python=sys.version, versions=versions, binary_sha256=sha(binary),
                      runner_sha256=sha(__file__), input_sha256=hashes, seed=20261112,
                      fixed_los_pypower_mu_edge_guard=16*np.finfo(float).eps,
                      mpi_binary_sha256=sha(args.mpi_binary) if args.mpi_binary else None)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.output/'summary.json').write_text(json.dumps(records, indent=2))
    print('Survey power reference experiment: PASS', flush=True)


if __name__ == '__main__':
    main()
