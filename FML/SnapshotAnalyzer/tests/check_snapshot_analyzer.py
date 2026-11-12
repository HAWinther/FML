#!/usr/bin/env python3
"""End-to-end checks of the snapshot_analyzer executable on a synthetic Gadget snapshot.

Usage: check_snapshot_analyzer.py path/to/snapshot_analyzer [--mpirun "mpirun --oversubscribe"] [--np "1 2 4"]
                                  [--workdir DIR]

The snapshot (100 Mpc/h, Omega_m = 0.3, two Gadget files) is a uniform background with three dense blobs (halos) and
a spherical hole of radius 15 Mpc/h (a void). Checks:
  - every analysis writes its files (the bispectrum: nbins^3 rows, symmetric, B > 0 on large scales; the cosmic-web
    fractions sum to 1; the Minkowski functionals: the excursion sets of the blobs have Euler characteristic 3), and
    the results do not depend on the number of MPI tasks or on the read mode (files_per_task / all_tasks);
  - the three largest FoF halos are the blobs, with the right number of particles; the mass functions hold all halos;
  - relaxed FoF (fastpm and paper mode): the blobs, the rfof columns, independent of the number of tasks, and (with
    scipy) the same halos as the independent implementation in FriendsOfFriends/tests/make_rfof_reference.py;
  - baryonification: the hosts are blobs, the corrected snapshot has the same particles (IDs, velocities, masses)
    and only moved positions near the hosts;
  - the largest void (voxel and spherical finders, and ZOBOV with CGAL) is the hole;
  - with an HDF5 build (and h5py): the same snapshot in HDF5 with per-particle masses gives the same results;
  - errors stop every task with a clear message (no hang): old parameter names, a missing snapshot.
Requires numpy (and h5py for the HDF5 check).
"""
import argparse
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

BOX = 100.0
OMEGA_M = 0.3
RHO_CRIT = 2.77536627e11  # h^2 Msun / Mpc^3
BLOBS = [((20.0, 20.0, 20.0), 6000), ((75.0, 30.0, 80.0), 4000), ((30.0, 80.0, 70.0), 3000)]
BLOB_SIGMA = 0.25
HOLE_CENTER, HOLE_RADIUS = (55.0, 55.0, 40.0), 15.0
SUPPORT = 10.0  # baryonification support / R200c (below half the box for the blobs)
NBACKGROUND = 48**3

failures = []


def check(ok, what):
    print(("[PASS] " if ok else "[FAIL] ") + what, flush=True)
    if not ok:
        failures.append(what)
    return ok


def make_particles():
    rng = np.random.default_rng(1234)
    pos = rng.uniform(0, BOX, size=(NBACKGROUND, 3))
    d = pos - np.array(HOLE_CENTER)
    pos = pos[np.sum(d * d, axis=1) > HOLE_RADIUS**2]
    for center, n in BLOBS:
        pos = np.vstack([pos, (np.array(center) + BLOB_SIGMA * rng.standard_normal((n, 3))) % BOX])
    vel = 100.0 * rng.standard_normal(pos.shape)
    ids = np.arange(1, len(pos) + 1, dtype=np.uint32)
    mass = OMEGA_M * RHO_CRIT * BOX**3 / len(pos) / 1e10
    return pos.astype(np.float32), vel.astype(np.float32), ids, mass


def write_gadget(prefix, pos, vel, ids, mass, nfiles=2, a=1.0, omega_m=OMEGA_M, box=BOX):
    parts = np.array_split(np.arange(len(pos)), nfiles)
    for i, idx in enumerate(parts):
        npart = [0, len(idx), 0, 0, 0, 0]
        nall = [0, len(pos), 0, 0, 0, 0]
        header = struct.pack('<6I6dddii6Iiiddddii6Ii', *npart, 0.0, mass, 0, 0, 0, 0, a, 1.0 / a - 1.0, 0, 0, *nall,
                             0, nfiles, box, omega_m, 1.0 - omega_m, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        header += b'\0' * (256 - len(header))
        with open(f'{prefix}.{i}', 'wb') as f:
            for block in (header, pos[idx].tobytes(), vel[idx].tobytes(), ids[idx].tobytes()):
                f.write(struct.pack('<I', len(block)) + block + struct.pack('<I', len(block)))


def write_gadget_hdf5(prefix, pos, vel, ids, mass, nfiles=2):
    import h5py
    parts = np.array_split(np.arange(len(pos)), nfiles)
    for i, idx in enumerate(parts):
        with h5py.File(f'{prefix}.{i}.hdf5', 'w') as f:
            h = f.create_group('Header')
            h.attrs['NumPart_ThisFile'] = np.array([0, len(idx), 0, 0, 0, 0], dtype=np.uint32)
            h.attrs['NumPart_Total'] = np.array([0, len(pos), 0, 0, 0, 0], dtype=np.uint32)
            h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
            h.attrs['MassTable'] = np.zeros(6)  # per-particle masses (PartType1/Masses)
            h.attrs['Time'], h.attrs['Redshift'], h.attrs['BoxSize'] = 1.0, 0.0, BOX
            h.attrs['NumFilesPerSnapshot'] = np.int32(nfiles)
            h.attrs['Omega0'], h.attrs['OmegaLambda'], h.attrs['HubbleParam'] = OMEGA_M, 1.0 - OMEGA_M, 0.7
            g = f.create_group('PartType1')
            g['Coordinates'], g['Velocities'], g['ParticleIDs'] = pos[idx], vel[idx], ids[idx]
            g['Masses'] = np.full(len(idx), mass)


def hybrid_bias_check(r, tasks):
    """Hybrid bias spectra from a linear field file (option C, lattice by ID) and from an IC snapshot (option A):
    Zel'dovich IC and final snapshots of a Gaussian field on a 32^3 lattice in EdS (D = a)"""
    N, box, a_ic = 32, 128.0, 1.0 / 21.0
    rng = np.random.default_rng(42)
    m = np.fft.fftfreq(N, 1.0 / N)
    mz = np.fft.rfftfreq(N, 1.0 / N)
    kx, ky, kz = np.meshgrid(2 * np.pi * m, 2 * np.pi * m, 2 * np.pi * mz, indexing='ij')  # 1/box
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    dk = np.fft.rfftn(rng.standard_normal((N, N, N))) * (np.sqrt(k2) / (2 * np.pi))**-1.2
    dk[0, 0, 0] = 0.0
    dk[N // 2, :, :] = dk[:, N // 2, :] = dk[:, :, N // 2] = 0.0  # no Nyquist modes
    delta = np.fft.irfftn(dk, (N, N, N))
    delta *= 0.02 / delta.std()  # delta_L at z = 20
    dk = np.fft.rfftn(delta)
    psi = np.stack([np.fft.irfftn(1j * kk / k2 * dk, (N, N, N)) for kk in (kx, ky, kz)], axis=-1)  # -div psi = delta
    q = (np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing='ij'), axis=-1) + 0.5) / N
    lattice_ids = np.arange(N**3, dtype=np.uint32) + 7  # lattice index (ix N + iy) N + iz plus an offset
    mass = RHO_CRIT * box**3 / N**3 / 1e10
    workdir = r.workdir
    for name, a in (('hb_ic', a_ic), ('hb_final', 1.0)):
        pos = ((q + a / a_ic * psi) % 1.0).reshape(-1, 3) * box
        order = rng.permutation(N**3)  # the files are not in lattice order
        write_gadget(str(workdir / name), pos[order].astype(np.float32), np.zeros((N**3, 3), np.float32),
                     lattice_ids[order], mass, nfiles=3, a=a, omega_m=1.0, box=box)
    with open(workdir / 'hb_linear_field.dat', 'wb') as f:
        f.write(b'FMLDLIN1' + struct.pack('<qqdd', N, 3, box, 1.0 / a_ic - 1.0))
        f.write(delta.astype('<f8').tobytes())

    base = {'snapshot_input': 'hb_final', 'pofk': False, 'hybridbias': True, 'hybridbias_nmesh': 32,
            'hybridbias_id_offset': 7}
    runs = {'C': {'hybridbias_linear_field_source': 'file', 'hybridbias_linear_field_file': 'hb_linear_field.dat'},
            'A': {'hybridbias_ic_snapshot': 'hb_ic', 'hybridbias_ic_lpt_correction': 'none'},
            'A2lpt': {'hybridbias_ic_snapshot': 'hb_ic'}}
    res = {}
    for key, extra in runs.items():
        for n in tasks:
            name = f'hb_{key}_np{n}'
            code, out = r.run(name, {**base, **extra, 'output_folder': name}, n)
            if not check(code == 0, f'hybrid bias {key} on {n} task(s): runs'):
                print(out[-3000:])
                return
            res[(key, n)] = table(workdir / name / 'hybridbias_z0.000.txt')
    for key in runs:
        check(all(same(res[(key, tasks[0])], res[(key, n)], 1e-5) for n in tasks),
              f'hybrid bias {key}: the same spectra on {tasks} tasks')
    c, a0, a2 = res[('C', tasks[0])], res[('A', tasks[0])], res[('A2lpt', tasks[0])]
    # Normalized differences |dP_ab| / sqrt(P_aa P_bb)
    pairs = [(i, j) for i in range(5) for j in range(i, 5)]
    diag = [pairs.index((i, i)) for i in range(5)]
    norm = np.stack([np.sqrt(np.abs(c[:, 2 + diag[i]] * c[:, 2 + diag[j]])) for i, j in pairs], axis=1)
    d0 = np.abs(a0[:, 2:17] - c[:, 2:17]) / norm
    check(d0.max() < 1e-3, f'hybrid bias: the IC snapshot (Zel\'dovich, no correction) gives the file\'s spectra '
                           f'(max {d0.max():.2e} of sqrt(P_aa P_bb))')
    d2 = np.abs(a2[:, 2:17] - c[:, 2:17]) / norm
    check(0 < d2.max() < 0.05, f'hybrid bias: the 2LPT correction of a Zel\'dovich IC is small ({d2.max():.2e})')
    # Plinear: (D / D_L)^2 box^3 |delta_k / N^3|^2 averaged over the modes of each bin (bin = round(|m| - 1) for the
    # wave number m in units of the fundamental mode; the Nyquist-free field has no modes cut below k_Ny)
    mm = np.sqrt(np.add.outer(np.add.outer(m**2, m**2), m**2)).ravel()
    pk = (np.abs(np.fft.fftn(delta) / N**3)**2).ravel() * box**3 / a_ic**2
    idx = np.floor(mm - 1 + 0.5).astype(int)
    expected = np.array([pk[(idx == b) & (mm > 0)].mean() for b in range(N // 2)])
    inner = np.arange(N // 2) < N // 2 - 2
    rel = np.abs(c[inner, 17] / expected[inner] - 1)
    check(len(c) == N // 2 and rel.max() < 1e-6,
          f'hybrid bias: Plinear is (D / D_L)^2 |delta_L(k)|^2 of the file in the same bins ({rel.max():.1e})')
    # The advected delta_L correlates with the matter (1) on large scales: r = P_1d / sqrt(P_11 P_dd) close to 1
    rcc = c[:, 3] / np.sqrt(c[:, 2] * c[:, 7])
    check(np.all(rcc[c[:, 0] < 0.2] > 0.9), f'hybrid bias: r(1, delta_L) > 0.9 for k < 0.2 h/Mpc ({rcc[:3].round(3)})')
    # Errors: a wrong ID offset; an IC snapshot of another box
    code, out = r.run('hb_bad_offset', {**base, **runs['C'], 'hybridbias_id_offset': 8, 'output_folder': 'hb_bad'})
    check(code not in (0, 'timeout') and 'outside' in out, 'hybrid bias: IDs outside the lattice are an error')


def read_gadget(files):
    """Positions, velocities, ids and masses (header table) of a binary Gadget snapshot"""
    pos, vel, ids, masses = [], [], [], set()
    for filename in files:
        with open(filename, 'rb') as f:
            def block():
                size = struct.unpack('<I', f.read(4))[0]
                data = f.read(size)
                f.read(4)
                return data
            header = block()
            n = struct.unpack('<6I', header[:24])[1]
            table_mass = struct.unpack('<6d', header[24:72])[1]
            pos.append(np.frombuffer(block(), dtype='<f4').reshape(n, 3))
            vel.append(np.frombuffer(block(), dtype='<f4').reshape(n, 3))
            raw = block()
            ids.append(np.frombuffer(raw, dtype='<u8' if len(raw) == 8 * n else '<u4'))
            if table_mass > 0:
                masses.add(table_mass)
            else:
                raw = block()
                masses.update(np.frombuffer(raw, dtype='<f4' if len(raw) == 4 * n else '<f8').tolist())
    pos, vel, ids = np.concatenate(pos), np.concatenate(vel), np.concatenate(ids)
    order = np.argsort(ids)
    return pos[order], vel[order], ids[order], masses


def table(path):
    return np.loadtxt(path, ndmin=2)


def same(a, b, rtol=1e-6):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.all(np.abs(a - b) <= rtol * (np.abs(a) + np.abs(b)) + 1e-30)


class Runner:
    def __init__(self, exe, mpirun, workdir):
        self.exe, self.mpirun, self.workdir = exe, mpirun, workdir

    def run(self, name, params, np_tasks=1, timeout=600):
        lua = self.workdir / f'{name}.lua'
        lines = []
        for k, v in params.items():
            if isinstance(v, bool):
                v = 'true' if v else 'false'
            elif isinstance(v, str):
                v = f'"{v}"'
            elif isinstance(v, (list, tuple)):
                v = '{' + ', '.join(f'"{x}"' if isinstance(x, str) else str(x) for x in v) + '}'
            lines.append(f'{k} = {v}')
        lua.write_text('\n'.join(lines) + '\n')
        cmd = [self.exe, str(lua)]
        if self.mpirun:
            cmd = shlex.split(self.mpirun) + ['-np', str(np_tasks)] + cmd
        try:
            p = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            return 'timeout', str(e.stdout or '') + str(e.stderr or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe')
    ap.add_argument('--mpirun', default='mpirun')
    ap.add_argument('--np', default='1 2 4')
    ap.add_argument('--workdir', default='check_snapshot_analyzer_output')
    args = ap.parse_args()
    exe = str(Path(args.exe).resolve())
    workdir = Path(args.workdir).resolve()
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    tasks = [int(n) for n in args.np.split()] if args.mpirun else [1]
    r = Runner(exe, args.mpirun, workdir)

    pos, vel, ids, mass = make_particles()
    write_gadget(str(workdir / 'snap'), pos, vel, ids, mass)
    print(f'Synthetic snapshot: {len(pos)} particles, mass {mass:.6g} x 1e10 Msun/h', flush=True)

    # What the build supports
    probe = {'snapshot_input': 'snap', 'output_folder': 'probe', 'pofk': False}
    code, out = r.run('probe_cgal', {**probe, 'dtfe': True})
    have_cgal = 'USE_CGAL=true build' not in out
    code, out = r.run('probe_hdf5', {**probe, 'pofk': True, 'snapshot_format': 'gadget_hdf5'})
    have_hdf5 = 'USE_HDF5=true build' not in out
    print(f'Build: CGAL {have_cgal}, HDF5 {have_hdf5}', flush=True)

    base = {
        'snapshot_input': 'snap',
        'pofk_nmesh': 64, 'pofk_interlacing': True, 'pofk_density_assignment_method': 'PCS',
        'pofk_multipole': True, 'pofk_multipole_nmesh': 64,
        'bispectrum': True, 'bispectrum_nmesh': 32, 'bispectrum_nbins': 8,
        'cosmicweb': True, 'cosmicweb_nmesh': 64, 'cosmicweb_lambda_thresholds': [0.0, 0.2, 0.4],
        'minkowski': True, 'minkowski_nmesh': 64, 'minkowski_threshold_variable': 'delta',
        'minkowski_threshold_min': 0.0, 'minkowski_threshold_max': 11.0, 'minkowski_nthresholds': 12,
        'fof': True, 'fof_so_definitions': ['200m', '200c', 'vir'], 'fof_buffer_length_mpch': 5.0,
        'baryonification': True, 'baryonification_output_particles': True, 'baryonification_baryon_fraction': 0.16,
        'baryonification_fof_buffer_mpch': 5.0, 'baryonification_max_radius_over_r200': SUPPORT,
        'voids': True, 'spherical_voids': True, 'void_nmesh': 64,
    }
    if have_cgal:
        base.update({'zobov_voids': True, 'dtfe': True, 'dtfe_nmesh': 32, 'dtfe_buffer_fraction': 1.0})
    files = ['pofk_z0.000.txt', 'pofk_multipole_z0.000.txt', 'bispectrum_z0.000.txt', 'cosmicweb_z0.000.txt',
             'minkowski_z0.000.txt', 'halos_z0.000.txt', 'massfunc_z0.000.txt',
             'massfunc_so200m_z0.000.txt', 'massfunc_so200c_z0.000.txt', 'massfunc_sovir_z0.000.txt',
             'baryonification_z0.000.txt', 'pofk_baryonified_z0.000.txt', 'voids_z0.000.txt',
             'voids_spherical_z0.000.txt'] + (['voids_zobov_z0.000.txt', 'dtfe_spectra_z0.000.txt'] if have_cgal else [])

    runs = [(n, 'files_per_task') for n in tasks] + [(max(tasks), 'all_tasks')]
    results = {}
    for n, mode in runs:
        name = f'np{n}_{mode}'
        code, out = r.run(name, {**base, 'output_folder': name, 'snapshot_read_mode': mode}, n)
        ok = check(code == 0, f'{name}: runs ({code})')
        if not ok:
            print(out[-3000:])
            continue
        missing = [f for f in files if not (workdir / name / f).exists()]
        check(not missing, f'{name}: writes every output file {missing if missing else ""}')
        results[name] = workdir / name

    if not results:
        print('No successful run')
        return 1
    ref_name = next(iter(results))
    ref = results[ref_name]

    # The same results for any number of tasks and read mode (halo order depends on the decomposition: sort)
    def sorted_rows(path):
        t = table(path)
        return t[np.lexsort(t.T[::-1])]
    for name, d in results.items():
        if name == ref_name:
            continue
        for f in ['pofk_z0.000.txt', 'pofk_multipole_z0.000.txt', 'bispectrum_z0.000.txt', 'cosmicweb_z0.000.txt',
                  'minkowski_z0.000.txt', 'pofk_baryonified_z0.000.txt', 'baryonification_z0.000.txt']:
            check(same(sorted_rows(ref / f), sorted_rows(d / f), 1e-6), f'{name}: {f} as {ref_name}')
        # Voids: the same voids; the barycenters (sums in a different order) to 0.01 Mpc/h, not the density ratio
        # (min/max density, ill conditioned when the minimum is ~0)
        for f in ['voids_z0.000.txt', 'voids_spherical_z0.000.txt']:
            a, b = table(ref / f), table(d / f)
            exact = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 16, 17]
            ok = a.shape == b.shape and same(a[:, exact], b[:, exact], 1e-6) and \
                np.all(np.abs(a[:, 12:15] - b[:, 12:15]) < 0.01)
            check(ok, f'{name}: {f} as {ref_name}')
        halos_a, halos_b = sorted_rows(ref / 'halos_z0.000.txt'), sorted_rows(d / 'halos_z0.000.txt')
        big_a, big_b = halos_a[halos_a[:, 0] > 1000], halos_b[halos_b[:, 0] > 1000]
        check(same(big_a, big_b, 1e-6), f'{name}: the large halos as {ref_name}')

    # Bispectrum: nbins^3 rows; the bins and triangle counts symmetric in (k1, k2, k3); B > 0 on large scales
    b = table(ref / 'bispectrum_z0.000.txt')
    nb = base['bispectrum_nbins']
    B = b[:, 6].reshape(nb, nb, nb)
    N = b[:, 8].reshape(nb, nb, nb)
    check(b.shape == (nb**3, 9) and np.all(np.isfinite(b)), f'bispectrum: {nb}^3 rows of 9 finite columns')
    check(same(N, N.transpose(1, 0, 2), 1e-12) and same(N, N.transpose(2, 1, 0), 1e-12) and
          np.allclose(B, B.transpose(1, 0, 2), rtol=1e-6, atol=1e-6 * np.abs(B).max()),
          'bispectrum: B and N symmetric under permutations of (k1, k2, k3)')
    check(B[1, 1, 1] > 0, f'bispectrum: positive equilateral B on large scales (clustered snapshot) ({B[1, 1, 1]:.4g})')

    # Cosmic web: the fractions sum to 1, more voids and fewer knots for a higher threshold
    w = table(ref / 'cosmicweb_z0.000.txt')
    check(w.shape == (3, 9) and np.allclose(w[:, 1:5].sum(1), 1) and np.allclose(w[:, 5:9].sum(1), 1) and
          np.all(np.diff(w[:, 1]) > 0) and np.all(np.diff(w[:, 4]) <= 0),
          'cosmic web: volume and mass fractions sum to 1; more voids, not more knots for a higher lambda_th')
    # For lambda_th >= 0.2 the knots are the cores of the blobs: their mass fraction is that of the blobs
    blob_fraction = sum(n for _, n in BLOBS) / len(pos)
    check(np.all(np.abs(w[1:, 8] - blob_fraction) < 0.01),
          f'cosmic web: knots = the blobs (mass fraction {w[1:, 8].round(4).tolist()} vs {blob_fraction:.4f})')
    # Minkowski functionals: for 3 <= delta_th <= 10 the excursion set is the three blobs (Euler characteristic 3,
    # V3 = 3 / Box^3); V0 decreases with the threshold
    m = table(ref / 'minkowski_z0.000.txt')
    blobs_only = (m[:, 1] >= 3) & (m[:, 1] <= 10)
    check(m.shape == (12, 13) and np.all(m[blobs_only, 10] == 3) and np.allclose(m[blobs_only, 5] * BOX**3, 3) and
          np.all(np.diff(m[:, 2]) <= 0),
          f'Minkowski functionals: Euler characteristic 3 (the blobs) for 3 <= delta_th <= 10 '
          f'({m[blobs_only, 10].astype(int).tolist()}); V0 decreasing')

    # Halos: the three largest are the blobs
    halos = table(ref / 'halos_z0.000.txt')
    halos = halos[np.argsort(-halos[:, 0])]
    for i, (center, n) in enumerate(BLOBS):
        h = halos[i]
        d = (h[2:5] - np.array(center) + BOX / 2) % BOX - BOX / 2
        check(np.linalg.norm(d) < 0.2 and 0.97 * n <= h[0] <= 1.05 * n,
              f'halo {i}: at the blob center ({np.linalg.norm(d):.3f} Mpc/h) with {int(h[0])} ~ {n} particles')
        check(abs(h[1] / (h[0] * mass * 1e10) - 1) < 1e-6, f'halo {i}: mass = particles x particle mass')
    mf = table(ref / 'massfunc_z0.000.txt')
    dlnM = np.log(mf[1, 0] / mf[0, 0])
    nhalos_in_range = np.sum((halos[:, 1] >= mf[0, 0] * np.exp(-dlnM / 2)) & (halos[:, 1] < mf[-1, 0] * np.exp(dlnM / 2)))
    check(abs(np.sum(mf[:, 2]) * dlnM * BOX**3 - nhalos_in_range) < 1e-6 * max(1, nhalos_in_range),
          'mass function: dn/dlnM holds every halo in its range')

    # Baryonification: hosts at the blobs; the corrected snapshot has the same particles, moved near the hosts
    hosts = table(ref / 'baryonification_z0.000.txt')
    near_blob = [min(np.linalg.norm((h[:3] - np.array(c) + BOX / 2) % BOX - BOX / 2) for c, _ in BLOBS) for h in hosts]
    check(len(hosts) == len(BLOBS) and max(near_blob) < 0.2, f'baryonification: {len(hosts)} hosts, at the blobs')
    out_files = sorted((ref).glob('gadget_baryonified_z0.000.*'))
    bpos, bvel, bids, bmass = read_gadget(out_files)
    check(np.array_equal(bids, ids), 'baryonified snapshot: the same particle IDs')
    check(np.array_equal(bvel, vel), 'baryonified snapshot: unchanged velocities')
    check(len(bmass) >= 1 and all(abs(m / mass - 1) < 1e-6 for m in bmass),
          f'baryonified snapshot: the same particle mass ({len(bmass)} distinct value(s))')
    moved = np.linalg.norm((bpos - pos + BOX / 2) % BOX - BOX / 2, axis=1) > 1e-4
    dist = np.min([np.linalg.norm((pos - np.array(c) + BOX / 2) % BOX - BOX / 2, axis=1) for c, _ in BLOBS], axis=0)
    rmax = SUPPORT * hosts[:, 4].max()
    check(moved.any() and dist[moved].max() < rmax * 1.01,
          f'baryonified snapshot: {moved.sum()} particles moved, all within the support of a host')

    # Voids: the largest is the hole
    for f, rtol in [('voids_z0.000.txt', 0.5), ('voids_spherical_z0.000.txt', 0.3)] + \
            ([('voids_zobov_z0.000.txt', 0.5)] if have_cgal else []):
        v = table(ref / f)
        v = v[np.argmax(v[:, 4])]
        d = np.linalg.norm((v[1:4] - np.array(HOLE_CENTER) + BOX / 2) % BOX - BOX / 2)
        check(d < 3.0 and abs(v[4] / HOLE_RADIUS - 1) < rtol,
              f'{f}: the largest void is the hole (offset {d:.2f}, R = {v[4]:.2f} Mpc/h)')
    if have_cgal:
        text = (ref / 'dtfe_spectra_z0.000.txt').read_text()
        check('nan' not in text.lower(), 'DTFE spectra are finite')

    # HDF5 input with per-particle masses: the same results as the binary snapshot
    if have_hdf5:
        try:
            import h5py  # noqa: F401
            write_gadget_hdf5(str(workdir / 'snaph5'), pos, vel, ids, mass)
            n = max(tasks)
            code, out = r.run('hdf5', {**base, 'snapshot_input': 'snaph5', 'snapshot_format': 'gadget_hdf5',
                                       'output_folder': 'hdf5'}, n)
            if check(code == 0, f'HDF5 input runs ({code})'):
                ref_n = results.get(f'np{n}_files_per_task', ref)
                for f in ['pofk_z0.000.txt', 'baryonification_z0.000.txt', 'voids_z0.000.txt']:
                    check(same(table(ref_n / f), table(workdir / 'hdf5' / f), 1e-6), f'HDF5 input: {f} as binary')
                check(len(list((workdir / 'hdf5').glob('gadget_baryonified_z0.000.*.hdf5'))) == n,
                      'HDF5 input: the baryonified snapshot is HDF5')
            else:
                print(out[-3000:])
        except ImportError:
            print('h5py not available: skipping the HDF5 input check')

    # Relaxed FoF (fof_relaxed): the blobs, the rfof columns, independent of the number of tasks, and (with scipy)
    # the same halos as the independent implementation in FriendsOfFriends/tests/make_rfof_reference.py
    rfof_base = {'snapshot_input': 'snap', 'pofk': False, 'fof': True, 'fof_relaxed': True,
                 'fof_buffer_length_mpch': 5.0}
    rfof_results = {}
    for mode in ['fastpm', 'paper']:
        for n in tasks:
            name = f'rfof_{mode}_np{n}'
            code, out = r.run(name, {**rfof_base, 'fof_relaxed_mode': mode, 'output_folder': name}, n)
            if check(code == 0, f'{name}: runs ({code})'):
                rfof_results[(mode, n)] = table(workdir / name / 'halos_z0.000.txt')
            else:
                print(out[-3000:])
    for (mode, n), h in rfof_results.items():
        check(h.shape[1] == 11 + 4, f'relaxed FoF {mode}, {n} tasks: the 4 rfof columns ({h.shape[1]} columns)')
        ref_h = rfof_results.get((mode, tasks[0]))
        if n != tasks[0] and ref_h is not None:
            a, b = ref_h[np.lexsort(ref_h.T[::-1])], h[np.lexsort(h.T[::-1])]
            check(a.shape == b.shape and np.array_equal(a[:, [0, 11, 12]], b[:, [0, 11, 12]]) and same(a, b, 1e-6),
                  f'relaxed FoF {mode}: {n} tasks give the catalogue of {tasks[0]} task(s)')
    for mode in ['fastpm', 'paper']:
        h = rfof_results.get((mode, tasks[0]))
        if h is None:
            continue
        h = h[np.argsort(-h[:, 0])]
        ok = True
        for i, (center, nb) in enumerate(BLOBS):
            d = (h[i, 2:5] - np.array(center) + BOX / 2) % BOX - BOX / 2
            ok = ok and np.linalg.norm(d) < 0.2 and 0.97 * nb <= h[i, 0] <= 1.05 * nb and h[i, 12] == 6
        check(ok, f'relaxed FoF {mode}: the three largest halos are the blobs (in the last bin)')
        check(np.all((h[:, 12] >= 0) & (h[:, 12] <= 6)) and np.all(h[:, 14] < 7.02),
              f'relaxed FoF {mode}: bins in 0-6 and sigma_v/V_std < r0 = 7.02')
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FriendsOfFriends' / 'tests'))
        import make_rfof_reference as rfof_ref
        rfof_ref.MASS_UNIT = mass * 1e10
        p64, v64 = pos.astype(np.float64) / BOX, vel.astype(np.float64)
        for mode in ['fastpm', 'paper']:
            h = rfof_results.get((mode, tasks[0]))
            if h is None:
                continue
            try:
                expected = rfof_ref.relaxed_fof(p64 % 1.0, v64, mode, 0.0, 1.0)
            except rfof_ref.Tight:
                print(f'relaxed FoF {mode}: a pair or halo is within 1e-10 of a threshold; skipping the reference')
                continue
            ref_rows = sorted((len(m), p, b) for p, b, m in expected)
            got_rows = sorted((int(r[0]), int(r[11]), int(r[12])) for r in h)
            check(ref_rows == got_rows, f'relaxed FoF {mode}: the halos (size, pass, bin) of the independent '
                                        f'implementation ({len(got_rows)} vs {len(ref_rows)} halos)')
    except ImportError:
        print('scipy not available: skipping the comparison with the independent relaxed FoF')

    hybrid_bias_check(r, tasks)

    # Errors stop every task with a clear message
    n = max(tasks)
    code, out = r.run('old_names', {'snapshot_input': 'snap', 'output_file': 'x.txt', 'so': True}, n, timeout=120)
    check(code not in (0, 'timeout') and 'no longer used' in out and 'output_folder' in out,
          'an old parameter name stops with the replacement')
    code, out = r.run('missing', {'snapshot_input': 'no_such_snapshot', 'output_folder': 'missing'}, n, timeout=120)
    check(code not in (0, 'timeout') and 'no_such_snapshot' in out, 'a missing snapshot stops with an error (no hang)')
    code, out = r.run('typo', {'snapshot_input': 'snap', 'output_folder': 'typo', 'pofk_nmseh': 32,
                               'pofk_nmesh': 32}, 1)
    check(code == 0 and '[pofk_nmseh]' in out, 'an unknown parameter name gives a warning')

    if failures:
        print(f'{len(failures)} SnapshotAnalyzer checks FAILED')
        return 1
    print('All SnapshotAnalyzer checks passed')
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
