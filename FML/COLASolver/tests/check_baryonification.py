#!/usr/bin/env python3
"""Snapshot baryonification integration check. Usage: check_baryonification.py nbody outdir.

MPIRUN defaults to 'mpirun -np 2'; set MPIRUN='' for a serial executable. Requires numpy.
Checks nonzero corrections, unchanged future evolution, exact empty-catalogue controls, parameter/unit
validation, provenance, and optional GADGET/HDF5 corrected particle output. No hydro-calibration claim.
"""
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / 'regression'))
import cases


def read_gadget(prefix):
    rows = []
    for filename in sorted(prefix.parent.glob(prefix.name + '.*')):
        with filename.open('rb') as f:
            def block():
                size = struct.unpack('<I', f.read(4))[0]
                data = f.read(size)
                assert struct.unpack('<I', f.read(4))[0] == size
                return data
            header = block()
            count = struct.unpack('<6I', header[:24])[1]
            pos = np.frombuffer(block(), dtype='<f4').reshape(count, 3)
            vel = np.frombuffer(block(), dtype='<f4').reshape(count, 3)
            ids = block()
            ids = np.frombuffer(ids, dtype='<i8' if len(ids) == 8 * count else '<i4')
            rows.append((ids, pos, vel))
    ids, pos, vel = (np.concatenate([r[j] for r in rows]) for j in range(3))
    order = np.argsort(ids)
    return ids[order], pos[order], vel[order]


def main():
    exe, outdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    launch = shlex.split(os.environ.get('MPIRUN', 'mpirun -np 2'))
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='passive')
    outdir.mkdir(parents=True, exist_ok=True)
    common = dict(cases.BASE, simulation_boxsize=128.0, force_nmesh=64, pofk_nmesh=64,
                  output_redshifts=[1.0, 0.0], timestep_nsteps=[12, 12],
                  ic_fix_amplitude=True, ic_sigma8_normalization=True, ic_sigma8=1.5,
                  baryonification_fof_buffer_mpch=8.0, baryonification_min_halo_mass_msunh=1e13,
                  baryonification_max_radius_over_r200=10.0, baryonification_profile_radial_bins=256,
                  baryonification_output_particles=True)

    def run(name, overrides, expected_error=None):
        d = outdir / name
        d.mkdir(exist_ok=True)
        p = dict(common, **overrides)
        (d / 'parameterfile.lua').write_text(''.join(f'{k} = {cases.lua_value(v)}\n' for k, v in p.items()))
        with (d / 'log.txt').open('w') as log:
            result = subprocess.run(launch + [str(exe), 'parameterfile.lua'], cwd=d, env=env,
                                    stdout=log, stderr=log)
        text = (d / 'log.txt').read_text()
        if expected_error:
            assert result.returncode != 0 and expected_error in text, f'{name}: wrong failure; see {d}/log.txt'
        else:
            assert result.returncode == 0, f'{name}: see {d}/log.txt'
        return d

    baseline = run('baseline', dict(baryonification=False))
    feedback = run('feedback', dict(baryonification=True))
    empty = run('empty', dict(baryonification=True, baryonification_min_halo_mass_msunh=1e30))
    moved = False
    for z in ('1.000', '0.000'):
        sub = Path('output') / f'snapshot_Reg_z{z}'
        b, f, e = baseline / sub, feedback / sub, empty / sub
        # Compare the actual evolved outputs, including the output AFTER baryonification was applied at z=1.
        for original in b.iterdir():
            assert original.read_bytes() == (f / original.name).read_bytes(), f'Live simulation changed: {original.name}'
            assert original.read_bytes() == (e / original.name).read_bytes(), f'Empty control changed: {original.name}'
        spec = np.loadtxt(f / f'pofk_baryonified_z{z}.txt')
        control = np.loadtxt(e / f'pofk_baryonified_z{z}.txt')
        assert np.isfinite(spec).all() and np.isfinite(control).all()
        valid = control[:, 4] == 1
        assert valid.any() and np.array_equal(control[valid, 3], np.ones(valid.sum()))
        normal = np.loadtxt(b / f'pofk_z{z}.txt')
        assert np.allclose(spec[:, 2], normal[:, 1], rtol=3e-5, atol=1e-8)
        moved |= np.any(np.abs(spec[spec[:, 4] == 1, 3] - 1) > 1e-5)
        manifest = f / f'baryonification_z{z}.txt'
        text = manifest.read_text()
        assert 'baryonification_gas_mass_scale_msunh = 100000000000000' in text
        assert '# velocities = unchanged' in text and '# halo_mass_definition = 200c' in text
        rows = [line for line in text.splitlines() if line and not line.startswith('#')]
        assert '# rejected_unresolved = ' in text and '# rejected_support_over_half_box = 0' in text
        if rows:
            halos = np.loadtxt(manifest, ndmin=2)
            # Hosts have at least baryonification_fof_min_particles (20) particles inside R200c
            particle_mass = float(text.split('# particle_mass_msunh = ')[1].split()[0])
            assert (halos[:, 3] >= 20 * particle_mass * (1 - 1e-12)).all(), 'Host with an unresolved M200c'
            # M200c/R200c definition at z=0: input length and mass conventions cannot be mixed.
            if z == '0.000':
                density = halos[:, 3] / (4 * np.pi / 3 * halos[:, 4]**3)
                assert np.allclose(density, 200 * 2.775e11, rtol=5e-4)
            concentration = 5 * (halos[:, 3] / 1e14)**(-.1) / (1 + float(z))
            assert np.allclose(halos[:, 5], concentration, rtol=1e-13)
        ids, pos, vel = read_gadget(f / f'gadget_baryonified_z{z}')
        original_ids, original_pos, original_vel = read_gadget(b / f'gadget_z{z}')
        assert np.array_equal(ids, original_ids) and np.array_equal(vel, original_vel)
        assert (pos >= 0).all() and (pos < 128).all()
    assert moved, 'No nonzero correction exercised: inspect selected_host_count in the manifests'
    print('[PASS] nonzero correction, unit conversions, provenance, particle attributes, and unchanged later evolution')

    # Hosts whose support reaches half the box are skipped and counted, not a fatal error
    support = run('support', dict(baryonification=True, baryonification_max_radius_over_r200=1000.0))
    text = (support / 'output' / 'snapshot_Reg_z0.000' / 'baryonification_z0.000.txt').read_text()
    assert '# selected_host_count = 0' in text and '# rejected_support_over_half_box = 0' not in text, text[:800]
    print('[PASS] hosts with a support over half the box are rejected and reported')

    run('invalid_model', dict(baryonification=True, baryonification_model='unknown'), 'baryonification_model')
    run('invalid_units_value', dict(baryonification=True, baryonification_gas_mass_scale_msunh=-1), 'mass scale')
    run('missing_pofk', dict(baryonification=True, pofk=False), 'baryonification needs pofk')
    run('invalid_species', dict(baryonification=True, particle_baryons=True,
                               all_parameters_must_be_in_file=False), 'single CDM+baryon')
    run('invalid_gravity', dict(baryonification=True, gravity_model='mu',
                               gravity_model_mu_parametrization='DE', gravity_model_mu_mu0=.2,
                               all_parameters_must_be_in_file=False), 'baryonification currently requires')
    features = subprocess.check_output(launch + [str(exe), '--features'], env=env, text=True)
    if 'hdf5' in features:
        hdf = run('hdf5', dict(baryonification=True, output_fileformat='HDF5', output_particles=False))
        files = list(hdf.glob('output/snapshot_Reg_z0.000/gadget_baryonified_z0.000*.hdf5'))
        assert files, 'Missing corrected HDF5 snapshot when ordinary particle output is disabled'
        try:
            import h5py
        except ImportError:
            pass
        else:
            count = 0
            for filename in files:
                with h5py.File(filename) as f:
                    count += len(f['PartType1/ParticleIDs'])
            assert count == 32**3
        print('[PASS] corrected HDF5 output independent of ordinary particle output')
    print('ALL COLASOLVER BARYONIFICATION CHECKS PASSED')


if __name__ == '__main__':
    main()
