#!/usr/bin/env python3
"""Validate the existing COLA frame ordering with common-timestep PM/short forces.

Build test_short_range_simulation first. Uses Python's standard library only.
Checks manufactured nonzero frames, physical initial/final velocities, timestep
refinement, exact backends, interlacing, lightcone transparency and MPI agreement.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess


def distance(first, second):
    if first.keys() != second.keys():
        raise AssertionError('State identities differ')
    dx = dv = 0.
    for identity, values in first.items():
        for d in range(3):
            x = values[d] - second[identity][d]
            dx += (x - round(x)) ** 2
            dv += (values[d + 3] - second[identity][d + 3]) ** 2
    return math.sqrt(dx / (3 * len(first))), math.sqrt(dv / (3 * len(first)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--launcher', default='mpirun')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    if not binary.is_file():
        parser.error('Build test_short_range_simulation first')
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location('cola_cases', Path(__file__).with_name('cases.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = dict(module.BASE)
    base.update(all_parameters_must_be_in_file=False, simulation_name='ShortCOLA', simulation_boxsize=64.,
                simulation_use_cola=False, simulation_use_scaledependent_cola=False, particle_Npart_1D=8,
                ic_nmesh=16, force_nmesh=32, ic_initial_redshift=9., ic_sigma8_normalization=True, ic_sigma8=.1,
                output_redshifts=[0.], output_particles=False, pofk=False, pofk_multipole=False, fof=False,
                lightcone=False, force_density_assignment_method='PCS', force_deconvolve_window=True,
                force_greens_function_kernel='continuous', force_gradient_kernel='continuous',
                force_short_range=True, force_short_range_split_scale=2., force_short_range_softening_support=.2,
                force_short_range_backend='cell', force_short_range_cutoff_tolerance=1e-8,
                force_short_range_table_points=0, gravity_model_screening=False,
                gravity_model_fofr_fofr0=1e-5, gravity_model_fofr_nfofr=1., gravity_model_fofr_exact_solution=False,
                gravity_model_dgp_rcH0overc=1., gravity_model_dgp_exact_solution=False)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('cases.py'),
               repo / 'FML/SimulationCode/tests/test_short_range_simulation.cpp',
               repo / 'FML/SimulationCode/tests/ShortRangeCOLAReference.h',
               repo / 'FML/SimulationCode/tests/ShortRangeReference.h',
               repo / 'FML/SimulationCode/src/SimulationParameters.h', repo / 'FML/SimulationCode/src/COLA.h',
               repo / 'FML/SimulationCode/src/SimulationStep.h', repo / 'FML/SimulationCode/src/GravityModel.h',
               repo / 'FML/SimulationCode/src/ShortRangeForce.h', repo / 'FML/NBody/TimeStepping.h']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = dict(binary=str(binary), binary_sha256=digest(binary), commands=[], parameters={}, status='failed',
                      platform=platform.platform(), sources={str(p.relative_to(repo)): digest(p) for p in sources},
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True),
                      frame_errors=[], initial_errors={}, references={}, lightcone_errors={}, rank_errors={})
    summaries = []

    def run(label, overrides, mode='true-state', ranks=1, reject=False):
        folder = output / label
        folder.mkdir(exist_ok=True)
        params = dict(base, timestep_nsteps=[8], output_folder=str(folder / 'initialization'))
        params.update(overrides)
        parameterfile = folder / 'parameters.lua'
        parameterfile.write_text('\n'.join(k + ' = ' + module.lua_value(v) for k, v in params.items()) + '\n')
        provenance['parameters'][str(parameterfile.relative_to(output))] = digest(parameterfile)
        command = (shlex.split(args.launcher) + ['-np', str(ranks)] if args.mpi else []) + [str(binary), str(parameterfile), str(folder / 'state'), mode]
        provenance['commands'].append(command)
        with (folder / 'run.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1'))
        text = (folder / 'run.log').read_text()
        if reject:
            if result.returncode == 0 or 'force_short_range' not in text:
                raise AssertionError(('Unsupported setting accepted or unrelated failure', label))
            print(label, 'rejected', flush=True)
            return None
        if result.returncode:
            raise AssertionError(('Diagnostic failed', label, result.returncode, str(folder / 'run.log')))
        rows = {}
        for file in folder.glob('state.*.txt'):
            for line in file.read_text().splitlines():
                fields = line.split()
                identity = int(fields[0])
                if identity in rows:
                    raise AssertionError(('Duplicate ID', label, identity))
                rows[identity] = list(map(float, fields[1:]))
        expected = 7 if mode == 'frame-check' else 512
        if len(rows) != expected or any(len(v) != 6 or not all(map(math.isfinite, v)) for v in rows.values()):
            raise AssertionError(('Invalid saved state', label, len(rows)))
        if mode == 'frame-check':
            errors = re.findall(r'COLA_FRAME_ERROR (\d+) (\S+) (\S+)', text)
            if len(errors) != 3 or 'tests passed (9 checks)' not in text:
                raise AssertionError(('Missing frame oracle checks', label))
            provenance['frame_errors'].append(dict(label=label, velocity=max(float(e[1]) for e in errors),
                                                    position=max(float(e[2]) for e in errors)))
        print(label, 'passed', flush=True)
        return rows

    try:
        # Independent body-force oracle plus nonzero 1LPT/2LPT frame fields.
        # GR uses closed EdS frame factors; MG uses derivatives of public frame velocities.
        for model in ['GR', 'DGP', 'f(R)']:
            for scaled in ([False, True] if model == 'f(R)' else [False]):
                for order in [1, 2]:
                    for interlace in [1, 2]:
                        p = dict(gravity_model=model, simulation_use_cola=True, simulation_use_scaledependent_cola=scaled,
                                 simulation_cola_LPT_order=order, force_interlacing_order=interlace,
                                 simulation_boxsize=100., force_nmesh=64, force_short_range_split_scale=4.,
                                 gravity_model_fofr_fofr0=1e-6)
                        if model == 'GR':
                            p.update(cosmology_OmegaCDM=.95, cosmology_Omegab=.05, cosmology_TCMB_kelvin=1e-6)
                        name = f'frame-{model}-{scaled}-{order}-{interlace}'
                        run(name, p, mode='frame-check')
                        if args.mpi and order == 2 and interlace == 2 and (model == 'GR' or scaled):
                            for ranks in [2, 4]:
                                run(name + '-ranks-' + str(ranks), dict(p, force_short_range_backend='tree', force_short_range_opening_angle=0.), mode='frame-check', ranks=ranks)

        models = ['GR', 'DGP', 'f(R)']
        references, initial = {}, {}
        for model in models:
            initial[model] = run('initial-nbody-' + model, dict(gravity_model=model, output_redshifts=[9.], timestep_nsteps=[0]))
            coarse = run('reference64-' + model, dict(gravity_model=model, timestep_nsteps=[64]))
            references[model] = run('reference128-' + model, dict(gravity_model=model, timestep_nsteps=[128]))
            uncertainty = distance(coarse, references[model])
            provenance['references'][model] = uncertainty
            if max(uncertainty) > 2e-6:
                raise AssertionError(('N-body reference insufficiently converged', model, uncertainty))
        configurations = [('GR1', 'GR', False, 1), ('GR2', 'GR', False, 2), ('DGP2', 'DGP', False, 2),
                          ('fR2', 'f(R)', False, 2), ('fRscale1', 'f(R)', True, 1), ('fRscale2', 'f(R)', True, 2)]
        for label, model, scaled, order in configurations:
            settings = dict(gravity_model=model, simulation_use_cola=True, simulation_use_scaledependent_cola=scaled,
                            simulation_cola_LPT_order=order)
            state0 = run('initial-' + label, dict(settings, output_redshifts=[9.], timestep_nsteps=[0]))
            delta0 = distance(state0, initial[model])
            provenance['initial_errors'][label] = delta0
            if delta0[0] > 1e-10 or delta0[1] > 2e-8:
                raise AssertionError(('COLA physical initial state differs', label, delta0))
            for method in ['Quinn', 'Tassev']:
                states = {}
                for n in [4, 8, 16, 32]:
                    state = run(f'{label}-{method}-{n}', dict(settings, timestep_method=method, timestep_nsteps=[n]))
                    states[n] = state
                    dx, dv = distance(state, references[model])
                    summaries.append(dict(configuration=label, method=method, steps=n, position_rms=dx, velocity_rms=dv))
                fine_error = distance(states[32], references[model])
                if fine_error[0] > 5e-5 or fine_error[1] > 3e-5:
                    raise AssertionError(('COLA physical state fails N-body comparison', label, method, fine_error))
                for component in [0, 1]:
                    large, small = distance(states[4], states[32])[component], distance(states[16], states[32])[component]
                    if large > 1e-10 and small > .5 * large:
                        raise AssertionError(('COLA timestep refinement fails', label, method, component, large, small))
                if label in ['GR2', 'fRscale2']:
                    for variant, options in [('tree', dict(force_short_range_backend='tree', force_short_range_opening_angle=0.)),
                                             ('interlaced', dict(force_interlacing_order=2)),
                                             ('table', dict(force_short_range_table_points=16384))]:
                        state = run(f'{label}-{method}-{variant}', dict(settings, timestep_method=method, timestep_nsteps=[32], **options))
                        if variant == 'tree' and max(distance(state, states[32])) > 1e-10:
                            raise AssertionError(('Exact tree/cell COLA mismatch', label, method))
                        if variant == 'table' and max(distance(state, states[32])) > 1e-9:
                            raise AssertionError(('Tabulated/analytic COLA mismatch', label, method))
                        if variant == 'interlaced' and max(distance(state, references[model])) > 5e-5:
                            raise AssertionError(('Interlaced COLA mismatch', label, method))
                    if args.mpi:
                        for ranks in [2, 4]:
                            state = run(f'{label}-{method}-ranks-{ranks}', dict(settings, timestep_method=method, timestep_nsteps=[32]), ranks=ranks)
                            delta = distance(state, states[32])
                            provenance['rank_errors'][f'{label}-{method}-{ranks}'] = delta
                            if max(delta) > 2e-9:
                                raise AssertionError(('Rank-dependent COLA state', label, method, ranks, delta))
                    # The lightcone's split frame kick/drift and physical-velocity scope
                    # must leave the trajectory equal to its no-lightcone counterpart.
                    lowz = dict(settings, timestep_method=method, timestep_nsteps=[4], ic_initial_redshift=.005)
                    off = run(f'{label}-{method}-lightcone-off', lowz)
                    on = run(f'{label}-{method}-lightcone-on', dict(lowz, lightcone=True, plc_z_init=.005, plc_z_finish=0.,
                             plc_pos_observer=[.5,.5,.5], plc_use_replicas=True, plc_ndim_rep=3,
                             plc_output_gadgetfile=False, plc_output_asciifile=True, plc_relativistic_effects=True), ranks=2 if args.mpi else 1)
                    delta = distance(off, on)
                    provenance['lightcone_errors'][label + '-' + method] = delta
                    if max(delta) > 2e-9:
                        raise AssertionError(('Lightcone changes COLA dynamics', label, method, delta))
                    if 'LightconePotential' not in (output / f'{label}-{method}-lightcone-on/run.log').read_text():
                        raise AssertionError('Unsplit lightcone potential path was not exercised')
        zero_order = run('zero-order-quinn', dict(simulation_use_cola=True, simulation_cola_LPT_order=0, timestep_nsteps=[128]))
        if max(distance(zero_order, references['GR'])) > 1e-12:
            raise AssertionError('Zero-order COLA did not reduce to ordinary Quinn N-body')
        for label, setting in [('enforced', dict(simulation_use_cola=True, simulation_enforce_LPT_trajectories=True)),
                               ('tassev-nbody', dict(timestep_method='Tassev')),
                               ('tassev-zero-order', dict(simulation_use_cola=True, simulation_cola_LPT_order=0, timestep_method='Tassev')), ('fastpm', dict(timestep_method='FastPM')),
                               ('screened', dict(gravity_model='DGP', gravity_model_screening=True))]:
            run('reject-' + label, setting, reject=True)
        provenance['status'] = 'passed'
    finally:
        if summaries:
            with (output / 'summary.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('All common-timestep short-range COLA checks passed', flush=True)


if __name__ == '__main__':
    main()
