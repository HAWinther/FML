#!/usr/bin/env python3
"""Validate actual short-range SimulationCode trajectories against adaptive direct Ewald evolution.

Standard-library runner; --plot additionally requires matplotlib. The four small
fixtures are controlled finite-N systems, not equilibrium or production halos.
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
import shlex
import subprocess


def load(prefix):
    samples = {}
    with Path(str(prefix) + '-states.csv').open() as stream:
        for row in csv.DictReader(stream):
            sample, identity = int(row['sample']), int(row['id'])
            values = samples.setdefault(sample, {})
            if identity in values:
                raise AssertionError('Duplicate saved identity')
            values[identity] = {key: float(value) for key, value in row.items()}
    if set(samples) != set(range(17)):
        raise AssertionError('Missing synchronized output')
    identities = set(samples[0])
    if any(set(rows) != identities for rows in samples.values()):
        raise AssertionError('Particle identities changed')
    if any(not math.isfinite(v) for rows in samples.values() for row in rows.values() for v in row.values()):
        raise AssertionError('Nonfinite trajectory')
    return samples


def moments(rows):
    mass = sum(r['mass'] for r in rows.values())
    center = [sum(r['mass'] * r[d] for r in rows.values()) / mass for d in 'xyz']
    mean = [sum(r['mass'] * r['v' + d] for r in rows.values()) / mass for d in 'xyz']
    radius = math.sqrt(sum(r['mass'] * sum((r[d] - center[k]) ** 2 for k, d in enumerate('xyz')) for r in rows.values()) / mass)
    sigma = math.sqrt(sum(r['mass'] * sum((r['v' + d] - mean[k]) ** 2 for k, d in enumerate('xyz')) for r in rows.values()) / mass)
    return mass, center, mean, radius, sigma


def errors(actual, reference):
    if actual.keys() != reference.keys():
        raise AssertionError('Output times differ')
    mass, center0, mean0, radius0, sigma0 = moments(reference[0])
    output = []
    for sample, rows in actual.items():
        ref = reference[sample]
        if rows.keys() != ref.keys() or any(abs(rows[i]['a'] - ref[i]['a']) > 1e-14 or rows[i]['mass'] != ref[i]['mass'] for i in rows):
            raise AssertionError('Mismatched reference times, masses or identities')
        dx2 = sum(r['mass'] * sum((r[d] - ref[i][d]) ** 2 for d in 'xyz') for i, r in rows.items())
        dv2 = sum(r['mass'] * sum((r['v' + d] - ref[i]['v' + d]) ** 2 for d in 'xyz') for i, r in rows.items())
        _, center, mean, radius, sigma = moments(rows)
        _, refcenter, refmean, refradius, refsigma = moments(ref)
        phase, separation, refseparation = 0., 0., 0.
        if len(rows) == 2:
            r = [rows[2][d] - rows[1][d] for d in 'xyz']
            s = [ref[2][d] - ref[1][d] for d in 'xyz']
            separation, refseparation = math.sqrt(sum(x*x for x in r)), math.sqrt(sum(x*x for x in s))
            phase = abs(math.atan2(r[1], r[0]) - math.atan2(s[1], s[0]))
            phase = abs((phase + math.pi) % (2 * math.pi) - math.pi)
        output.append(dict(sample=sample, a=next(iter(rows.values()))['a'],
                           position_error=math.sqrt(dx2 / mass) / radius0,
                           velocity_error=math.sqrt(dv2 / mass) / sigma0,
                           com_error=math.sqrt(sum((x-y)**2 for x, y in zip(center, refcenter))) / radius0,
                           momentum_error=math.sqrt(sum((x-y)**2 for x, y in zip(mean, mean0))) / sigma0,
                           radius=radius, reference_radius=refradius, sigma=sigma, reference_sigma=refsigma,
                           separation=separation, reference_separation=refseparation, phase_error=phase))
    return output


def maximum(series, key):
    return max(row[key] for row in series)


def save_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, summary, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for fixture in sorted({r['fixture'] for r in summary}):
        rows = [r for r in summary if r['fixture'] == fixture and r['variant'] == 'cell']
        for ax, key in zip(axes, ['position_error', 'velocity_error']):
            ax.loglog([r['steps'] for r in rows], [r[key] for r in rows], 'o-', label=fixture)
    axes[0].set(xlabel='Quinn steps', ylabel='Max. RMS position error / initial RMS radius')
    axes[1].set(xlabel='Quinn steps', ylabel='Max. RMS velocity error / initial dispersion')
    for ax in axes:
        ax.grid(True, which='both', alpha=.25)
        ax.legend()
    fig.savefig(output / 'dynamics-convergence.png', dpi=180)
    fig.savefig(output / 'dynamics-convergence.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for fixture, ax in zip(['circular', 'eccentric', 'cluster', 'merger'], axes.flat):
        selected = [(name, rows) for (f, name), rows in curves.items() if f == fixture and name in ['cell', 'pm']]
        for name, rows in selected:
            key = 'separation' if fixture in ['circular', 'eccentric'] else 'radius'
            ax.plot([r['a'] for r in rows], [r[key] for r in rows], 'o-', ms=3, label='PM + short range' if name == 'cell' else 'PM (continuous gradient)')
        if selected:
            rows = selected[0][1]
            key = 'reference_separation' if fixture in ['circular', 'eccentric'] else 'reference_radius'
            ax.plot([r['a'] for r in rows], [r[key] for r in rows], 'k--', label='direct reference')
        ax.set(title='near-circular' if fixture == 'circular' else fixture, xlabel='Scale factor a', ylabel='Pair separation / box' if fixture in ['circular', 'eccentric'] else 'Mass-weighted RMS radius / box')
        ax.grid(True, alpha=.25)
        if selected:
            ax.legend()
    fig.savefig(output / 'dynamics-observables.png', dpi=180)
    fig.savefig(output / 'dynamics-observables.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', nargs='+', choices=['circular', 'eccentric', 'cluster', 'merger'], default=['circular', 'eccentric', 'cluster', 'merger'])
    parser.add_argument('--steps', nargs='+', type=int, default=None)
    parser.add_argument('--mesh', type=int, default=64)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--no-variants', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.steps is not None and (len(args.steps) < 3 or args.steps != sorted(set(args.steps)) or any(n < 16 or n % 16 for n in args.steps)):
        parser.error('Provide at least three increasing step counts, each a positive multiple of 16')
    if args.mesh < 16 or args.mesh % 4:
        parser.error('Mesh must be >=16 and divisible by four')
    binary, output = args.binary.resolve(), args.output.resolve()
    if not binary.is_file():
        parser.error('Build benchmark_short_range_dynamics first')
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location('dynamics_cases', Path(__file__).with_name('cases.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = dict(module.BASE)
    base.update(all_parameters_must_be_in_file=False, simulation_name='Dynamics', simulation_boxsize=100.,
                simulation_use_cola=False, cosmology_OmegaCDM=.25, cosmology_Omegab=.05, cosmology_h=.7,
                particle_Npart_1D=4, particle_allocation_factor=2., ic_nmesh=16, ic_initial_redshift=1.,
                ic_sigma8_normalization=True, ic_sigma8=.01, output_particles=False, pofk=False, pofk_multipole=False,
                fof=False, bispectrum=False, lightcone=False, force_nmesh=args.mesh,
                force_density_assignment_method='PCS', force_greens_function_kernel='continuous',
                force_gradient_kernel='continuous', force_deconvolve_window=True,
                force_short_range=True, force_short_range_split_scale=4., force_short_range_softening_support=1.25,
                force_short_range_backend='cell', force_short_range_cutoff_tolerance=1e-10, force_short_range_table_points=0)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('cases.py'),
               repo / 'FML/SimulationCode/src/SimulationInit.h', repo / 'FML/SimulationCode/src/TimeStepping.h',
               repo / 'FML/SimulationCode/src/ShortRangeForce.h', repo / 'FML/SimulationCode/src/ShortRangeInteraction.h',
               repo / 'FML/SimulationCode/src/GravityModel_GR.h', repo / 'FML/NBody/NBody.h',
               repo / 'FML/MPIParticles/MPIParticles.h', repo / 'FML/SimulationCode/tests/benchmark_short_range_dynamics.cpp',
               repo / 'FML/SimulationCode/tests/DirectDynamicsReference.h', repo / 'FML/SimulationCode/tests/DynamicsFixtures.h',
               repo / 'FML/SimulationCode/tests/ClusteredForceReference.h', repo / 'FML/SimulationCode/src/SimulationStep.h',
               repo / 'FML/SimulationCode/src/GravityModel.h', repo / 'FML/NBody/TimeStepping.h', repo / 'FML/SimulationCode/src/KickDrift.h']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = dict(binary=str(binary), binary_sha256=digest(binary), platform=platform.platform(),
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True),
                      sources={str(p.relative_to(repo)): digest(p) for p in sources}, commands=[], parameter_files={}, status='failed',
                      steps=args.steps, mesh=args.mesh, fixtures=args.fixtures)
    steps_by_fixture = {f: args.steps or ([128, 256, 512, 1024] if f == 'eccentric' else [32, 64, 128, 256]) for f in args.fixtures}
    provenance['steps_by_fixture'] = steps_by_fixture
    provenance['reference_convergence'] = {}
    provenance['convergence_ratios'] = {}
    provenance['rank_differences'] = {}
    provenance['tree_cell_differences'] = {}
    summary, curves = [], {}

    def run(fixture, label, mode, resolution, overrides=None, ranks=1):
        folder = output / (fixture + '-' + label)
        folder.mkdir(exist_ok=True)
        params = dict(base, output_folder=str(folder / 'initialization'), timestep_nsteps=[steps_by_fixture[fixture][-1]],
                      output_redshifts=[1 / dict(circular=.65, eccentric=.65, cluster=.54, merger=.53)[fixture] - 1])
        params.update(overrides or {})
        parameterfile = folder / 'parameters.lua'
        parameterfile.write_text('\n'.join(k + ' = ' + module.lua_value(v) for k, v in params.items()) + '\n')
        provenance['parameter_files'][str(parameterfile.relative_to(output))] = digest(parameterfile)
        prefix = folder / 'trajectory'
        command = (shlex.split(args.launcher) + ['-np', str(ranks)] if args.mpi else []) + [str(binary), str(parameterfile), str(prefix), fixture, mode, str(resolution)]
        provenance['commands'].append(command)
        with (folder / 'run.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)
        print(fixture, label, 'completed', flush=True)
        return load(prefix), prefix

    try:
        for fixture in args.fixtures:
            coarse, _ = run(fixture, 'reference-coarse', 'reference', 1e-10)
            reference, _ = run(fixture, 'reference-fine', 'reference', 1e-12)
            delta = errors(coarse, reference)
            provenance['reference_convergence'][fixture] = {k: maximum(delta, k) for k in ['position_error', 'velocity_error']}
            if maximum(errors(reference, reference), 'momentum_error') > 1e-9:
                raise AssertionError(('Direct reference momentum drift', fixture))
            provenance['convergence_ratios'][fixture] = {}
            if max(maximum(delta, k) for k in ['position_error', 'velocity_error']) > 1e-6:
                raise AssertionError(('Unconverged direct ODE reference', fixture, delta))
            temporal = []
            finest, finest_series = None, None
            step_counts = steps_by_fixture[fixture]
            for nsteps in step_counts:
                state, _ = run(fixture, 'cell-' + str(nsteps), 'simulation', nsteps)
                series = errors(state, reference)
                record = dict(fixture=fixture, variant='cell', steps=nsteps,
                              **{k: maximum(series, k) for k in ['position_error', 'velocity_error', 'com_error', 'momentum_error', 'phase_error']})
                summary.append(record)
                temporal.append(record)
                finest, finest_series = state, series
            for key in ['position_error', 'velocity_error']:
                ratios = [a[key] / b[key] for a, b in zip(temporal, temporal[1:])]
                provenance['convergence_ratios'][fixture][key] = ratios
                if not (temporal[-1][key] < .2 * temporal[0][key] and any(2.5 < r < 6 for r in ratios)):
                    raise AssertionError(('No second-order convergence window', fixture, key, ratios))
            if temporal[-1]['position_error'] > .005 or temporal[-1]['velocity_error'] > .01:
                raise AssertionError(('Resolved trajectory inaccurate', fixture, temporal[-1]))
            if any(r['momentum_error'] > 1e-9 for r in temporal):
                raise AssertionError(('Canonical momentum drift', fixture, temporal))
            curves[fixture, 'cell'] = finest_series
            save_csv(output / (fixture + '-cell-series.csv'), finest_series)
            if not args.no_variants:
                for variant, overrides in [('tree', dict(force_short_range_backend='tree', force_short_range_opening_angle=0.)),
                                            ('interlaced', dict(force_interlacing_order=2)),
                                            ('mesh-coarse', dict(force_nmesh=args.mesh // 2)),
                                            ('pm', dict(force_short_range=False))]:
                    state, _ = run(fixture, variant, 'simulation', steps_by_fixture[fixture][-1], overrides)
                    series = errors(state, reference)
                    summary.append(dict(fixture=fixture, variant=variant, steps=steps_by_fixture[fixture][-1],
                                        **{k: maximum(series, k) for k in ['position_error', 'velocity_error', 'com_error', 'momentum_error', 'phase_error']}))
                    if variant == 'tree':
                        provenance['tree_cell_differences'][fixture] = {k: maximum(errors(state, finest), k) for k in ['position_error', 'velocity_error']}
                        if max(provenance['tree_cell_differences'][fixture].values()) > 1e-8:
                            raise AssertionError(('Exact tree/cell trajectories differ', fixture))
                    if variant == 'interlaced' and (maximum(series, 'position_error') > .005 or maximum(series, 'velocity_error') > .01):
                        raise AssertionError(('Interlaced trajectory inaccurate', fixture))
                    if variant == 'pm' and maximum(series, 'position_error') < 4 * temporal[-1]['position_error']:
                        raise AssertionError(('No improvement over PM-only', fixture))
                    curves[fixture, variant] = series
                    save_csv(output / (fixture + '-' + variant + '-series.csv'), series)
            if args.mpi:
                for ranks in [2, 4]:
                    state, prefix = run(fixture, 'ranks-' + str(ranks), 'simulation', steps_by_fixture[fixture][-1], ranks=ranks)
                    delta = errors(state, finest)
                    provenance['rank_differences'][fixture + '-' + str(ranks)] = {k: maximum(delta, k) for k in ['position_error', 'velocity_error']}
                    if max(maximum(delta, k) for k in ['position_error', 'velocity_error']) > 1e-8:
                        raise AssertionError(('Rank-dependent dynamics', fixture, ranks))
                    if fixture == 'eccentric':
                        with Path(str(prefix) + '-checks.csv').open() as stream:
                            if int(list(csv.DictReader(stream))[-1]['migrations']) == 0:
                                raise AssertionError('Boundary orbit did not exercise ownership migration')
            print(fixture, 'convergence and trajectory checks passed', flush=True)
        save_csv(output / 'summary.csv', summary)
        if args.plot:
            plots(output, summary, curves)
        provenance['status'] = 'passed'
    finally:
        if summary:
            save_csv(output / 'summary.csv', summary)
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('All short-range dynamics checks passed; outputs:', output)


if __name__ == '__main__':
    main()
