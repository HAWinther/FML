#!/usr/bin/env python3
"""Run the test-only clustered GR force experiment, compare MPI ranks, and optionally plot.

Uses the standard library unless --plot is requested (then matplotlib is needed).
Build benchmark_clustered_force first. The output directory retains raw forces,
summary distributions, run logs, source/binary hashes and the exact commands.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot_results(prefix, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_rows(Path(str(prefix) + '-summary.csv'))
    particles = read_rows(Path(str(prefix) + '-particles.csv'))
    fixtures = ['halo', 'merger', 'boundary', 'shifted']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for fixture in fixtures:
        selected = [r for r in rows if r['fixture'] == fixture and r['case'] in ['mesh16', 'mesh32', 'mesh64']]
        axes[0].loglog([int(r['mesh']) for r in selected], [float(r['rms']) for r in selected], 'o-', label=fixture)
        selected = [r for r in particles if r['fixture'] == fixture and r['case'] == 'mesh64']
        values = sorted(float(r['relative_error']) for r in selected)
        axes[1].semilogx(values, [(i + 1) / len(values) for i in range(len(values))], label=fixture)
    axes[0].set(xlabel='Force mesh per dimension', ylabel='RMS force error / reference RMS', title='Cell backend: mesh convergence')
    axes[1].set(xlabel='Relative force error (denominator floor: 0.001 × reference RMS)', ylabel='Cumulative fraction', title='64³ cell force: sampled error distribution')
    for ax in axes:
        ax.grid(True, which='both', alpha=.25)
        ax.legend()
    fig.savefig(output / 'convergence.png', dpi=180)
    fig.savefig(output / 'convergence.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for fixture, ax in zip(fixtures, axes.flat):
        for case in ['mesh64', 'tree03', 'tree06']:
            selected = [r for r in particles if r['fixture'] == fixture and r['case'] == case]
            ax.scatter([float(r['radius']) for r in selected], [float(r['scaled_absolute_error']) for r in selected], s=10, alpha=.65, label=case)
        ax.set(xscale='log', yscale='log', xlabel='Radius from generating halo center / box size', ylabel='Absolute force error / reference RMS', title=fixture)
        ax.grid(True, which='both', alpha=.25)
        ax.legend()
    fig.savefig(output / 'radial-tree-errors.png', dpi=180)
    fig.savefig(output / 'radial-tree-errors.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=2048)
    parser.add_argument('--targets', type=int, default=128)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    if not binary.is_file():
        parser.error('Build benchmark_clustered_force and provide its path')
    if not (128 <= args.particles <= 16384 and 16 <= args.targets <= args.particles):
        parser.error('Require 128..16384 particles and 16..N targets')
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[4]
    sources = [Path(__file__).resolve(), repo / 'FML/SimulationCode/tests/benchmark_clustered_force.cpp',
               repo / 'FML/SimulationCode/tests/ClusteredForceReference.h', repo / 'FML/SimulationCode/tests/ShortRangeReference.h',
               repo / 'FML/SimulationCode/src/GravityModel.h', repo / 'FML/SimulationCode/src/GravityModel_GR.h',
               repo / 'FML/SimulationCode/src/ShortRangeInteraction.h', repo / 'FML/SimulationCode/src/ShortRangeForce.h',
               repo / 'FML/SimulationCode/src/ShortRangeSettings.h', repo / 'FML/NBody/NBody.h',
               repo / 'FML/ShortRangeGravity/ShortRangeGravity.h', repo / 'FML/ShortRangeGravity/Octree.h',
               repo / 'FML/ShortRangeGravity/SlabGhosts.h']
    provenance = dict(binary=str(binary), binary_sha256=digest(binary), platform=platform.platform(),
                      particles=args.particles, targets=args.targets, omp_num_threads=1,
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True),
                      sources={str(p.relative_to(repo)): digest(p) for p in sources}, commands=[])
    prefixes = []
    try:
        for ranks in ([1, 2, 4] if args.mpi else [1]):
            prefix = output / f'ranks-{ranks}'
            command = (shlex.split(args.launcher) + ['-np', str(ranks)] if args.mpi else []) + [str(binary), str(prefix), str(args.particles), str(args.targets)]
            provenance['commands'].append(command)
            with (output / f'ranks-{ranks}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)
            prefixes.append(prefix)
            print(f'{ranks} ranks: clustered force checks passed', flush=True)
        reference = {(r['fixture'], r['case'], r['id']): r for r in read_rows(Path(str(prefixes[0]) + '-particles.csv'))}
        differences = {}
        for prefix in prefixes[1:]:
            current = {(r['fixture'], r['case'], r['id']): r for r in read_rows(Path(str(prefix) + '-particles.csv'))}
            if current.keys() != reference.keys():
                raise AssertionError('MPI output particle sets differ')
            errors = {}
            for key, row in current.items():
                accum = errors.setdefault(row['case'], [0., 0.])
                for d in 'xyz':
                    value, expected = float(row['force_' + d]), float(reference[key]['force_' + d])
                    accum[0] += (value - expected) ** 2
                    accum[1] += expected ** 2
            differences[prefix.name] = {case: math.sqrt(e / scale) for case, (e, scale) in errors.items()}
            for case, error in differences[prefix.name].items():
                # Finite-opening trees may change their accepted nodes with rank layout.
                # Measure that approximation; demand numerical invariance only for exact backends.
                tolerance = .01 if case == 'tree03' else .04 if case == 'tree06' else 1e-10
                if error > tolerance:
                    raise AssertionError(('MPI rank-dependent forces', prefix.name, case, error, tolerance))
        provenance['rank_relative_rms_differences'] = differences
        if args.plot:
            plot_results(prefixes[0], output)
        provenance['status'] = 'passed'
    finally:
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Raw force vectors, distributions and provenance saved to', output)


if __name__ == '__main__':
    main()
