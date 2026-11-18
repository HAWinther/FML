#!/usr/bin/env python3
"""Bounded, resumable numerical screen. Raw outputs belong outside the library."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def launch(command, log, timeout):
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env=dict(os.environ, OMP_NUM_THREADS='2', OMP_WAIT_POLICY='passive'))
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as interruption:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(interruption, KeyboardInterrupt):
                raise
            code = -1
    return dict(argv=command, exit_code=code, seconds=time.monotonic() - started)


def table(prefix, ranks, kind):
    rows = {}
    for rank in range(ranks):
        with Path(f'{prefix}.{kind}.rank{rank}.csv').open() as stream:
            for row in csv.DictReader(stream):
                key = tuple(int(row[c]) for c in ('nx', 'ny', 'nz'))
                if key in rows:
                    raise ValueError(f'duplicate mode {key}')
                rows[key] = row
    keys = sorted(rows)
    def complex_column(re, im):
        return np.array([complex(float(rows[k][re]), float(rows[k][im])) for k in keys])
    if kind == 'jacalpha':
        return keys, complex_column('re', 'im')
    return dict(keys=keys, prediction=complex_column('re', 'im'),
                columns=np.array([complex_column(f'c{a}re', f'c{a}im') for a in range(4)]).T,
                gradient=complex_column('gre', 'gim'),
                weight=np.array([float(rows[k]['weight']) for k in keys]))


def assess(plan, records):
    reference = records.get('serial-reference')
    if not reference or not reference.get('runtime_passed'):
        return {'reference_available': False, 'cases': {}}
    ref = [table(reference['prefix'], 1, f'state{s}') for s in range(3)]
    keys, alpha = table(reference['prefix'], 1, 'jacalpha')
    assert keys == ref[1]['keys']
    jacobian = np.column_stack((alpha, ref[1]['columns']))
    weights = ref[1]['weight'] / (plan['noise_power'] / plan['boxsize'] ** 3)
    fisher = np.real(jacobian.conj().T @ (weights[:, None] * jacobian))
    scale = 1 / np.sqrt(np.diag(fisher))
    scaled = fisher * scale[:, None] * scale[None, :]
    eigenvalues, vectors = np.linalg.eigh(scaled)
    keep = eigenvalues > plan['gates']['fisher_scaled_rcond'] * eigenvalues[-1]
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    rank = int(keep.sum())
    alpha_sigma = float(scale[0] * np.sqrt(inverse[0, 0])) if rank == 5 else None
    ref_g = np.array(reference['metrics']['scalar_gradients']).reshape(3, 5)
    results = {}
    for name, record in records.items():
        if not record.get('runtime_passed'):
            results[name] = dict(passed=False, reason=record.get('error', 'runtime unavailable'))
            continue
        metric = record['metrics']
        states = [table(record['prefix'], metric['ranks'], f'state{s}') for s in range(3)]
        distances, field_errors = [], []
        for state, expected in zip(states, ref):
            assert state['keys'] == expected['keys']
            assert np.array_equal(state['weight'], expected['weight'])
            distances.append(float(np.sqrt(np.sum(weights * np.abs(state['prediction'] - expected['prediction']) ** 2))))
            numerator = np.sqrt(np.sum(state['weight'] * np.abs(state['gradient'] - expected['gradient']) ** 2))
            denominator = max(1., np.sqrt(np.sum(state['weight'] * np.abs(expected['gradient']) ** 2)))
            field_errors.append(float(numerator / denominator))
        delta_score = (np.array(metric['scalar_gradients']).reshape(3, 5) - ref_g) * scale
        score_distance = np.sqrt(np.maximum(0., np.einsum('si,ij,sj->s', delta_score, inverse, delta_score)))
        shifts = delta_score @ inverse.T * scale
        amplitude = float(np.max(np.abs(shifts[:, 0])) / alpha_sigma) if alpha_sigma else None
        checks = dict(
            finite_differences=max(metric['field_fd_error'], metric['alpha_fd_error']) < plan['gates']['finite_difference_scaled_error'],
            hmc_smoke=metric['hmc_divergences'] == 0,
            model_distance=max(distances) < plan['gates']['noise_weighted_model_distance'],
            field_score=max(field_errors) < plan['gates']['active_field_score_relative_error'],
            fisher_score=rank == 5 and float(score_distance.max()) < plan['gates']['local_fisher_score_distance'],
            amplitude_proxy=rank == 5 and amplitude < plan['gates']['local_amplitude_shift_in_fisher_sigma'])
        parity = None
        if not name.startswith('serial-'):
            serial = records['serial-' + plan['mpi_validation_case']]
            parity = max(float(np.max(np.abs(state['prediction'] - table(serial['prefix'], 1, f'state{s}')['prediction'])))
                         for s, state in enumerate(states))
            checks['mpi_parity'] = parity < plan['gates']['mpi_mode_absolute_error']
        results[name] = dict(passed=all(checks.values()), checks=checks,
                             model_distances=distances, field_score_errors=field_errors,
                             local_fisher_score_distances=score_distance.tolist(),
                             max_local_amplitude_shift_in_fisher_sigma=amplitude,
                             potential_shifts=(np.array(metric['potential']) - reference['metrics']['potential']).tolist(),
                             mpi_prediction_absolute_error=parity)
    return dict(reference_available=True, fisher_rank=rank, fisher_scaled_eigenvalues=eigenvalues.tolist(),
                local_fisher_alpha_sigma=alpha_sigma, cases=results,
                note='Local Fisher proxies do not establish a posterior mean shift or posterior SD budget.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--mpi-slab', type=Path)
    parser.add_argument('--mpi-pencil', type=Path)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    library = Path(__file__).resolve().parents[3]
    if out == library or library in out.parents:
        parser.error("Raw outputs must be outside FieldLevel/")
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('Use a fresh output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if plan['version'] != 1:
        parser.error('Unsupported native screen contract')
    executables = {'serial': args.exe.resolve()}
    if args.mpi_slab:
        executables['slab'] = args.mpi_slab.resolve()
    if args.mpi_pencil:
        executables['pencil'] = args.mpi_pencil.resolve()
    manifest = dict(plan_sha256=digest(args.plan), executable_sha256={k: digest(v) for k, v in executables.items()},
                    omp_num_threads=2, mpirun=args.mpirun)
    if args.resume:
        assert json.loads((out / 'manifest.json').read_text()) == manifest, 'Run contract changed'
    else:
        save(out / 'manifest.json', manifest)
        save(out / 'frozen-plan.json', plan)
    data = out / 'frozen.rank0.data.csv'
    ref = plan['reference']
    def command(case, prefix, action='assess'):
        return [action, str(prefix), str(case['mesh']), str(case['product']), str(case['density']),
                str(case['epsilon']), str(data) if action == 'assess' else '-']
    generation_path = out / 'generation.json'
    if not generation_path.exists():
        record = launch([str(executables['serial']), *command(ref, out / 'frozen', 'generate')],
                        out / 'generation.log', plan['budget']['case_timeout_seconds'])
        record['data_sha256'] = digest(data) if record['exit_code'] == 0 and data.exists() else None
        save(generation_path, record)
    generation = json.loads(generation_path.read_text())
    if generation['exit_code'] != 0:
        raise RuntimeError('Reference data generation failed; retained log')
    assert digest(data) == generation['data_sha256'], 'Frozen data changed'
    cases = [('serial-' + case['name'], 'serial', 1, case) for case in [ref, *plan['cases']]]
    selected = next(c for c in plan['cases'] if c['name'] == plan['mpi_validation_case'])
    for backend in ('slab', 'pencil'):
        if backend in executables:
            cases += [(f'{backend}-{n}', backend, n, selected) for n in (1, 2, 4)]
    records = {}
    spent = {'serial': generation['seconds'], 'mpi': 0.}
    for name, backend, ranks, case in cases:
        record_path = out / (name + '.done.json')
        group = 'serial' if backend == 'serial' else 'mpi'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            for path, checksum in record.get('output_sha256', {}).items():
                assert digest(path) == checksum, 'Saved output changed'
            spent[group] += record.get('seconds', 0.)
            records[name] = record
            continue
        budget = plan['budget']['serial_total_seconds' if group == 'serial' else 'mpi_total_seconds'] - spent[group]
        if budget <= 0:
            records[name] = dict(runtime_passed=False, error='Predeclared total time budget exhausted')
            continue
        prefix = out / name
        meta = Path(str(prefix) + '.json')
        meta.unlink(missing_ok=True)
        cmd = [str(executables[backend]), *command(case, prefix)]
        if group == 'mpi':
            cmd = [*shlex.split(args.mpirun), '-np', str(ranks), *cmd]
        record = launch(cmd, out / (name + '.log'), min(budget, plan['budget']['case_timeout_seconds']))
        record.update(prefix=str(prefix), configuration=case)
        record['runtime_passed'] = record['exit_code'] == 0 and meta.exists()
        if record['runtime_passed']:
            record['metrics'] = json.loads(meta.read_text())
            assert record['metrics']['ranks'] == ranks
            record['output_sha256'] = {str(p): digest(p) for p in out.glob(name + '.*.csv')}
            record['output_sha256'][str(meta)] = digest(meta)
            for state in range(3):
                loaded = table(prefix, ranks, f'state{state}')
                assert np.sum(loaded['weight']) == 124
                assert np.all(np.isfinite(loaded['prediction'])) and np.all(np.isfinite(loaded['gradient']))
        else:
            record['error'] = 'Nonzero exit, timeout or missing native completion metadata'
        save(record_path, record)
        records[name] = record
        spent[group] += record['seconds']
        print(name, 'complete' if record['runtime_passed'] else record['error'], flush=True)
    analysis = assess(plan, records)
    save(out / 'summary.json', dict(records=records, analysis=analysis, spent_seconds=spent,
                                   data_sha256=digest(data), plan_sha256=manifest['plan_sha256']))
    print('Numerical screen complete; results:', out / 'summary.json', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; completed cases are retained. Continue with --resume.', flush=True)
        raise SystemExit(130)
