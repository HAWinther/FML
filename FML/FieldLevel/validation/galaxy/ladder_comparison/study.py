#!/usr/bin/env python3
"""Bounded, frozen dataset-203 cold-end ladder comparison."""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
GALAXY = HERE.parent
CALIBRATION = GALAXY / 'calibration'
sys.path.insert(0, str(CALIBRATION))
from common import Budget, sha, write
import analyze as calibration_analysis
RUNNER = GALAXY / 'integrator_comparison/run.py'
CASES = ['baseline', 'refined']
SOURCES = [Path(__file__).resolve(), HERE/'study.json', HERE/'preflight.py',
           HERE/'plan_baseline.json', HERE/'plan_refined.json', RUNNER,
           GALAXY/'integrator_comparison/FieldHMC.h', GALAXY/'tempering/Main.cpp',
           GALAXY/'tempering/Exchange.h', GALAXY/'tempering/MockSettings.h',
           GALAXY/'tempering/PriorPredictive.h', GALAXY/'tempering/analyze.py',
           GALAXY/'tempering/check_prior.py', GALAXY/'tempering/plan.json',
           GALAXY/'tempering/sbc_ranks.py', CALIBRATION/'analyze.py',
           CALIBRATION/'study.py', CALIBRATION/'common.py']


def prepare(root, source, exe, diagnostics):
    if any(p.name != 'RUN.lock' for p in root.iterdir()):
        raise ValueError('New comparison needs empty output')
    protocol = json.loads((HERE/'study.json').read_text())
    original = json.loads((source/'plan.json').read_text())
    if original['mock_id'] != 203 or original['seed_layout'] != 3:
        raise ValueError('This declared study requires original dataset 203')
    plans = {name: json.loads((HERE/f'plan_{name}.json').read_text()) for name in CASES}
    expected = dict(original)
    expected.update(samples=512, scope=plans['baseline']['scope'],
                    comparison=plans['baseline']['comparison'], stopping=plans['baseline']['stopping'])
    expected['field'] = dict(original['field'], step_size=.075, n_leapfrog=16)
    if protocol['posterior_gates'] != original['posterior_gates'] or protocol['ensembles'] != original['ensembles']:
        raise ValueError('Original convergence gates and chain count must be retained')
    if plans['baseline'] != expected:
        raise ValueError('Baseline changes undeclared controls')
    expected_refined = dict(expected, betas=[1, math.sqrt(.5), .5, math.sqrt(.125), *expected['betas'][2:]])
    if plans['refined'] != expected_refined:
        raise ValueError('Refinement changes controls beyond the declared ladder')
    if protocol != dict(protocol, dataset=203, cohorts=CASES, warmup=256, samples=512,
                        ensembles=4, wall_limit_seconds=14400, batch_reservation_seconds=2400,
                        field_step=.075, field_leapfrog=16, exchange_passes=4,
                        added_betas=[math.sqrt(.5), math.sqrt(.125)]):
        raise ValueError('Declared study controls changed')
    files = [source/'plan.json', source/'calibration-summary.json']
    for e in range(4):
        files += [source/f'ensemble{e}'/name for name in ['target.json', 'CURRENT', 'summary.json']]
        files += [source/f'ensemble{e}/draws/draw-{j}.rank-0.json' for j in range(2048)]
    frozen = dict(protocol=protocol, binary_sha256=sha(exe), diagnostics_sha256=sha(diagnostics),
                  sources={str(p): sha(p) for p in SOURCES}, original_directory=str(source),
                  original={str(p): sha(p) for p in files}, plans={})
    for name, plan in plans.items():
        (root/name/'plans').mkdir(parents=True)
        path = root/name/'plans/mock203.json'
        write(path, plan)
        frozen['plans'][name] = sha(path)
        write(root/name/'FROZEN.json', dict(binary_sha256=sha(exe)))
    write(root/'FROZEN.json', frozen)
    print('Frozen original 203 and both four-chain 256/512 plans; no chains launched', flush=True)


def verify(root, exe, diagnostics):
    frozen = json.loads((root/'FROZEN.json').read_text())
    if sha(exe) != frozen['binary_sha256'] or sha(diagnostics) != frozen['diagnostics_sha256']:
        raise ValueError('Frozen executable changed')
    for group in ['sources', 'original']:
        for path, digest in frozen[group].items():
            if sha(path) != digest:
                raise ValueError('Frozen source/input changed: '+path)
    for name, digest in frozen['plans'].items():
        if sha(root/name/'plans/mock203.json') != digest:
            raise ValueError('Frozen cohort changed')
    return frozen


def run(root, exe, diagnostics, resume, max_rounds, lock):
    frozen = verify(root, exe, diagnostics)
    protocol = frozen['protocol']
    if not (root/'budget.json').exists() and any((root/name/'mock203').exists() for name in CASES):
        raise ValueError('Missing budget ledger beside native work')
    budget = Budget(root/'budget.json', protocol['wall_limit_seconds'], protocol['batch_reservation_seconds'])
    children = {}
    stop = False
    signaled = set()

    def pause(signum=None, frame=None):
        nonlocal stop
        stop = True
        (root/'PAUSE').touch()
        for name in CASES:
            directory = root/name/'mock203'
            if directory.exists():
                (directory/'PAUSE').touch()
        for child in list(children.values()):
            if child.poll() is None and child.pid not in signaled:
                child.send_signal(signal.SIGTERM)
                signaled.add(child.pid)

    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    if resume:
        (root/'PAUSE').unlink(missing_ok=True)
        for name in CASES:
            (root/name/'mock203/PAUSE').unlink(missing_ok=True)
    rounds = 0
    reason = None

    def batch(name):
        start = time.monotonic()
        command = [sys.executable, str(RUNNER), str(root/name/'mock203'), '--exe', str(exe),
                   '--plan', str(root/name/'plans/mock203.json'), '--workers', '4', '--max-chunks', '1']
        with (root/f'{name}.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
            children[name] = child
            code = child.wait()
        if code:
            raise RuntimeError(name+' runner failed; preserve work and inspect logs')
        elapsed = time.monotonic()-start
        receipt = root/name/'cost.json'
        cost = json.loads(receipt.read_text()) if receipt.exists() else dict(active_cohort_wall_seconds=0., batches=0)
        cost['active_cohort_wall_seconds'] += elapsed
        cost['batches'] += 1
        write(receipt, cost)
        states = [json.loads((root/name/f'mock203/ensemble{e}/summary.json').read_text()) for e in range(4)]
        print(name, [s['completed_iterations'] for s in states], f'{elapsed:.2f} active seconds', flush=True)

    while True:
        todo = [name for name in CASES if not (root/name/'mock203/completed.json').exists()]
        if not todo:
            break
        if stop or (root/'PAUSE').exists():
            reason = 'pause_requested'
            break
        if max_rounds is not None and rounds >= max_rounds:
            reason = 'bounded_round_limit'
            break
        if not budget.reserve():
            reason = 'sampling_time_budget'
            break
        start = time.monotonic()
        futures = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(batch, name) for name in todo]
                while not all(f.done() for f in futures):
                    concurrent.futures.wait(futures, timeout=1)
                    if stop or (root/'PAUSE').exists():
                        pause()
                for future in futures:
                    future.result()
        finally:
            # A failed batch may have missing timing receipts. Keep its full reservation.
            if len(futures) == len(todo) and all(f.done() and f.exception() is None for f in futures):
                budget.finish(time.monotonic()-start)
            children.clear()
        rounds += 1
    complete = all((root/name/'mock203/completed.json').exists() for name in CASES)
    write(root/('completed.json' if complete else 'paused.json'),
          dict(complete=complete, reason=reason, budget=budget.state))
    if complete:
        (root/'paused.json').unlink(missing_ok=True)
    print('Comparison sampling '+('complete' if complete else 'paused'), flush=True)


def analyze(root, exe, diagnostics):
    frozen = verify(root, exe, diagnostics)
    if not json.loads((root/'completed.json').read_text())['complete']:
        raise ValueError('Both fixed budgets must complete before final analysis')
    outcomes = {}
    values = {}
    wall_complete = json.loads((root/'budget.json').read_text())['unrefunded_crashed_batches'] == 0
    for name in CASES:
        plan = json.loads((root/name/'plans/mock203.json').read_text())
        result = calibration_analysis.dataset(root/name, plan, frozen['protocol'], diagnostics)
        result.pop('ranks', None)
        states = json.loads((root/name/'mock203/completed.json').read_text())
        total = sum(r['field_gradient_evaluations_total'] for s in states for r in s['replicas'])
        sampling = sum(r['field_gradient_evaluations_sampling'] for s in states for r in s['replicas'])
        seconds = json.loads((root/name/'cost.json').read_text())['active_cohort_wall_seconds']
        transport = json.loads((root/name/'mock203/summary.json').read_text())
        # The native analysis independently reconstructs all counters and qualified trips.
        attempts = np.sum([s['swap_attempts'] for s in states], axis=0)
        accepted = np.sum([s['swap_accepted'] for s in states], axis=0)
        result.update(field_gradient_evaluations_total=total, field_gradient_evaluations_sampling=sampling,
                      active_cohort_wall_seconds=seconds, wall_cost_complete=wall_complete,
                      min_pooled_edge_acceptance=float(np.min(accepted/attempts)),
                      edge_acceptance_by_ensemble=[(np.array(s['swap_accepted'])/s['swap_attempts']).tolist() for s in states],
                      betas=plan['betas'], transport_reconstruction=transport)
        for d in result['diagnostics']:
            d['bulk_ess_per_total_field_gradient'] = d['bulk_ess']/total if d['bulk_ess'] is not None else None
            d['bulk_ess_per_active_second'] = d['bulk_ess']/seconds if d['bulk_ess'] is not None and wall_complete else None
        outcomes[name] = result
        values[name] = np.stack([np.loadtxt(root/name/f'mock203/calibration.chain{e}.csv', delimiter=',', skiprows=1)[:, 1:] for e in range(4)])
    target = Path(frozen['original_directory'])/'ensemble0/target.json'
    for name in CASES:
        for e in range(4):
            if (root/name/f'mock203/ensemble{e}/target.json').read_bytes() != target.read_bytes():
                raise ValueError('Comparison target differs from original dataset 203')
    report = dict(complete=True, scope=frozen['protocol']['scope'], outcomes=outcomes,
                  original_inputs_unchanged=True, all_cohort_targets_match_original=True,
                  calibration_claim=False, limits=frozen['protocol']['limits']+[
                      'All failed-chain ESS, intervals and work-normalized estimates remain provisional',
                      'Active wall seconds include startup/replay/contention but exclude waiting for the other cohort'])
    write(root/'comparison-summary.json', report)
    plot(root, outcomes, values)
    print(json.dumps({name:dict(passed=v['passed'], divergences=v['measured_divergences'],
                               failed_coordinates=sum(not d['passed'] for d in v['diagnostics']),
                               min_pooled_edge_acceptance=v['min_pooled_edge_acceptance'],
                               roundtrips=v['roundtrips'], gradient_calls=v['field_gradient_evaluations_total'])
                      for name,v in outcomes.items()}, indent=2), flush=True)


def plot(root, outcomes, values):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(root/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for column,name in enumerate(CASES):
        result = outcomes[name]
        for e in range(4):
            color = plt.get_cmap('tab10').colors[e]
            axes[0,column].plot(values[name][e,:,0], lw=.6, color=color, label=f'Chain {e+1}')
            axes[1,column].plot(values[name][e,:,10], lw=.6, color=color)
            axes[2,column].plot(range(len(result['betas'])-1), result['edge_acceptance_by_ensemble'][e], 'o-', ms=3, color=color)
        axes[0,column].set(title=f"{name}: {len(result['betas'])} replicas; gates {'PASS' if result['passed'] else 'FAIL'}", ylabel='sigma8 / reference sigma8')
        axes[0,column].legend(fontsize=7,ncol=2)
        axes[1,column].set(xlabel='Retained draw', ylabel='sin001 white-field projection')
        axes[2,column].set(xlabel='Temperature edge (0 = cold end)', ylabel='Exchange acceptance')
        axes[2,column].axhline(.05, color='black', ls='--', lw=.8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Dataset 203: bounded cold-end ladder comparison\nBoth half-step HMC; four chains each, 256 warmup +512 retained; all failures preserved')
    fig.savefig(root/'ladder-comparison.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-rounds', type=int)
    args = parser.parse_args()
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error('max-rounds must be positive')
    if args.phase == 'prepare' and args.source is None:
        parser.error('prepare needs --source')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'prepare':
            prepare(root, args.source.resolve(), args.exe.resolve(), args.diagnostics.resolve())
        elif args.phase == 'run':
            run(root, args.exe.resolve(), args.diagnostics.resolve(), args.resume, args.max_rounds, lock)
        else:
            analyze(root, args.exe.resolve(), args.diagnostics.resolve())


if __name__ == '__main__':
    main()
