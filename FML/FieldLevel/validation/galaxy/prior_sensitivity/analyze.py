#!/usr/bin/env python3
"""Freeze inputs, then audit a bounded prior change on existing cold chains."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / 'refinement/weights.py'
spec = importlib.util.spec_from_file_location('paired_weights', WEIGHTS)
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def log_ratio(x, case, bounds):
    """Normalized alternative/baseline scalar prior density ratio."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or x.shape[2] < 5 or not np.isfinite(x).all():
        raise ValueError('Finite chain/draw/coordinate array required')
    for j, (lo, hi) in enumerate(bounds):
        if not lo < hi or np.any((x[..., j] < lo) | (x[..., j] > hi)):
            raise ValueError('Source draw outside declared baseline support')
    out = np.zeros(x.shape[:2])
    if case['log_amplitude']:
        lo, hi = bounds[0]
        if lo <= 0:
            raise ValueError('Positive amplitude support required')
        out += np.log((hi-lo)/math.log(hi/lo)) - np.log(x[..., 0])
    scales = case['nuisance_scales']
    if scales is not None:
        if len(scales) != 3:
            raise ValueError('Three nuisance scales required')
        for j, scale in zip((2, 3, 4), scales):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Finite positive Gaussian scale required')
            lo, hi = bounds[j]
            mass = .5*(math.erf(hi/(scale*math.sqrt(2))) - math.erf(lo/(scale*math.sqrt(2))))
            out += math.log((hi-lo)/(scale*math.sqrt(2*math.pi)*mass)) - .5*(x[..., j]/scale)**2
    return out

def summarize(x, plan, source_passed):
    zero = np.zeros(x.shape[:2])
    base_mean, base_sd, _, _ = weights.moments(x, zero)
    if not np.all(base_sd > 0):
        raise ValueError('Nondegenerate monitored coordinates required')
    rows = []
    for case in plan['cases']:
        logw = log_ratio(x, case, plan['baseline_scalar_bounds'])
        mean, sd, w, _ = weights.moments(x, logw)
        delta, error, _, _ = weights.shift(x, zero, logw, plan['batch_lengths'])
        overlap = [float(1/(np.sum(v*v)*v.size)) for v in [w.ravel(), *[weights.normalized(v)[0] for v in logw]]]
        largest = max([float(w.max()*w.size), *[float(weights.normalized(v)[0].max()*v.size) for v in logw]])
        screen = plan['reliability_screen']
        gates = dict(source_mixing=bool(source_passed),
                     overlap=min(overlap) >= screen['min_overlap_fraction_per_chain_and_pooled'],
                     maximum_weight=largest <= screen['max_weight_relative_to_uniform'],
                     shift_precision=float(np.max(error/base_sd)) <= screen['max_paired_shift_mcse_in_baseline_sd'])
        coords = []
        for j, name in enumerate(plan['parameters']):
            intervals = {}
            for level in plan['interval_probabilities']:
                lo, hi = weights.quantiles(x[..., j].ravel(), w.ravel(), ((1-level)/2, 1-(1-level)/2))
                intervals[str(level)] = [float(lo), float(hi)]
            coords.append(dict(parameter=name, mean=float(mean[j]), sd=float(sd[j]),
                               median=float(weights.quantiles(x[..., j].ravel(), w.ravel(), (.5,))[0]),
                               mean_shift_baseline_sd=float(delta[j]/base_sd[j]),
                               paired_shift_mcse_baseline_sd=float(error[j]/base_sd[j]), intervals=intervals))
        rows.append(dict(case=case['name'], reliability_screen_passed=all(gates.values()), gates=gates,
                         overlap_pooled_and_chains=overlap, max_weight_relative_to_uniform=largest,
                         coordinates=coords))
    return dict(baseline_mean=base_mean.tolist(), baseline_sd=base_sd.tolist(), cases=rows)

def prepare(root, planpath, sources):
    if any(p.name != 'RUN.lock' for p in root.iterdir()):
        raise ValueError('Preparation requires an empty output directory')
    plan = json.loads(planpath.read_text())
    if len(sources) != len(plan['datasets']) or len({p.resolve() for p in sources}) != len(sources):
        raise ValueError('Supply all declared datasets in order')
    (root/'plan.json').write_bytes(planpath.read_bytes())
    manifest = dict(plan_sha256=sha(root/'plan.json'), sources={}, analysis_sha256=sha(Path(__file__)),
                    weights_sha256=sha(WEIGHTS), status='Frozen before sensitivity evaluation')
    for name, source in zip(plan['datasets'], sources):
        source = source.resolve()
        dest = root/'inputs'/name
        dest.mkdir(parents=True)
        files = ['plan.json', 'posterior-summary.json', 'target.json', 'completed.json']
        mapping = {'target.json': 'ensemble0/target.json'}
        files += [f'joint.chain{e}.csv' for e in range(plan['chains'])]
        files += [f'CURRENT{e}' for e in range(plan['chains'])]
        mapping.update({f'CURRENT{e}': f'ensemble{e}/CURRENT' for e in range(plan['chains'])})
        hashes = {}
        for filename in files:
            src = source/mapping.get(filename, filename)
            before = sha(src)
            shutil.copyfile(src, dest/filename)
            if sha(src) != before or sha(dest/filename) != before:
                raise ValueError('Source changed while freezing inputs')
            hashes[filename] = before
        # Verify the same generating target for all chains without inspecting its truth.
        for e in range(plan['chains']):
            if sha(source/f'ensemble{e}/target.json') != hashes['target.json']:
                raise ValueError('Different targets across chains')
        source_plan = json.loads((dest/'plan.json').read_text())
        if source_plan['samples'] != plan['samples'] or source_plan['ensembles'] != plan['chains']:
            raise ValueError('Unexpected source budget')
        manifest['sources'][name] = dict(path=str(source), files=hashes, mapping=mapping)
    write_json(root/'FROZEN.json', manifest)
    print('Frozen plan, analysis and inputs; no sensitivity estimates evaluated')

def analyze(root):
    frozen = json.loads((root/'FROZEN.json').read_text())
    if sha(root/'plan.json') != frozen['plan_sha256'] or sha(Path(__file__)) != frozen['analysis_sha256'] or sha(WEIGHTS) != frozen['weights_sha256']:
        raise ValueError('Frozen plan or analysis changed')
    plan = json.loads((root/'plan.json').read_text())
    reports = {}
    arrays = {}
    for name in plan['datasets']:
        item = frozen['sources'][name]
        dest = root/'inputs'/name
        for filename, digest in item['files'].items():
            if sha(dest/filename) != digest or sha(Path(item['path'])/item['mapping'].get(filename, filename)) != digest:
                raise ValueError('Frozen input/source changed: '+name+'/'+filename)
        post = json.loads((dest/'posterior-summary.json').read_text())
        chains = []
        for e in range(plan['chains']):
            file = dest/f'joint.chain{e}.csv'
            if file.read_text().splitlines()[0].split(',') != ['draw', *plan['parameters']]:
                raise ValueError('Unexpected monitor layout')
            rows = np.loadtxt(file, delimiter=',', skiprows=1)
            if rows.shape != (plan['samples'], 1+len(plan['parameters'])) or not np.array_equal(rows[:, 0], np.arange(plan['samples'])):
                raise ValueError('Incomplete or reordered cold records')
            chains.append(rows[:, 1:])
        x = np.stack(chains)
        arrays[name] = x
        reports[name] = summarize(x, plan, post['passed'])
    report = dict(scope=plan['scope'], datasets=reports, source_inputs_unchanged=True,
                  all_reliability_screens_passed=all(c['reliability_screen_passed'] for d in reports.values() for c in d['cases']),
                  calibrated_recovery_claim=False, limits=plan['limits'])
    write_json(root/'summary.json', report)
    plot(root, plan, arrays, reports)
    print(json.dumps({name: [{k:c[k] for k in ('case', 'reliability_screen_passed', 'overlap_pooled_and_chains')} for c in d['cases']] for name, d in reports.items()}, indent=2))

def plot(root, plan, arrays, reports):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for col, name in enumerate(plan['datasets']):
        x = arrays[name]
        for k, case in enumerate(plan['cases']):
            row = reports[name]['cases'][k]
            w, _ = weights.normalized(log_ratio(x, case, plan['baseline_scalar_bounds']))
            label = case['name'] + ('' if row['reliability_screen_passed'] else ' (provisional)')
            axes[0, col].hist(x[..., 0].ravel(), bins=np.linspace(.5, 1.5, 36), weights=w.ravel(), density=True,
                              histtype='step', color=colors[k], label=label)
            if k:
                coords = row['coordinates'][:5]
                axes[1, col].errorbar(np.arange(5)+(k-2)*.12, [r['mean_shift_baseline_sd'] for r in coords],
                                      yerr=[2*r['paired_shift_mcse_baseline_sd'] for r in coords], fmt='o', ms=3,
                                      color=colors[k], label=label, capsize=2)
        axes[0, col].set(title=name, xlabel='sigma8 / reference sigma8', ylabel='Reweighted density')
        axes[0, col].legend(fontsize=7)
        axes[1, col].axhline(0, color='black', lw=.7)
        axes[1, col].set_xticks(np.arange(5), plan['parameters'][:5], rotation=35, ha='right')
        axes[1, col].set(ylabel='Mean shift / baseline SD (±2 paired MCSE)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Prior sensitivity on existing cold chains\nSame support, data and likelihood; importance estimates, not calibration')
    fig.savefig(root/'prior-sensitivity.png', dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'analyze'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--sources', type=Path, nargs='+')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            prepare(root, args.plan, args.sources or [])
        else:
            analyze(root)

if __name__ == '__main__':
    main()
