"""Report all predeclared N16 mocks, including missing and non-mixing cases."""
import argparse
import csv
import json
import math
from pathlib import Path


def wilson(successes, total):
    """95% score interval for independent datasets, not correlated field modes."""
    if not total:
        return None
    z = 1.959963984540054
    fraction = successes/total
    den = 1+z*z/total
    center = (fraction+z*z/(2*total))/den
    radius = z*math.sqrt(fraction*(1-fraction)/total+z*z/(4*total*total))/den
    return [max(0., center-radius), min(1., center+radius)]



def exact_binomial(successes, total):
    """Clopper-Pearson 95% bounds by binomial-tail inversion (small n)."""
    if not total:
        return None
    def tail(probability, first, last):
        return sum(math.comb(total, j)*probability**j*(1-probability)**(total-j)
                   for j in range(first, last+1))
    def root(first, last, increasing):
        lo, hi = 0., 1.
        for _ in range(100):
            mid = (lo+hi)/2
            value = tail(mid, first, last)
            if (value < .025) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0. if successes == 0 else root(successes, total, True),
            1. if successes == total else root(0, successes, False)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.input/'plan.json').read_text())
    assert plan['schema'] == 'FML-N16-repeated-mocks-v1'
    assert len(plan['mocks']) == 4 and [(m['ic_seed'], m['noise_seed']) for m in plan['mocks']] == [(291+100*i, 292+100*i) for i in range(4)]
    assert plan['gates'] == dict(rhat=1.05, ess_bulk=400, ess_tail=200)
    amendment_path = a.input/'scope-amendment.json'
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    if amendment:
        assert amendment['original_declared'] == 4 and amendment['effective_declared'] == 3 and amendment['skipped_ids'] == [3]
        assert amendment['sampling_settings_changed'] is False
        assert not (a.input/'mock3').exists(), 'User-skipped mock must never have started'
    effective_mocks = [m for m in plan['mocks'] if not amendment or m['id'] not in amendment['skipped_ids']]
    total = len(effective_mocks)
    commands = json.loads((a.input/'commands.json').read_text())
    if amendment:
        assert not any(r['id'] == 3 for r in commands), 'User-skipped mock was launched'

    assert len({r['id'] for r in commands}) == len(commands)
    rows = []
    for mock in effective_mocks:
        folder = a.input/mock['directory']
        row = dict(id=mock['id'], ic_seed=mock['ic_seed'], noise_seed=mock['noise_seed'],
                   complete=False, consistency_passed=False, mixing_passed=False,
                   passed=False, error=None)
        command = next((r for r in commands if r['id'] == mock['id']), None)
        path = folder/'analysis/summary.json'
        if not path.exists():
            row['error'] = 'Missing summary; native/preflight/incomplete case retained'
            row['exit_code'] = command.get('exit_code') if command else None
            failure = folder/'preflight-failure.json'
            if failure.exists():
                row['preflight_failure'] = json.loads(failure.read_text())
            rows.append(row)
            continue
        s = json.loads(path.read_text())
        caseplan = json.loads((folder/'plan.json').read_text())
        assert all(caseplan[k] == plan['protocol'][k] for k in ['n', 'm', 'steps', 'chains', 'warmup', 'draws', 'leapfrog', 'epsilon', 'target_acceptance', 'divergence_threshold', 'adapt_mass', 'predictive_stride', 'timeout_seconds'])
        assert caseplan['gates'] == plan['gates']
        assert caseplan['scheme'] == 'preconditioned'
        assert (s['problem']['ic_seed'], s['problem']['noise_seed']) == (mock['ic_seed'], mock['noise_seed'])
        assert command is not None and command.get('exit_code') == (0 if s['passed'] else 1)
        row.update(complete=True, passed=s['passed'], consistency_passed=s['consistency_passed'],
                   mixing_passed=s['mixing_passed'], diagnostic_extrema=s['diagnostic_extrema'],
                   posterior=s['posterior'], recovery=s['recovery'],
                   mean_physical_field=s['mean_physical_field'], predictive=s['predictive'],
                   divergences=sum(m['divergences'] for m in s['metadata']),
                   acceptance=[m['acceptance'] for m in s['metadata']],
                   evaluations=sum(m['evaluations'] for m in s['metadata'])+252,
                   failed_checks=[c['check'] for c in s['checks'] if not c['passed']])
        rows.append(row)
    complete = [r for r in rows if r['complete']]
    all_passed = len(complete) == len(rows) and all(r['passed'] for r in rows)
    inclusion = {}
    for name in ['A', 'N']:
        count = sum(r['recovery']['scalar_95_interval_contains_truth'][name] for r in complete)
        inclusion[name] = dict(successes=count, available=len(complete), declared=len(rows),
                               missing=len(rows)-len(complete), fraction=count/len(complete) if complete else None,
                               wilson_95=wilson(count, len(complete)), clopper_pearson_95=exact_binomial(count, len(complete)),
                               intervals_reliable=all_passed)
    summary = dict(passed=all_passed, declared=len(rows), original_declared=len(plan['mocks']), user_skipped=[m for m in plan['mocks'] if m not in effective_mocks], scope_amendment=amendment, completed=len(complete),
                   mixing_passed=sum(r['mixing_passed'] for r in rows),
                   consistency_passed=sum(r['consistency_passed'] for r in rows),
                   mocks=rows, scalar_interval_inclusion=inclusion,
                   limits=[
                       'Fresh datasets at fixed true A=0 and noise power=1e-4; limited repeated-data recovery, not prior-predictive SBC.',
                       'Every declared dataset and failure remains visible. Missing results are not successes; coverage of available intervals is unreliable if any case fails.',
                       'Wilson intervals use datasets as units. A and N estimates on the same datasets are correlated; N and noise_power interval inclusion are the same event.',
                       '122 field modes within each mock are correlated, so their interval-inclusion counts are not pooled as independent binomial trials.',
                       'Same-model N16 only; no N32 mixing, resolution convergence or broader cosmology/noise-regime claim.',
                       'HMC seeds are held fixed across datasets. Wilson intervals describe dataset-to-dataset variation and omit residual Monte Carlo error.'])
    if amendment:
        summary['limits'].append('Original four-mock scope reduced to three at user request after two results were known and the third was running. The fourth is user-skipped, not missing or failed. Wilson and added exact binomial intervals are descriptive fixed-n nominal bounds, not an optional-stopping coverage guarantee. Exact bounds supplement the predeclared Wilson approximation because only three mocks remain.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (a.output/'mocks.csv').open('w') as f:
        keys = ['id', 'ic_seed', 'noise_seed', 'complete', 'consistency_passed', 'mixing_passed', 'passed', 'max_rhat', 'min_bulk_ess', 'min_tail_ess', 'A_median', 'A_lower', 'A_upper', 'noise_median', 'noise_lower', 'noise_upper', 'field_correlation', 'field_relative_error', 'mode_intervals_containing_truth', 'error']
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n');w.writeheader()
        for r in rows:
            item = {k:r[k] for k in keys if k in r}
            if r['complete']:
                item.update(r['diagnostic_extrema'])
                for key, name in [('A', 'A'), ('noise', 'noise_power')]:
                    v = r['posterior'][name]
                    item.update({key+'_median':v['median'], key+'_lower':v['quantile_025'], key+'_upper':v['quantile_975']})
                item.update(field_correlation=r['mean_physical_field']['correlation'], field_relative_error=r['mean_physical_field']['relative_error'], mode_intervals_containing_truth=r['recovery']['active_mode_95_interval_contains_truth_count'])
            w.writerow(item)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, name, true, scale, label in [(axes[0,0], 'A', 0., 1., 'Log initial-power amplitude A'), (axes[0,1], 'noise_power', 1e-4, 1e-4, 'Noise power / injected')]:
        for r in complete:
            v = r['posterior'][name];median=v['median']/scale
            ax.errorbar(r['id']+1, median, yerr=[[(v['median']-v['quantile_025'])/scale],[(v['quantile_975']-v['median'])/scale]], fmt='o', color='steelblue' if r['passed'] else 'darkorange', capsize=4)
        ax.axhline(true/scale, color='black', ls='--', label='Injected truth');ax.set(xticks=list(range(1,total+1)), xlabel='Predeclared mock', ylabel=label, title='Median and marginal 95% interval');ax.legend(fontsize=8)
    ax = axes[1,0]
    for r in complete:
        ax.scatter(r['id']+1,r['mean_physical_field']['correlation'],color='steelblue' if r['passed'] else 'darkorange')
    ax.set(xticks=list(range(1,total+1)),xlabel='Predeclared mock',ylabel='Initial-field mean correlation with truth',ylim=(0,1))
    ax=axes[1,1];ax.axis('off')
    text=f"Completed: {len(complete)}/{total}; all checks passed: {all_passed}\nMixing passes: {summary['mixing_passed']}/{total}\n\n95% interval truth inclusion by dataset:\n"
    for name in ['A','N']:
        v=inclusion[name];ci=v['clopper_pearson_95'];text+=f"{name}: {v['successes']}/{v['available']} available"
        if ci:text+=f"; exact 95% [{ci[0]:.2f}, {ci[1]:.2f}]"
        text+='\n'
    text+=f'\n{total} mocks: substantial coverage uncertainty\nFixed truth; not simulation-based calibration\nMissing/failed mocks retained; no replacements\nOrange denotes a failed case'
    ax.text(0,.98,text,va='top',fontsize=10,linespacing=1.6)
    fig.suptitle('Repeated same-model N16 mocks: unchanged four-chain sampler')
    fig.savefig(a.output/'overview.png',dpi=160);fig.savefig(a.output/'overview.pdf');plt.close(fig)
    print(json.dumps(dict(passed=all_passed,completed=len(complete),mixing_passed=summary['mixing_passed'],scalar_interval_inclusion=inclusion),indent=2))
    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
