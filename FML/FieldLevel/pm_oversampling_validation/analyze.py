"""Bounded dense-particle PM convergence; finite references, unchanged gates."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
gates = []
def check(valid, text, kind='consistency'):
    gates.append(dict(check=text, passed=bool(valid), kind=kind))
def load(prefix, suffix):
    return np.atleast_1d(np.loadtxt(str(prefix)+'.'+suffix+'.csv', delimiter=','))
f = np.fft.fftfreq(8)*8
z = np.fft.rfftfreq(8)*8
x, y, zz = np.meshgrid(f, f, z, indexing='ij')
q = x*x+y*y+zz*zz
w = np.broadcast_to(np.where((z == 0) | (z == 4), 1., 2.), q.shape)
mask = (q > 0) & (q <= 5)
dof = float(w[mask].sum())
def norm(v):
    return float(np.sqrt(np.sum(w[mask]*np.abs(v[mask])**2)))
cases, families, fd = {}, {}, {}
reference = load(a.input/'science/data-reference', 'initial').ravel()
for c in json.loads((a.input/'plan.json').read_text()):
    key = f"{c['n']}-{c['m']}-{c['steps']}"
    prefix = a.input/('science/case-'+key)
    initial = load(prefix, 'initial').ravel()
    gradient = load(prefix, 'gradient').ravel()
    v = load(prefix, 'modes').reshape(8, 8, 5, 2)
    prediction = v[..., 0]+1j*v[..., 1]
    check(initial.shape == (18,) and np.max(np.abs(initial-reference)) < 2e-12,
          key+': matched physical initial coefficients')
    check(gradient.shape == (20,) and np.all(np.isfinite(gradient)) and
          np.all(np.isfinite(prediction)) and np.max(np.abs(prediction[~mask])) == 0,
          key+': finite common-band outputs/gradients')
    cases[key] = dict(c, prediction=prediction, gradient=gradient)
    for family in c['families']:
        families.setdefault(family, []).append(key)
    if Path(str(prefix)+'.fd.csv').exists():
        values = np.atleast_1d(np.genfromtxt(str(prefix)+'.fd.csv', delimiter=',', names=True))
        fd[key] = {}
        for probe in [0, 1]:
            r = values[values['probe'] == probe]
            error = np.abs(r['finite_difference']-r['analytic'])/np.maximum(1., np.abs(r['analytic']))
            check(len(r) == 8 and np.all(np.isfinite(error)) and np.min(error[-4:]) < 2e-5
                  and np.sum(error[-4:] < 1e-4) >= 2, key+f': refined FD probe {probe}')
            fd[key][str(probe)] = dict(h=r['h'].tolist(), scaled_error=error.tolist())
check(len(fd) == 1, 'one oversampled field/scalar directional-check fixture')
convergence = {}
for family, keys in families.items():
    variable = 'm' if family == 'dense-force' else ('steps' if family == 'time' else 'n')
    keys.sort(key=lambda k: cases[k][variable])
    fine = cases[keys[-1]]
    rows = []
    for key in keys:
        c = cases[key]
        delta = c['prediction']-fine['prediction']
        rows.append(dict(key=key, resolution=c[variable],
                         field_error=norm(delta)/max(norm(fine['prediction']), 1e-30),
                         gradient_error=float(np.linalg.norm(c['gradient']-fine['gradient'])/
                                              max(np.linalg.norm(fine['gradient']), 1e-30)),
                         noise_rms_units=norm(delta)/np.sqrt(1e-4*dof)))
    check(rows[-2]['field_error'] < rows[0]['field_error'], family+': improving field', 'accuracy')
    check(rows[-2]['gradient_error'] < rows[0]['gradient_error'], family+': improving gradient', 'accuracy')
    check(rows[-2]['field_error'] < .05 and rows[-2]['gradient_error'] < .15,
          family+': finest field <5% and gradient <15%', 'accuracy')
    check(rows[-2]['noise_rms_units'] < .1, family+': finest discrepancy <0.1 noise SD', 'accuracy')
    convergence[family] = dict(variable=variable, reference=keys[-1], rows=rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
titles = {'dense-force': 'Force mesh: N=64, 16 steps\nreference M=64',
          'particles-M16': 'Particles: M=16, 16 steps\nreference N=64',
          'particles-M32': 'Particles: M=32, 16 steps\nreference N=64',
          'time': 'Time: N=32, M=16\nreference 32 steps'}
fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout='constrained')
for j, (family, c) in enumerate(convergence.items()):
    rows = c['rows'][:-1]
    xx = [r['resolution'] for r in rows]
    for i, metric, threshold, label in [(0, 'field_error', .05, 'Relative density difference'),
                                        (1, 'gradient_error', .15, 'Relative projected gradient difference')]:
        ax = axes[i, j]
        ax.loglog(xx, [r[metric] for r in rows], 'o-')
        ax.axhline(threshold, color='gray', ls=':', label=f'{100*threshold:g}% criterion')
        ax.set(title=titles[family], xlabel={'m': 'Force mesh M', 'n': 'Particle/IC lattice N', 'steps': 'Time steps'}[c['variable']], ylabel=label)
        ax.set_xticks(xx, [str(v) for v in xx])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
fig.suptitle('Matched band-limited ICs; unchanged CIC/spectral force; separate finite references')
fig.savefig(a.output/'convergence.png', dpi=170)
fig.savefig(a.output/'convergence.pdf')
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for key, probes in fd.items():
    for probe, r in probes.items():
        ax.loglog(r['h'], np.maximum(r['scaled_error'], 1e-16), 'o-', label='field' if probe == '0' else 'field + scalars')
ax.axhline(2e-5, color='gray', ls=':')
ax.set(title='Oversampled adjoint: N=32, M=16, 16 steps', xlabel='Central-difference h', ylabel='Scaled directional error')
ax.legend()
fig.savefig(a.output/'gradients.png', dpi=170)
plt.close(fig)
summary = dict(passed=all(g['passed'] for g in gates),
               consistency_passed=all(g['passed'] for g in gates if g['kind'] == 'consistency'),
               gates=gates, convergence=convergence, finite_differences=fd,
               limits=['One weakly nonlinear realization, IC k<10 and observed k<15; no shell-crossing or high-cutoff claim.',
                       'Each family uses its own finite reference, not an exact continuum solution.',
                       'No MCMC or repeated-mock coverage study in this follow-up; no force-law change.'])
(a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with (a.output/'convergence.csv').open('w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['family', 'case', 'field_error', 'gradient_error', 'noise_rms_units'])
    for family, c in convergence.items():
        for r in c['rows']:
            writer.writerow([family, r['key'], r['field_error'], r['gradient_error'], r['noise_rms_units']])
print(json.dumps(dict(passed=summary['passed'], consistency_passed=summary['consistency_passed'],
                      failed=[g['check'] for g in gates if not g['passed']]), indent=2))
if not summary['passed']:
    raise SystemExit(1)
