#!/usr/bin/env python3
"""Write BaccoReference.h: predictions of the baccoemu Python package (pip install baccoemu==2.3.0, with the weights
extracted into the package folder as it does on first use) for test_bacco.cpp.

    python3 make_bacco_reference.py > BaccoReference.h

The networks are evaluated in float64 (baccoemu itself casts weights and inputs to float32, which gives ~1e-6
relative noise): the weights are float32 numbers, so this is the same network evaluated exactly, and the C++ port
(double) should agree to ~1e-12. With --float32 the package is run unpatched, to see the float32 noise.

To keep the file small (CI allows at most 1 MB per file) a case stores at most MAX_KEPT of its expected values: every
stride-th value of the flattened output, with the full length so that the test still checks it. The k values the
package returns are stored only for the bispectrum (its triangles), the only case where the test compares them.
"""
import re
import sys
import numpy as np
import jax

FLOAT32 = '--float32' in sys.argv
if not FLOAT32:
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import baccoemu  # noqa: E402
from baccoemu import utils  # noqa: E402

if not FLOAT32:
    def predict64(self, x):
        a = jnp.asarray(np.asarray(x, dtype=np.float64))
        for w, b in self.params[:-1]:
            a = jnp.maximum(0, jnp.matmul(a, jnp.asarray(w, dtype=jnp.float64)) + jnp.asarray(b, dtype=jnp.float64))
        w, b = self.params[-1]
        return jnp.matmul(a, jnp.asarray(w, dtype=jnp.float64)) + jnp.asarray(b, dtype=jnp.float64)
    utils.load_model.predict = predict64

NAN = float('nan')
MAX_KEPT = 100
FIELDS = ['Omega_cold', 'Omega_m', 'Omega_b', 'sigma8_cold', 'A_s', 'h', 'n_s', 'M_nu', 'w0', 'wa',
          'M_c', 'eta', 'beta', 'M1_z0_cen', 'theta_out', 'theta_inn', 'M_inn']
PY = dict(Omega_cold='omega_cold', Omega_m='omega_matter', Omega_b='omega_baryon', sigma8_cold='sigma8_cold',
          A_s='A_s', h='hubble', n_s='ns', M_nu='neutrino_mass', w0='w0', wa='wa', M_c='M_c', eta='eta',
          beta='beta', M1_z0_cen='M1_z0_cen', theta_out='theta_out', theta_inn='theta_inn', M_inn='M_inn')


def f32(x):
    return float(np.float32(x))


def arr(v):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return '{' + ', '.join('NAN' if np.isnan(x) else repr(float(x)) for x in v) + '}'


def make64(emu_model):
    """Burger 2025 networks: float64 weights (the inputs are float32 numbers in the tests)"""
    if not FLOAT32:
        e = emu_model.__self__
        e.weights = [[np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)] for w, b in e.weights]


emus = {}
for nl in ['Angulo2021', 'Arico2023']:
    for bar in ['Arico2021', 'Burger2025']:
        for sb in [False, True]:
            m = baccoemu.Matter_powerspectrum(nonlinear_model_name=nl, baryonic_model_name=bar, smeared_bao=sb,
                                              no_wiggles=not sb, verbose=False)
            if bar == 'Burger2025':
                make64(m.emulator['baryon']['model'])
            emus[(nl, bar, sb)] = m
bk = baccoemu.Matter_bispectrum(verbose=False)
import contextlib  # noqa: E402
with contextlib.redirect_stdout(sys.stderr):  # some loaders print even with verbose=False
    lb = baccoemu.Lbias_expansion(verbose=False)
    lbz = baccoemu.Lbias_expansion_RSD(verbose=False)
make64(bk.emulator['baryon']['model'])

# kind, options, params, a, cold, k, k_lin, pk_lin, expected, k_out, error
cases = []


def add(kind, opts, P, a, cold, k, fn, k_lin=None, pk_lin=None, k1=None, k2=None, k3=None, extra=None):
    err, res, kout = '', [], []
    try:
        out = fn()
        if isinstance(out, tuple):
            kout, res = out[0], out[1]
        else:
            res = out
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        m = re.search(r'Param (\w+)=', msg)
        err = m.group(1) if m else 'error'
    cases.append(dict(kind=kind, opts=opts, P=P, a=a, cold=cold, k=[] if k is None else k, k_lin=k_lin or [],
                      pk_lin=pk_lin or [], k1=k1 or [], k2=k2 or [], k3=k3 or [], extra=extra or [],
                      expected=np.concatenate([np.ravel(np.asarray(r, dtype=float)) for r in res])
                      if isinstance(res, (tuple, list)) and len(res) and np.ndim(res[0]) > 0 else np.ravel(res),
                      kout=np.ravel(np.asarray(kout, dtype=float)), err=err))


def coords(P, a):
    d = {PY[f]: v for f, v in P.items()}
    d['expfactor'] = a
    return d


lcdm = dict(Omega_cold=0.31, Omega_b=0.049, sigma8_cold=0.81, h=0.67, n_s=0.965, M_nu=0.0, w0=-1.0, wa=0.0)
nu = dict(Omega_m=0.32, Omega_b=0.05, A_s=2.1e-9, h=0.7, n_s=0.97, M_nu=0.15, w0=-0.9, wa=0.1)
wide = dict(Omega_cold=0.42, Omega_b=0.035, sigma8_cold=1.05, h=0.55, n_s=0.88, M_nu=0.3, w0=-1.25, wa=-0.3)
bar = dict(M_c=14.0, eta=-0.3, beta=-0.22, M1_z0_cen=10.5, theta_out=0.25, theta_inn=-0.86, M_inn=13.4)
kcust = list(np.geomspace(2e-3, 4.5, 37))
kbar = list(np.geomspace(5e-3, 6.0, 23))

E = emus[('Angulo2021', 'Arico2021', False)]

# sigma8 / A_s
for P in [nu, dict(nu, M_nu=0.0, w0=-1.0, wa=0.0)]:
    for cold in [True, False]:
        add('sigma8', '', P, 1.0, cold, None, lambda P=P, c=cold: E.get_sigma8(cold=c, **coords(P, 1.0)))
        add('sigma12', '', P, 1.0, cold, None, lambda P=P, c=cold: E.get_sigma12(cold=c, **coords(P, 1.0)))
for P in [lcdm, wide]:
    add('A_s', '', P, 1.0, True, None, lambda P=P: E.get_A_s(**coords(P, 1.0)))

# linear, no-wiggles, smeared BAO
for P in [lcdm, nu, wide]:
    for a in [1.0, 0.43]:
        for cold in [True, False]:
            add('linear', '', P, a, cold, None, lambda P=P, a=a, c=cold: E.get_linear_pk(cold=c, **coords(P, a)))
            add('linear', '', P, a, cold, kcust,
                lambda P=P, a=a, c=cold: E.get_linear_pk(cold=c, k=np.array(kcust), **coords(P, a)))
        add('no_wiggles', '', P, a, True, None, lambda P=P, a=a: E.get_no_wiggles_pk(**coords(P, a)))
        add('no_wiggles', '', P, a, True, kcust,
            lambda P=P, a=a: E.get_no_wiggles_pk(k=np.array(kcust), **coords(P, a)))
for P in [lcdm, nu]:
    S = emus[('Angulo2021', 'Arico2021', True)]
    add('smeared_bao', 'smeared_bao', P, 0.8, True, kcust,
        lambda P=P: S.get_smeared_bao_pk(k=np.array(kcust), **coords(P, 0.8)))
    add('smeared_bao', 'smeared_bao', P, 0.8, True, None, lambda P=P: S.get_smeared_bao_pk(**coords(P, 0.8)))

# nonlinear
for nl, Ps in [('Angulo2021', [lcdm, nu]), ('Arico2023', [lcdm, nu, wide])]:
    for sb in [False, True]:
        M = emus[(nl, 'Arico2021', sb)]
        opts = nl + (' smeared_bao' if sb else '')
        for P in Ps:
            for a in [1.0, 0.62]:
                for cold in [True, False]:
                    add('boost', opts, P, a, cold, None,
                        lambda M=M, P=P, a=a, c=cold: M.get_nonlinear_boost(cold=c, **coords(P, a)))
                    add('boost', opts, P, a, cold, kcust,
                        lambda M=M, P=P, a=a, c=cold: M.get_nonlinear_boost(cold=c, k=np.array(kcust),
                                                                             **coords(P, a)))
                    add('pk_nl', opts, P, a, cold, kcust,
                        lambda M=M, P=P, a=a, c=cold: M.get_nonlinear_pk(cold=c, k=np.array(kcust),
                                                                          **coords(P, a)))

# nonlinear with a given linear P(k) (a toy shape, on a wide k grid)
klin = list(np.geomspace(1e-4, 50.0, 400))
pklin = list(2e4 * (np.array(klin) / 0.02) / (1 + (np.array(klin) / 0.02) ** 2.6)
             * (1 + 0.04 * np.sin(np.array(klin) * 105.0) * np.exp(-(np.array(klin) / 0.25) ** 2)))
M = emus[('Arico2023', 'Arico2021', False)]
add('boost', 'Arico2023', lcdm, 0.9, True, kcust,
    lambda: M.get_nonlinear_boost(k=np.array(kcust), k_lin=np.array(klin), pk_lin=np.array(pklin), **coords(lcdm, 0.9)),
    k_lin=klin, pk_lin=pklin)

# baryons
for P in [lcdm, nu]:
    for a in [1.0, 0.55]:
        add('baryon', 'Arico2021', dict(P, **bar), a, True, None,
            lambda P=P, a=a: E.get_baryonic_boost(**coords(dict(P, **bar), a)))
        add('baryon', 'Arico2021', dict(P, **bar), a, True, kbar,
            lambda P=P, a=a: E.get_baryonic_boost(k=np.array(kbar), **coords(dict(P, **bar), a)))
add('pk_nl_baryons', 'Arico2023 Arico2021', dict(lcdm, **bar), 0.7, True, kbar,
    lambda: emus[('Arico2023', 'Arico2021', False)].get_nonlinear_pk(baryonic_boost=True, k=np.array(kbar),
                                                                      **coords(dict(lcdm, **bar), 0.7)))

# Burger 2025 (inputs that are float32 numbers: the package evaluates in float32 inputs)
bur = {f: f32(v) for f, v in dict(Omega_cold=0.30, Omega_b=0.048, sigma8_cold=0.80, h=0.68, n_s=0.96, M_nu=0.0,
                                   w0=-1.0, wa=0.0, M_c=13.5, eta=-0.25, beta=-0.1, M1_z0_cen=11.0,
                                   theta_inn=-1.0).items()}
kb32 = [f32(x) for x in np.geomspace(0.02, 15.0, 29)]
B = emus[('Angulo2021', 'Burger2025', False)]
for a in [1.0, 0.5]:
    a32 = f32(a)
    add('baryon', 'Burger2025', bur, a32, True, kb32,
        lambda a=a32: B.get_baryonic_boost(k=np.array(kb32), **coords(bur, a)))
    add('baryon', 'Burger2025', bur, a32, True, None, lambda a=a32: B.get_baryonic_boost(**coords(bur, a)))
rng = np.random.default_rng(3)
K = [[f32(x) for x in rng.uniform(0.05, 8.0, 25)] for _ in range(3)]
burbk = {f: v for f, v in bur.items() if f not in ('h', 'n_s', 'M_nu', 'w0', 'wa')}
add('bispectrum', '', burbk, f32(0.75), True, None,
    lambda: bk.get_baryonic_boost(k1=np.array(K[0]), k2=np.array(K[1]), k3=np.array(K[2]), check_k_extrapolation=False,
                                  **coords(burbk, f32(0.75))), k1=K[0], k2=K[1], k3=K[2])

# Lagrangian bias expansion, real space (bias = b1, b2, bs2, blaplacian)
klb = list(np.geomspace(0.012, 0.7, 23))
bias = [0.4, -0.3, 0.2, 1.5]
for P in [lcdm, nu]:
    for a in [1.0, 0.6]:
        add('lbias_lpt', '', P, a, True, None, lambda P=P, a=a: lb.get_lpt_pk(**coords(P, a)))
        add('lbias_lpt', '', P, a, True, klb, lambda P=P, a=a: lb.get_lpt_pk(k=np.array(klb), **coords(P, a)))
        add('lbias_bao', '', P, a, True, klb, lambda P=P, a=a: lb.get_smeared_bao_pk(k=np.array(klb), **coords(P, a)))
        add('lbias_pnn', '', P, a, True, None, lambda P=P, a=a: lb.get_nonlinear_pnn(**coords(P, a)))
        add('lbias_pnn', '', P, a, True, klb, lambda P=P, a=a: lb.get_nonlinear_pnn(k=np.array(klb), **coords(P, a)))
        add('lbias_galaxy', '', P, a, True, klb,
            lambda P=P, a=a: list(lb.get_galaxy_real_pk(bias=bias, k=np.array(klb), **coords(P, a))[1:]), extra=bias)
add('lbias_pnn', '', dict(lcdm, sigma8_cold=0.95), 1.0, True, None,
    lambda: lb.get_nonlinear_pnn(**coords(dict(lcdm, sigma8_cold=0.95), 1.0)))

# Lagrangian bias expansion, redshift space (extra = bias, f_sat, lambda_FoG, epsilon_1, epsilon_2, epsilon_3, nbar)
kz = list(np.geomspace(0.012, 0.6, 19))
for P in [lcdm, nu]:
    for a in [1.0, 0.7]:
        add('rsd_lpt', '', P, a, True, None, lambda P=P, a=a: lbz.get_lpt_pk(**coords(P, a)))
        add('rsd_lpt', '', P, a, True, kz, lambda P=P, a=a: lbz.get_lpt_pk(k=np.array(kz), **coords(P, a)))
        add('rsd_bao', '', P, a, True, kz, lambda P=P, a=a: lbz.get_smeared_bao_pk(k=np.array(kz), **coords(P, a)))
        add('rsd_pnn', '', P, a, True, None, lambda P=P, a=a: lbz.get_nonlinear_pnn(**coords(P, a)))
        add('rsd_pnn', '', P, a, True, kz, lambda P=P, a=a: lbz.get_nonlinear_pnn(k=np.array(kz), **coords(P, a)))
        for ex in [[0.15, 1.8, NAN, NAN, NAN, NAN], [0.1, 2.5, 1.0, -0.5, NAN, 3e-4], [0.1, 2.5, 1.0, -0.5, 0.7, 3e-4]]:
            kw = dict(zip(['f_sat', 'lambda_FoG', 'epsilon_1', 'epsilon_2', 'epsilon_3', 'mean_num_dens'],
                          [None if np.isnan(x) else x for x in ex]))
            add('rsd_galaxy', '', P, a, True, kz,
                lambda P=P, a=a, kw=kw: list(lbz.get_galaxy_pk(bias=bias, k=np.array(kz), **kw, **coords(P, a))[1:]),
                extra=bias + ex)
        kw = dict(f_sat=0.1, lambda_FoG=2.5, epsilon_1=1.0, epsilon_2=-0.5, mean_num_dens=3e-4)
        add('rsd_galaxy_2d', '', P, a, True, kz,
            lambda P=P, a=a: lbz.get_galaxy_2Dpk(bias=bias, k=np.array(kz), **kw, **coords(P, a))[2],
            extra=bias + [0.1, 2.5, 1.0, -0.5, NAN, 3e-4])

# out of range: the parameter named in the error
add('linear', '', dict(lcdm, Omega_cold=0.8), 1.0, True, None, lambda: E.get_linear_pk(**coords(dict(lcdm, Omega_cold=0.8), 1.0)))
add('boost', 'Angulo2021', dict(lcdm, sigma8_cold=0.6), 1.0, True, None,
    lambda: E.get_nonlinear_boost(**coords(dict(lcdm, sigma8_cold=0.6), 1.0)))
add('boost', 'Angulo2021', lcdm, 0.3, True, None, lambda: E.get_nonlinear_boost(**coords(lcdm, 0.3)))
add('rsd_pnn', '', dict(lcdm, n_s=1.05), 1.0, True, None,
    lambda: lbz.get_nonlinear_pnn(**coords(dict(lcdm, n_s=1.05), 1.0)))
add('baryon', 'Arico2021', dict(lcdm, **dict(bar, eta=0.9)), 1.0, True, None,
    lambda: E.get_baryonic_boost(**coords(dict(lcdm, **dict(bar, eta=0.9)), 1.0)))

print('// Written by make_bacco_reference.py (baccoemu %s%s). Do not edit.' % (baccoemu.__version__ if hasattr(
    baccoemu, '__version__') else '2.3.0', ', float32' if FLOAT32 else ', networks evaluated in float64'))
print('#ifndef FML_EMULATOR_BACCO_REFERENCE_HEADER\n#define FML_EMULATOR_BACCO_REFERENCE_HEADER\n')
print('#include <cmath>\n#include <string>\n#include <vector>\n')
print('namespace BaccoReference {')
print('    struct Case {\n        std::string kind, options;\n        std::vector<double> params; // ' +
      ', '.join(FIELDS) + ' (NaN: not given)')
print('        double a;\n        bool cold;\n        std::vector<double> k, k_lin, pk_lin, k1, k2, k3, extra, expected, k_out;'
      '\n        std::string error; // the parameter out of range, or empty'
      '\n        size_t size, stride; // expected holds every stride-th of the size values of the output\n    };')
print('    // clang-format off')
print('    inline const std::vector<Case> cases = {')
for c in cases:
    p = [c['P'].get(f, NAN) for f in FIELDS]
    n = len(c['expected'])
    stride = max(1, -(-n // MAX_KEPT))
    kout = c['kout'] if c['kind'] == 'bispectrum' else []
    ps = '{' + ', '.join('NAN' if np.isnan(x) else repr(float(x)) for x in p) + '}'
    print('        {"%s", "%s", %s, %r, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "%s", %d, %d},' % (
        c['kind'], c['opts'], ps, float(c['a']), 'true' if c['cold'] else 'false', arr(c['k']), arr(c['k_lin']),
        arr(c['pk_lin']), arr(c['k1']), arr(c['k2']), arr(c['k3']), arr(c['extra']), arr(c['expected'][::stride]),
        arr(kout), c['err'], n, stride))
print('    };\n    // clang-format on\n} // namespace BaccoReference\n\n#endif')
print('%d cases, %d errors' % (len(cases), sum(1 for c in cases if c['err'])), file=sys.stderr)
