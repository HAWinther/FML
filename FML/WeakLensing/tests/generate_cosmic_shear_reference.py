"""Regenerate ReferenceCosmicShear.h for test_cosmic_shear.cpp (not needed to run the tests).

* Tomographic shear C_ell (GG + NLA, photo-z shifts) and xi_+- from CCL (pyccl 3.3.6), for a toy analytic
  P(k, z) given to CCL as a dense Pk2D table, Smail-type n(z) and CCL's chi(z) and D(z) (tabulated here, so
  the test checks the projection code with identical inputs).
* Log-COSEBIs following Schneider, Eifler & Krause 2010 literally (monomial coefficients, eqs. 4, 31-35;
  T_- from eq. 3) in 60-digit arithmetic (mpmath 1.3.0).
"""
from pathlib import Path
import numpy as np
import pyccl as ccl
import mpmath as mp

assert ccl.__version__ == '3.3.6' and mp.__version__ == '1.3.0'
h = 0.7
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=h, n_s=0.96, sigma8=0.8, m_nu=0.0)
cosmo.cosmo.gsl_params.INTEGRATION_LIMBER_EPSREL = 1e-7
cosmo.cosmo.gsl_params.INTEGRATION_DISTANCE_EPSREL = 1e-10
cosmo.cosmo.gsl_params.ODE_GROWTH_EPSREL = 1e-10
cosmo.cosmo.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False
cosmo.cosmo.gsl_params.NZ_NORM_SPLINE_INTEGRATION = False
cosmo.cosmo.gsl_params.INTEGRATION_EPSREL = 1e-8


def pk_h(k, z):  # k [h/Mpc], P [(Mpc/h)^3]; the same formula is in test_cosmic_shear.cpp
    g = (1 + z) ** -0.8
    x = k / 0.02
    return g * g * 2e4 * x ** 0.96 / (1 + x * x) ** 1.9 * (1 + g * g * (k / 0.5) ** 1.5)


a_arr = np.linspace(1 / 8.0, 1.0, 500)
lk_arr = np.linspace(np.log(1e-5), np.log(2e3), 3000)
K, A = np.meshgrid(np.exp(lk_arr), a_arr)
pk2d = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=np.log(pk_h(K / h, 1 / A - 1) / h ** 3), is_logp=True)

zt = np.linspace(0, 7, 701)
chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + zt)) * h
D = ccl.growth_factor(cosmo, 1 / (1 + zt))
omega_m = cosmo['Omega_m']
C1rho = 5e-14 * ccl.physical_constants.RHO_CRITICAL

z0s = [0.3, 0.55, 0.8]
dzs = [-0.02, 0.0, 0.03]


def nz(z, z0):
    z = np.asarray(z)
    return np.where(z > 0, z * z * np.exp(-np.abs(z / z0) ** 1.5), 0.0)


ia = dict(A=0.8, eta=1.2, z_pivot=0.62)
zfine = np.linspace(0, 6.1, 12201)
tr_g, tr_ia = [], []
for z0, dz in zip(z0s, dzs):
    dndz = (zfine, nz(zfine - dz, z0))
    tr_g.append(ccl.WeakLensingTracer(cosmo, dndz=dndz, n_samples=4096))
    tr_ia.append(ccl.WeakLensingTracer(cosmo, dndz=dndz, n_samples=4096,
                                       ia_bias=(zt, ia['A'] * ((1 + zt) / (1 + ia['z_pivot'])) ** ia['eta'])))
ells = np.array([2., 10., 50., 200., 1000., 5000., 20000.])
pairs = [(i, j) for i in range(3) for j in range(i, 3)]
cl_gg = [ccl.angular_cl(cosmo, tr_g[i], tr_g[j], ells, p_of_k_a=pk2d) for i, j in pairs]
cl_tot = [ccl.angular_cl(cosmo, tr_ia[i], tr_ia[j], ells, p_of_k_a=pk2d) for i, j in pairs]

# xi_+- of the IA spectra on a dense ell grid, C_ell a natural cubic spline in (ln ell, ln C) as in
# CosmicShear.h, integrated over 2 <= ell <= 1e5 with QUADPACK between the Bessel zeros. (CCL's correlation()
# extrapolates C_ell outside the given ell range, which changes xi_+ at 300' by 2% and xi_- at 1' by 10%.)
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.special import jv
ell_dense = 2.0 * 5e4 ** (np.arange(600) / 599.0)
thetas_arcmin = np.array([0.5, 1, 3, 10, 30, 100, 300.])


def hankel(spl, theta, nu):
    f = lambda l: l / (2 * np.pi) * np.exp(spl(np.log(l))) * jv(nu, l * theta)
    edges = [2.0]
    while edges[-1] < 1e5:
        edges.append(min(1e5, edges[-1] + np.pi / theta if edges[-1] >= np.pi / theta * (1 - 1e-12) else min(np.pi / theta, edges[-1] * 1.5)))
    return sum(quad(f, a, b, epsabs=0, epsrel=1e-11, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))


xi, cl_dense = [], []
for i, j in [(0, 0), (1, 2)]:
    c = ccl.angular_cl(cosmo, tr_ia[i], tr_ia[j], ell_dense, p_of_k_a=pk2d)
    spl = CubicSpline(np.log(ell_dense), np.log(c), bc_type='natural')
    xi.append(([hankel(spl, t / 60 * np.pi / 180, 0) for t in thetas_arcmin],
               [hankel(spl, t / 60 * np.pi / 180, 4) for t in thetas_arcmin]))
    cl_dense.append(c)

# Independent check of the Limber integrals for the exact (analytic) n(z), QUADPACK (CCL is ~3e-4 low at
# ell = 2e4 for the low-redshift bin)
chi_of_z = CubicSpline(zt, chi)
z_of_chi = CubicSpline(chi, zt)


def n_exact(zz, b):
    zs = zz - dzs[b]
    return zs * zs * np.exp(-abs(zs / z0s[b]) ** 1.5) if zs > 0 else 0.0


norm = [quad(lambda zz: n_exact(zz, b), 0, 7, limit=400, epsrel=1e-13)[0] for b in range(3)]


def q(c, b):
    zl = float(z_of_chi(c))
    g = quad(lambda zz: n_exact(zz, b) / norm[b] * (1 - c / chi_of_z(zz)), zl, 7, limit=400, epsrel=1e-12)[0]
    return 1.5 * omega_m / 2997.92458 ** 2 * c * (1 + zl) * g


def cl_quad(l, i, j):
    S = (l - 1) * l * (l + 1) * (l + 2) / (l + .5) ** 4
    f = lambda c: q(c, i) * q(c, j) / c ** 2 * pk_h((l + .5) / c, float(z_of_chi(c))) if (l + .5) / c < 2e3 / h else 0
    return S * quad(f, 1e-4, float(chi_of_z(6.9)), limit=1000, epsrel=1e-10, points=[10, 100, 1000, 3000])[0]


quad_cases = [(50., 0, 0), (200., 2, 2), (1000., 1, 2), (20000., 0, 0)]
quad_vals = [cl_quad(*cse) for cse in quad_cases]

# ---- log-COSEBIs, SEK10 in 60 digits
mp.mp.dps = 60
tmin = mp.mpf(0.5) / 60 * mp.pi / 180
tmax = mp.mpf(300) / 60 * mp.pi / 180
Z = mp.log(tmax / tmin)
nmax = 5


def J(k, j):
    return mp.quad(lambda z: mp.e ** (k * z) * z ** j, [0, Z / 2, Z])


Jc = {(k, j): J(k, j) for k in (1, 2, 4) for j in range(2 * nmax + 3)}
coef = []
for n in range(1, nmax + 1):
    m = n + 1  # unknowns cbar_0..cbar_n, cbar_{n+1} = 1
    M = mp.matrix(m, m)
    r = mp.matrix(m, 1)
    for j in range(m):
        M[0, j] = Jc[(2, j)]
        M[1, j] = Jc[(4, j)]
    r[0] = -Jc[(2, n + 1)]
    r[1] = -Jc[(4, n + 1)]
    for row, c in enumerate(coef):  # orthogonal to T_1..T_{n-1}
        for j in range(m):
            M[2 + row, j] = sum(c[i] * Jc[(1, i + j)] for i in range(len(c)))
        r[2 + row] = -sum(c[i] * Jc[(1, i + n + 1)] for i in range(len(c)))
    cb = list(mp.lu_solve(M, r)) + [mp.mpf(1)]
    s = sum(cb[i] * cb[j] * Jc[(1, i + j)] for i in range(n + 2) for j in range(n + 2))
    N = mp.sqrt((mp.e ** Z - 1) / s)
    coef.append([N * x for x in cb])
roots = [sorted([mp.re(x) for x in mp.polyroots(c[::-1], maxsteps=500, extraprec=200)]) for c in coef]
norms = [c[-1] for c in coef]


def tplus(n, theta):
    z = mp.log(theta / tmin)
    return sum(c * z ** j for j, c in enumerate(coef[n - 1]))


def tminus(n, theta):
    a = mp.quad(lambda t: t * tplus(n, t) * (4 / theta ** 2 - 12 * t * t / theta ** 4), [tmin, theta])
    return tplus(n, theta) + a


cos_theta_arcmin = [0.5, 0.8, 5.0, 47.0, 299.0]
tvals = [[(float(tplus(n, mp.mpf(t) / 60 * mp.pi / 180)), float(tminus(n, mp.mpf(t) / 60 * mp.pi / 180)))
          for t in cos_theta_arcmin] for n in range(1, nmax + 1)]


def arr(name, values):
    return f'inline const double {name}[] = {{' + ', '.join(format(float(v), '.17g') for v in values) + '};\n'


text = '// Generated by generate_cosmic_shear_reference.py (pyccl 3.3.6, mpmath 1.3.0); do not edit.\n'
text += '#ifndef FML_WEAK_LENSING_COSMIC_SHEAR_REFERENCE\n#define FML_WEAK_LENSING_COSMIC_SHEAR_REFERENCE\n'
text += f'inline const double ref_h = {h!r}, ref_omega_m = {float(omega_m)!r}, ref_C1rho = {float(C1rho)!r};\n'
text += f'inline const double ref_ia_A = {ia["A"]!r}, ref_ia_eta = {ia["eta"]!r}, ref_ia_zpiv = {ia["z_pivot"]!r};\n'
text += arr('ref_z', zt) + arr('ref_chi', chi) + arr('ref_D', D)
text += arr('ref_z0', z0s) + arr('ref_dz', dzs) + arr('ref_ells', ells)
text += 'inline const double ref_cl_gg[6][7] = {' + ','.join('{' + ','.join(format(v, '.17g') for v in c) + '}' for c in cl_gg) + '};\n'
text += 'inline const double ref_cl_tot[6][7] = {' + ','.join('{' + ','.join(format(v, '.17g') for v in c) + '}' for c in cl_tot) + '};\n'
text += 'inline const double ref_cl_quad[4][4] = {' + ','.join('{' + ','.join(format(float(v), '.17g') for v in (*cse, val)) + '}' for cse, val in zip(quad_cases, quad_vals)) + '};\n'
text += arr('ref_ell_dense', ell_dense)
text += 'inline const double ref_cl_dense[2][600] = {' + ','.join('{' + ','.join(format(v, '.17g') for v in c) + '}' for c in cl_dense) + '};\n'
text += arr('ref_theta_arcmin', thetas_arcmin)
text += 'inline const double ref_xi[2][2][7] = {' + ','.join(
    '{' + ','.join('{' + ','.join(format(v, '.17g') for v in x) + '}' for x in p) + '}' for p in xi) + '};\n'
text += arr('ref_cos_norm', norms)
text += 'inline const double ref_cos_roots[5][6] = {' + ','.join(
    '{' + ','.join(format(float(v), '.17g') for v in (r + [0] * (6 - len(r)))) + '}' for r in roots) + '};\n'
text += arr('ref_cos_theta_arcmin', cos_theta_arcmin)
text += 'inline const double ref_cos_T[5][5][2] = {' + ','.join(
    '{' + ','.join('{' + format(a, '.17g') + ',' + format(b, '.17g') + '}' for a, b in tv) + '}' for tv in tvals) + '};\n'
text += '#endif\n'
Path(__file__).with_name('ReferenceCosmicShear.h').write_text(text)
print(text[:3000])
