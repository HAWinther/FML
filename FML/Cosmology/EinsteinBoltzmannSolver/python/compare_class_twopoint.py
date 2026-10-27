#!/usr/bin/env python3
"""Compare the two-point functions written by ./cmb params/twopoint.ini with CLASS.

Usage:  python3 python/compare_class_twopoint.py [DIR] [--plot]      (default DIR = data/twopoint)

* sigma_8 (CLASS sigma8()).
* xi(r) from CLASS's linear P(k) (computed to k = 50/Mpc), transformed with an independent integrator
  (scipy quad with a sin weight on a spline of k P(k)), versus ours.
* C^TT(theta) (unlensed and lensed) from CLASS's C_ell with a Legendre sum, versus ours.
* Angular power spectrum of the projected density for Gaussian n(z) at z = 0.5 and 1 (sigma_z = 0.1):
  CLASS nCl with number_count_contributions = density, versus ours; and w(theta).
"""
import os
import sys

import numpy as np
from classy import Class
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d = args[0] if args else "data/twopoint"
p = read_params(os.path.join(d, "params.txt"))
h, T = p["h"], p["TCMB"] * 1e6
LMAX, LLSS = 2500, 2000
ZS = [0.5, 1.0]


def legendre_sum(cl, theta):
    """sum_l (2l+1)/(4pi) cl[l] P_l(cos theta), l >= 2"""
    out = np.zeros_like(theta)
    mu = np.cos(theta)
    p0, p1 = np.ones_like(mu), mu.copy()
    for ell in range(1, len(cl) - 1):
        p2 = ((2 * ell + 1) * mu * p1 - ell * p0) / (ell + 1)
        if ell + 1 >= 2:
            out += (2 * ell + 3) / (4 * np.pi) * cl[ell + 1] * p2
        p0, p1 = p1, p2
    return out


cp = class_params(p)
cp.update({"output": "tCl,pCl,lCl,mPk,nCl", "lensing": "yes", "l_max_scalars": LMAX + 500,
           "P_k_max_1/Mpc": 50.0, "selection": "gaussian", "selection_mean": ", ".join(str(z) for z in ZS),
           "selection_width": 0.1, "number_count_contributions": "density", "non_diagonal": 0,
           "l_max_lss": LLSS,
           # exact projection (no Limber) up to ell = 1000 z, so that the comparison tests our Limber switch
           "l_switch_limber_for_nc_local_over_z": 1000.0, "l_switch_limber_for_nc_los_over_z": 1000.0})
c = Class()
c.set(cp)
c.compute()

# sigma_8
s8 = float(open(os.path.join(d, "matter_correlation.txt")).readline().split("sigma_8 =")[1].split()[0])
print(f"sigma_8: ours {s8:.5f}  CLASS {c.sigma8():.5f}  rel {s8 / c.sigma8() - 1:+.2e}")

# xi(r)
xo = np.loadtxt(os.path.join(d, "matter_correlation.txt"))
kk = np.logspace(-5, np.log10(49.0), 3000)          # 1/Mpc
Pk = np.array([c.pk_lin(k, 0.0) for k in kk])        # Mpc^3
f = CubicSpline(np.log(kk), np.log(kk * Pk))
res = []
for r_h in (10, 20, 50, 80, 90, 100, 105, 110, 120, 150, 200):
    r = r_h / h
    val = 0.0
    edges = np.concatenate([[kk[0]], np.logspace(-3, np.log10(kk[-1]), 400)])
    for a, b in zip(edges[:-1], edges[1:]):
        val += quad(lambda k: np.exp(f(np.log(k))), a, b, weight="sin", wvar=r, limit=200)[0]
    res.append((r_h, np.interp(r_h, xo[:, 0], xo[:, 1]), val / (2 * np.pi**2 * r)))
res = np.array(res)
peak = np.max(xo[(xo[:, 0] > 80) & (xo[:, 0] < 120), 2])   # BAO peak of r^2 xi
print(f"xi(r) vs xi from CLASS's P(k) (scipy quad): r^2 xi BAO peak {peak:.3f} (Mpc/h)^2 at "
      f"{xo[(xo[:, 0] > 80) & (xo[:, 0] < 120)][np.argmax(xo[(xo[:, 0] > 80) & (xo[:, 0] < 120), 2]), 0]:.1f} Mpc/h")
for r_h, a, b in res:
    print(f"  r = {r_h:5.0f} Mpc/h: ours {a:+.4e}  CLASS {b:+.4e}  rel {a / b - 1:+.1e}  "
          f"r^2 diff / BAO peak {r_h**2 * (a - b) / peak:+.1e}")

# CMB C(theta)
th_deg = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 90, 120, 180.0])
th = np.radians(th_deg)
for lensed, fname in ((False, "cmb_correlation_unlensed.txt"), (True, "cmb_correlation_lensed.txt")):
    cl = (c.lensed_cl(LMAX) if lensed else c.raw_cl(LMAX))["tt"] * T**2
    Cc = legendre_sum(cl, th)
    o = np.loadtxt(os.path.join(d, fname))
    Co = np.interp(th_deg, o[:, 0], o[:, 1])
    print(f"C^TT(theta), {'lensed' if lensed else 'unlensed'}: max |ours - CLASS| / C(0) = "
          f"{np.max(np.abs(Co - Cc)) / Cc[0]:.2e}  (C(0) ours {Co[0]:.1f}, CLASS {Cc[0]:.1f} muK^2)")

# Projected density
dcl = c.density_cl(LLSS)
for i, z0 in enumerate(ZS):
    o = np.loadtxt(os.path.join(d, f"angular_cl_z{z0:g}.txt"))
    ell = o[:, 0].astype(int)
    cc = dcl["dd"][f"dens[{i + 1}]-dens[{i + 1}]"][ell]
    r = o[:, 1] / cc - 1
    for lo, hi in ((2, 20), (20, 100), (100, 500), (500, 2000)):
        s = (ell >= lo) & (ell <= hi)
        print(f"  C_ell density z = {z0}: ell {lo:4d}-{hi:4d}: max |ours/CLASS - 1| {np.max(np.abs(r[s])):.2e}")
    clc = np.zeros(LLSS + 1)
    clc[ell] = cc
    # w(theta) from both C_ell (ell <= 2000, so this tests the C_ell; the w file of ./cmb uses ell <= 30000)
    clo = np.zeros(LLSS + 1)
    clo[ell] = o[:, 1]
    tw = np.array([0.5, 1, 2, 3, 5, 10.0])
    wc, wo = legendre_sum(clc, np.radians(tw)), legendre_sum(clo, np.radians(tw))
    print(f"  w(theta) z = {z0}, ours/CLASS: " + "  ".join(f"{t}deg {a:+.3e}/{b:+.3e}" for t, a, b in zip(tw, wo, wc)))

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(xo[:, 0], xo[:, 2])
    ax[0].plot([x[0] for x in res], [x[0] ** 2 * x[2] for x in res], "o")
    ax[0].set(xlabel="r [Mpc/h]", ylabel="r^2 xi")
    o = np.loadtxt(os.path.join(d, "cmb_correlation_lensed.txt"))
    ax[1].plot(o[:, 0], o[:, 1])
    ax[1].set(xlabel="theta [deg]", ylabel="C^TT [muK^2]", xscale="log")
    w = np.loadtxt(os.path.join(d, "angular_w_z1.txt"))
    ax[2].loglog(w[:, 0], np.abs(w[:, 1]))
    ax[2].set(xlabel="theta [deg]", ylabel="|w|")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "compare_class_twopoint.png"), dpi=120)
