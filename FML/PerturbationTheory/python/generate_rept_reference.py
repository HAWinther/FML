#!/usr/bin/env python3
"""Regenerate the velocileptors reference of tests/test_rept.cpp (EPTRedshiftSpace.h).

Run from PerturbationTheory/. Requires velocileptors (https://github.com/sfschen/velocileptors), numpy, scipy and
pyFFTW. The input P(k) is tests/reference/lptrsd_linear_pk.txt. The no-wiggle spectrum is passed explicitly (the
Savitzky-Golay filter of velocileptors' default on the internal grid, interpolated back onto the input k), so that
both codes apply the cutoff once; rows S check the filter itself.
"""
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import velocileptors
from velocileptors.Utils.loginterp import loginterp
from velocileptors.EPT.ept_fullresum_fftw import REPT

ref = Path(__file__).resolve().parents[1] / "tests/reference"
k, p = np.loadtxt(ref / "lptrsd_linear_pk.txt", unpack=True)

N, emin, emax = 4000, -4, 3
kint = np.logspace(emin, emax, N)
praw = loginterp(k, p)(kint)
W = int(np.ceil(np.log(7) / np.log(kint[-1] / kint[-2])) // 2 * 2 + 1)
pnw_int = savgol_filter(praw, W, 4)
pnw = loginterp(kint, pnw_int)(k)
np.savetxt(ref / "rept_nowiggle_pk.txt", np.c_[k, pnw], fmt="%.12e",
           header="k (h/Mpc), the no-wiggle P(k) of tests/test_rept.cpp")

rows = []
for i in list(range(0, 300, 7)) + list(range(300, N - 300, 97)) + list(range(N - 300, N, 7)):
    rows.append(["S", i, kint[i], pnw_int[i]])

f = 0.75
rept = REPT(k, p, pnw=pnw, kmin=1e-2, kmax=0.5, nk=60, rbao=110, cutoff=20, jn=5, N=N, extrap_min=emin,
            extrap_max=emax, threads=1)
rows.append(["SIGMA", rept.sigma_squared_bao])
ept = rept.ept
for j, kk in enumerate(rept.kv):
    rows.append(["T", kk] + list(ept.pktable_ept[j, 1:8]) + list(ept.vktable_ept[j, 1:8])
                + list(ept.s0ktable_ept[j, 1:6]) + list(ept.s2ktable_ept[j, 1:6]) + list(ept.g1ktable_ept[j, 1:3])
                + list(ept.g3ktable_ept[j, 1:3]) + [ept.k0[j], ept.k2[j], ept.k4[j]])
pars = [1.8, 0.4, -0.3, 0.2, 10.0, -5.0, 2.0, 1.0, 800.0, -300.0, 100.0]
for apar, aperp in [(1.0, 1.0), (1.03, 0.97)]:
    for mu in [0.15, 0.7]:
        tab = rept.compute_redshift_space_power_table_at_mu(f, mu, apar=apar, aperp=aperp)
        for kk, r in zip(rept.kv, tab):
            rows.append(["M", apar, aperp, mu, kk] + list(r))
        kv, pk = rept.compute_redshift_space_power_at_mu(pars, f, mu, apar=apar, aperp=aperp, bFoG=0.5)
        for kk, r in zip(kv, pk):
            rows.append(["F", apar, aperp, mu, kk, r])
    kv, p0, p2, p4 = rept.compute_redshift_space_power_multipoles_tables(f, ngauss=4, apar=apar, aperp=aperp)
    for ell, tab in [(0, p0), (2, p2), (4, p4)]:
        for kk, r in zip(kv, tab):
            rows.append([f"P{ell}", apar, aperp, kk] + list(r))
with open(ref / "rept_velocileptors.txt", "w") as fp:
    fp.write(f"# velocileptors {getattr(velocileptors, '__version__', 'git')} REPT, f = {f}, N = {N}, kmin = 1e-2, "
             "kmax = 0.5, nk = 60, pnw from rept_nowiggle_pk.txt. Rows: S (index, k, savgol no-wiggle on the internal "
             "grid), SIGMA, T (k, pk 7, vk 7, s0 5, s2 5, g1 2, g3 2, k0 k2 k4), M (apar, aperp, mu, k, 19 columns), "
             "F (apar, aperp, mu, k, P(k,mu) for pars 1.8 0.4 -0.3 0.2 10 -5 2 1 800 -300 100, bFoG = 0.5), "
             "P0/P2/P4 (apar, aperp, k, 19 columns; ngauss = 4)\n")
    for r in rows:
        fp.write(" ".join(r[:1] + [f"{v:.15e}" if isinstance(v, float) else str(v) for v in r[1:]]) + "\n")
