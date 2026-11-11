#!/usr/bin/env python3
"""Regenerate the velocileptors reference of tests/test_lptrsd.cpp (LPTRedshiftSpace.h).

Run from PerturbationTheory/. Requires velocileptors (https://github.com/sfschen/velocileptors), numpy, scipy and
pyFFTW. The input P(k) (tests/reference/lptrsd_linear_pk.txt) is every 4th row of HaloModel/example/pofk_lin.txt.
velocileptors' defaults (jn = 5, N = 256, nmax = 5) are used; the C++ test sets the same options.
"""
from pathlib import Path
import numpy as np
import velocileptors
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

ref = Path(__file__).resolve().parents[1] / "tests/reference"
k, p = np.loadtxt(ref / "lptrsd_linear_pk.txt", unpack=True)
kv = np.logspace(-2, np.log10(0.3), 20)
rows = []
for kIR, pzel in [(None, False), (0.2, True)]:
    lpt = LPT_RSD(k, p, kIR=kIR, use_Pzel=pzel, jn=5, N=256, cutoff=10, threads=1)
    lpt.make_ptable(0.7, 0.6, kv=kv, nmax=5)
    for r in lpt.pktable:
        rows.append(["C", kIR or 0, int(pzel)] + list(r))
    lpt.make_pltable(0.7, apar=1.02, aperp=0.98, ngauss=3, kv=kv, nmax=5)
    for ell, tab in [(0, lpt.p0ktable), (2, lpt.p2ktable), (4, lpt.p4ktable)]:
        for kk, r in zip(kv, tab):
            rows.append([f"P{ell}", kIR or 0, int(pzel), kk] + list(r))
with open(ref / "lptrsd_velocileptors.txt", "w") as fp:
    fp.write(f"# velocileptors {getattr(velocileptors, '__version__', 'git')} LPT_RSD, f = 0.7: rows C (k, the 13 "
             "components at mu = 0.6) and P0/P2/P4 (k, the 19 columns, apar = 1.02, aperp = 0.98, ngauss = 3); "
             "columns 2-3: kIR, use_Pzel\n")
    for r in rows:
        fp.write(r[0] + " " + " ".join(f"{v:.17g}" for v in r[1:]) + "\n")
print("written", ref / "lptrsd_velocileptors.txt")
