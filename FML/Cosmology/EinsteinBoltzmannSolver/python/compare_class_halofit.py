#!/usr/bin/env python3
"""Compare the Halofit (nonlinear) matter power spectrum and the lensed spectra with nonlinear lensing with CLASS.

Usage:  python3 python/compare_class_halofit.py DIR [--plot]
DIR from ./cmb with nonlinear = halofit (and lensing = true): params.txt, matter_power.txt (and _z<z>.txt for z = 1,
with matter_power_z = 0, 1), cells_lensed.txt. CLASS runs with non_linear = halofit (Takahashi 2012). Printed:
P_NL/P_NL^CLASS - 1 at a few k for z = 0 (and 1), and lensed TT, EE, BB against CLASS.
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

d = sys.argv[1]
p = read_params(os.path.join(d, "params.txt"))
h = p["h"]
cp = class_params(p, recombination="RECFAST")
cp.update({"output": "tCl,pCl,lCl,mPk", "lensing": "yes", "non_linear": "halofit", "P_k_max_1/Mpc": 30.0,
           "l_max_scalars": 3000, "z_max_pk": 2.0})
cl = Class()
cl.set(cp)
cl.compute()
for z, f in ((0.0, "matter_power.txt"), (1.0, "matter_power_z1.txt")):
    path = os.path.join(d, f)
    if not os.path.exists(path):
        continue
    a = np.loadtxt(path)
    if a.shape[1] < 4:
        sys.exit(f"{f} has no nonlinear column (run with nonlinear = halofit)")
    print(f"z = {z}:  k [h/Mpc]   P_NL ours/CLASS - 1   (P_NL/P_L ours)")
    for kh in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        if kh > a[-1, 0]:
            continue
        pn = np.interp(np.log(kh), np.log(a[:, 0]), a[:, 3])
        pl = np.interp(np.log(kh), np.log(a[:, 0]), a[:, 1])
        pc = cl.pk(kh * h, z) * h**3
        print(f"   {kh:5.2f}   {pn / pc - 1:+.2e}   ({pn / pl:.3f})")
lens = os.path.join(d, "cells_lensed.txt")
if os.path.exists(lens):
    o = np.loadtxt(lens)
    ell = o[:, 0].astype(int)
    lmax = int(ell.max())
    raw = cl.lensed_cl(lmax)
    T = p["TCMB"] * 1e6
    f = ell * (ell + 1) / (2 * np.pi) * T**2
    print("lensed spectra, nonlinear lensing: max |ours/CLASS - 1|")
    for lo, hi in ((2, 500), (500, 1500), (1500, 2500)):
        s = (ell >= lo) & (ell <= min(hi, lmax))
        r = [np.max(np.abs(o[s, i] / (f[s] * raw[k][ell[s]]) - 1)) for i, k in ((1, "tt"), (2, "ee"), (4, "bb"))]
        print(f"  ell {lo:4d}-{hi:4d}:  TT {r[0]:.2e}  EE {r[1]:.2e}  BB {r[2]:.2e}")
