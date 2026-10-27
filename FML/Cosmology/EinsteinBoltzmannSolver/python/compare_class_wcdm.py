#!/usr/bin/env python3
"""Compare wCDM (dark-energy fluid with perturbations) with CLASS: the effect w != -1 relative to LCDM.

Usage:  python3 python/compare_class_wcdm.py LCDMDIR WCDMDIR
e.g.
    ./cmb pert_n_k=100 ell_max=1200 output_dir=data/lcdm_ref
    ./cmb pert_n_k=100 ell_max=1200 model=dark_scattering ds_xi=0 ds_w=-0.9 output_dir=data/wcdm
WCDMDIR is a dark_scattering run with ds_xi = 0 (no scattering); CLASS runs its fluid dark energy with the
same w and cs2 (use_ppf = no). Prints X(wCDM)/X(LCDM) for C_ell^TT and P(k) in both codes.
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
dl, dw = sys.argv[1], sys.argv[2]
pl, pw = read_params(os.path.join(dl, "params.txt")), read_params(os.path.join(dw, "params.txt"))
if pw.get("ds_xi", 0.0) != 0.0:
    print("warning: ds_xi != 0 in", dw, "- CLASS has no dark scattering, the comparison is for wCDM")
cl_ours = {d: np.loadtxt(os.path.join(d, "cells.txt")) for d in (dl, dw)}
lmax = int(min(cl_ours[dl][-1, 0], cl_ours[dw][-1, 0]))


def run(extra):
    cp = class_params(pl)
    cp.update({"output": "tCl,mPk", "l_max_scalars": lmax + 100, "P_k_max_1/Mpc": 1.0})
    cp.update(extra)
    c = Class()
    c.set(cp)
    c.compute()
    return c


L = run({})
W = run({"Omega_Lambda": 0, "w0_fld": pw["ds_w"], "wa_fld": 0, "cs2_fld": pw.get("ds_cs2", 1.0), "use_ppf": "no"})
ell = np.arange(2, lmax + 1)
rc = W.raw_cl(lmax)["tt"][ell] / L.raw_cl(lmax)["tt"][ell]
a, b = cl_ours[dl], cl_ours[dw]
ro = np.interp(ell, b[:, 0], b[:, 1]) / np.interp(ell, a[:, 0], a[:, 1])
print("C_ell^TT(wCDM)/C_ell^TT(LCDM):   ell   ours   CLASS   difference")
for l in [2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]:
    if l <= lmax:
        print(f"  {l:5d}  {ro[l - 2]:.5f}  {rc[l - 2]:.5f}  {ro[l - 2] / rc[l - 2] - 1:+.1e}")
pa, pb = np.loadtxt(os.path.join(dl, "matter_power.txt")), np.loadtxt(os.path.join(dw, "matter_power.txt"))
h = pl["h"]
print("P(wCDM)/P(LCDM) at z = 0:   k [h/Mpc]   ours   CLASS   difference")
for k in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
    o = np.interp(k, pa[:, 0], pb[:, 1] / pa[:, 1])
    c = W.pk_lin(k * h, 0.0) / L.pk_lin(k * h, 0.0)
    print(f"  {k:8.4f}  {o:.5f}  {c:.5f}  {o / c - 1:+.1e}")
