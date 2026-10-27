#!/usr/bin/env python3
"""Compare the tensor spectra cells_tensor.txt in a directory (./cmb with r > 0) with CLASS (modes = t).

Usage:  python3 python/compare_class_tensors.py DIR [--plot] [--precise]
DIR must contain params.txt and cells_tensor.txt (ell, D_TT, D_EE, D_BB, D_TE in muK^2, tensor contributions).
CLASS runs with the same parameters, tensor modes only, no lensing, and the same r, n_t.
--precise runs CLASS with finer ell and k sampling and longer hierarchies (its defaults interpolate C_ell
from ell steps of 40, which misses part of the tensor TT oscillations).
Printed: max |ours/CLASS - 1| of TT, EE, BB in ell ranges and max |TE diff| / max |TE|.
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

d = sys.argv[1]
p = read_params(os.path.join(d, "params.txt"))
ours = np.loadtxt(os.path.join(d, "cells_tensor.txt"))
ell = ours[:, 0].astype(int)
lmax = int(ell.max())
cp = class_params(p, recombination="RECFAST")
cp.update({"output": "tCl,pCl", "modes": "t", "l_max_tensors": 2 * lmax, "r": p["r"], "n_t": p["n_t"]})
cp.pop("n_s", None)   # unused for tensors alone (CLASS refuses unread parameters)
if "--precise" in sys.argv:   # CLASS with finer ell sampling and higher accuracy
    cp.update({"l_linstep": 5, "l_logstep": 1.02, "q_linstep": 0.1, "k_step_sub": 0.01, "k_step_super": 0.0005,
               "l_max_g_ten": 12, "l_max_pol_g_ten": 12, "l_max_ur": 30, "tol_perturbations_integration": 1e-7,
               "start_sources_at_tau_c_over_tau_h": 0.001})
cl = Class()
cl.set(cp)
cl.compute()
raw = cl.raw_cl(lmax)   # (CLASS l_max_tensors = 2 lmax: its C_ell near l_max_tensors are cut off by its k_max)
T = p["TCMB"] * 1e6
f = ell * (ell + 1) / (2 * np.pi) * T**2
cls = {key: f * raw[key][ell] for key in ("tt", "ee", "bb", "te")}
print(f"r = {p['r']}, n_t = {p['n_t']}")
for lo, hi in [(2, 10), (10, 100), (100, 300), (300, 600)]:
    s = (ell >= lo) & (ell <= min(hi, lmax))
    r = [np.max(np.abs(ours[s, i] / cls[k][s] - 1)) for i, k in ((1, "tt"), (2, "ee"), (3, "bb"))]
    rte = np.max(np.abs(ours[s, 4] - cls["te"][s])) / np.max(np.abs(cls["te"]))
    print(f"  ell {lo:4d}-{hi:4d}:  TT {r[0]:.2e}   EE {r[1]:.2e}   BB {r[2]:.2e}   TE/max|TE| {rte:.2e}")
for l in (2, 10, 80, 200):
    i = l - 2
    print(f"  ell = {l}: ours TT, EE, BB = {ours[i, 1]:.4g}, {ours[i, 2]:.4g}, {ours[i, 3]:.4g};  "
          f"CLASS {cls['tt'][i]:.4g}, {cls['ee'][i]:.4g}, {cls['bb'][i]:.4g} muK^2")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for i, k in ((1, "tt"), (2, "ee"), (3, "bb")):
        ax[0].loglog(ell, ours[:, i], label=f"ours {k.upper()}")
        ax[0].loglog(ell, cls[k], "k--", lw=0.8)
        ax[1].semilogx(ell, ours[:, i] / cls[k] - 1, label=k.upper())
    ax[0].set_ylabel(r"$D_\ell$ [$\mu$K$^2$] (tensor)")
    ax[0].legend()
    ax[1].set(xlabel=r"$\ell$", ylabel="ours/CLASS - 1", ylim=(-0.03, 0.03))
    ax[1].legend()
    fig.suptitle(f"tensor modes, r = {p['r']} (dashed: CLASS)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "compare_class_tensors.png"), dpi=120)
