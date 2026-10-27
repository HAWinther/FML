#!/usr/bin/env python3
"""Compare cells.txt and matter_power.txt in a directory (written by ./cmb) with CLASS.

Usage:  python3 python/compare_class_spectra.py DIR [--plot] [--hyrec] [--precise]
DIR must contain params.txt, cells.txt (ell, D_TT, D_EE, D_TE in muK^2) and matter_power.txt
(k [h/Mpc], P [(Mpc/h)^3]). CLASS runs with the same parameters (incl. Omega_k), no lensing.
Printed: max |ours/CLASS - 1| for TT and EE in ell ranges, max |TE diff| / max |TE|, and P(k).
With --plot, writes DIR/compare_class.png. CLASS uses RECFAST (as our default recombination model); --hyrec
runs it with HyRec instead. Isocurvature modes (f_cdi, ... in params.txt) are passed on as ic = ad,cdi,...;
--precise runs CLASS with finer k and ell sampling (needed for the NIV mode, which CLASS's defaults get wrong by
~1-3%).
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

d = sys.argv[1]
p = read_params(os.path.join(d, "params.txt"))
cl_ours = np.loadtxt(os.path.join(d, "cells.txt"))
pk_ours = np.loadtxt(os.path.join(d, "matter_power.txt"))
ell = cl_ours[:, 0].astype(int)
lmax = int(ell.max())

cp = class_params(p, recombination="HyRec" if "--hyrec" in sys.argv else "RECFAST")
cp.update({"output": "tCl,pCl,mPk", "l_max_scalars": max(lmax, 2500), "P_k_max_1/Mpc": 1.0})
if "--precise" in sys.argv:
    cp.update({"l_linstep": 5, "l_logstep": 1.02, "q_linstep": 0.1, "k_step_sub": 0.01, "k_step_super": 0.0005,
               "tol_perturbations_integration": 1e-7, "start_small_k_at_tau_c_over_tau_h": 1e-4,
               "start_large_k_at_tau_h_over_tau_k": 1e-4})
cl = Class()
cl.set(cp)
cl.compute()
raw = cl.raw_cl(lmax)
T = p["TCMB"] * 1e6
f = ell * (ell + 1) / (2 * np.pi) * T**2
tt, ee, te = (f * raw[key][ell] for key in ("tt", "ee", "te"))

h = p["h"]
k = pk_ours[:, 0]
m = k * h < 0.9
pk_class = np.array([cl.pk_lin(kk * h, 0.0) * h**3 for kk in k[m]])

print(f"Omega_k = {p['OmegaK']}")
for lo, hi in [(2, 30), (30, 200), (200, 1000), (1000, 2000)]:
    s = (ell >= lo) & (ell <= min(hi, lmax))
    if not s.any():
        continue
    rtt = np.max(np.abs(cl_ours[s, 1] / tt[s] - 1))
    ree = np.max(np.abs(cl_ours[s, 2] / ee[s] - 1))
    rte = np.max(np.abs(cl_ours[s, 3] - te[s])) / np.max(np.abs(te))
    print(f"  ell {lo:4d}-{hi:4d}:  TT {rtt:.2e}   EE {ree:.2e}   TE/max|TE| {rte:.2e}")
print(f"  P(k), k < 0.9/Mpc: {np.max(np.abs(pk_ours[m, 1] / pk_class - 1)):.2e}")
lpk = np.argmax(tt)
print(f"  first TT peak: ours ell = {ell[np.argmax(cl_ours[:, 1])]}, CLASS ell = {ell[lpk]}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].semilogx(ell, cl_ours[:, 1], label="ours")
    ax[0].semilogx(ell, tt, "--", label="CLASS")
    ax[0].set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
    ax[0].legend()
    ax[1].semilogx(ell, cl_ours[:, 1] / tt - 1, label="TT")
    ax[1].semilogx(ell, cl_ours[:, 2] / ee - 1, label="EE")
    ax[1].semilogx(ell, (cl_ours[:, 3] - te) / np.max(np.abs(te)), label="TE/max|TE|")
    ax[1].set(xlabel=r"$\ell$", ylabel="ours/CLASS - 1", ylim=(-0.05, 0.05))
    ax[1].legend()
    fig.suptitle(rf"$\Omega_K = {p['OmegaK']}$")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "compare_class.png"), dpi=120)
