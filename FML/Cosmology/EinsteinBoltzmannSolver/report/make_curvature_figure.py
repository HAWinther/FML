#!/usr/bin/env python3
"""Figures and numbers for the curvature section of the report.

Usage (from the project root):
  for K in -0.2 -0.05 0 0.05 0.2; do
    ./cmb params/curvature.ini OmegaK=$K output_dir=data/curvature_$K > data/log_curv_$K.txt; done
  python3 report/make_curvature_figure.py
Reads data/curvature_<OmegaK>/{params.txt, cells.txt, background.txt}, the last-scattering numbers
from data/log_curv_<OmegaK>.txt, runs CLASS for each model (unlensed), and writes
report/figures/curvature_spectra.pdf and report/data/curvature_numbers.txt.
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from classy import Class

sys.path.insert(0, "python")
from params import class_params, read_params  # noqa: E402

MODELS = ["-0.2", "-0.05", "0", "0.05", "0.2"]
C = {"-0.2": "#eb6834", "-0.05": "#eda100", "0": "#3d3d3a", "0.05": "#1baf7a", "0.2": "#2a78d6"}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm"})

rows = []
fig, ax = plt.subplots(3, 1, figsize=(3.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1]))
for K in MODELS:
    d = f"data/curvature_{K}"
    p = read_params(os.path.join(d, "params.txt"))
    cl = np.loadtxt(os.path.join(d, "cells.txt"))
    ell = cl[:, 0].astype(int)
    cp = class_params(p)
    cp.update({"output": "tCl,pCl", "l_max_scalars": 2500})
    c = Class()
    c.set(cp)
    c.compute()
    raw = c.raw_cl(int(ell.max()))
    f = ell * (ell + 1) / (2 * np.pi) * (p["TCMB"] * 1e6) ** 2
    tt, ee = f * raw["tt"][ell], f * raw["ee"][ell]

    # Acoustic scale ell_A = pi r(x_*) / r_s(x_*) from our background and recombination
    log = open(f"data/log_curv_{K}.txt").read()
    xs = float(re.findall(r"Last scattering \(tau = 1\):\s+x = ([-\d.]+)", log)[0])
    rs = float(re.findall(r"Sound horizon at tau = 1\s+= ([\d.]+)", log)[0])
    bg = np.loadtxt(os.path.join(d, "background.txt"))
    dA = np.interp(xs, bg[:, 0], bg[:, 14])  # Mpc
    r = dA / np.exp(xs)
    lA = np.pi * r / rs

    lab = rf"$\Omega_K={float(K):+g}$" if K != "0" else r"$\Omega_K=0$"
    ax[0].plot(ell, cl[:, 1], color=C[K], label=lab)
    ax[1].plot(ell, cl[:, 1] / tt - 1, color=C[K])
    ax[2].plot(ell, cl[:, 2] / ee - 1, color=C[K])

    s_lo, s_hi = ell <= 30, (ell >= 30) & (ell <= 2000)
    rows.append((K, p.get("OmegaK", 0.0), 1 - (p["OmegaB"] + p["OmegaCDM"]) - float(K) - 9.3e-5,
                 ell[np.argmax(cl[:, 1])], ell[np.argmax(tt)], lA,
                 np.max(np.abs(cl[s_lo, 1] / tt[s_lo] - 1)), np.max(np.abs(cl[s_hi, 1] / tt[s_hi] - 1)),
                 np.max(np.abs(cl[ell >= 100, 2] / ee[ell >= 100] - 1)), cl[ell == 2, 1][0], tt[ell == 2][0]))

ax[0].set(ylabel=r"$D_\ell^{TT}$ [$\mu$K$^2$]", xscale="log", xlim=(2, 2000))
ax[0].legend(loc="upper left")
ax[1].set(ylabel="TT: ours/CLASS$-1$", ylim=(-0.04, 0.04))
ax[2].set(xlabel=r"$\ell$", ylabel="EE: ours/CLASS$-1$", ylim=(-0.1, 0.1))
fig.savefig("report/figures/curvature_spectra.pdf")
fig.savefig("data/curvature_spectra.png", dpi=150)

with open("report/data/curvature_numbers.txt", "w") as fp:
    fp.write("# OmegaK OmegaLambda ell_peak_ours ell_peak_CLASS ell_A maxTT_l<=30 maxTT_30-2000 maxEE_l>=100 D2_ours D2_CLASS\n")
    for r_ in rows:
        fp.write(f"{r_[1]:+.2f} {r_[2]:.4f} {r_[3]} {r_[4]} {r_[5]:.2f} {r_[6]:.2e} {r_[7]:.2e} {r_[8]:.2e} "
                 f"{r_[9]:.1f} {r_[10]:.1f}\n")
print(open("report/data/curvature_numbers.txt").read())
