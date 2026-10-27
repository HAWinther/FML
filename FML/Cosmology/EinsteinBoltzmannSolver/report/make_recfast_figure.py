#!/usr/bin/env python3
"""Figure and numbers for the RECFAST section of the report.

Usage (from the project root):
  ./cmb transfer_ells= output_dir=data/rf/fid
  ./cmb transfer_ells= output_dir=data/rf/fid_peebles recombination=peebles
  ./cmb reionization=false pert_n_k=2 ell_max=20 output_dir=data/rf/norei
  ./cmb reionization=false recombination=peebles pert_n_k=2 ell_max=20 output_dir=data/rf/norei_peebles
  python3 report/make_recfast_figure.py
Runs CLASS (RECFAST and HyRec) with the same parameters and writes report/figures/recfast.pdf and
report/data/recfast_numbers.txt.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

sys.path.insert(0, "python")
from params import class_params, read_params  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm"})
C_RF, C_PB, C_HY = "#2a78d6", "#eb6834", "#1baf7a"
num = {}


def rec(d):
    r = np.loadtxt(os.path.join(d, "recombination.txt"))
    x = r[:, 0]
    return {k: CubicSpline(x, r[:, i]) for k, i in [("Xe", 1), ("tau", 3), ("Tb", 9), ("s", 11)]}


def class_run(p, recomb, extra):
    cp = class_params(p, recombination=recomb)
    cp.update(extra)
    c = Class()
    c.set(cp)
    c.compute()
    return c


# --- recombination history, no reionization ---------------------------------
pn = read_params("data/rf/norei/params.txt")
ours = {"RECFAST": rec("data/rf/norei"), "Peebles": rec("data/rf/norei_peebles")}
cls = {}
for r in ("RECFAST", "HyRec"):
    c = class_run(pn, r, {})
    th = c.get_thermodynamics()
    o = np.argsort(th["z"])
    der = c.get_current_derived_parameters(["z_star", "rs_star", "z_d", "rs_d"])
    cls[r] = {"Xe": CubicSpline(th["z"][o], th["x_e"][o]), **der}
z = np.logspace(0, np.log10(8000), 800)
for name, o in ours.items():
    xs = brentq(lambda xx: o["tau"](xx) - 1.0, -8, -6)
    num[f"{name}_zstar"] = np.expm1(-xs)
    num[f"{name}_rs_star"] = float(o["s"](xs))
    for r in ("RECFAST", "HyRec"):
        dev = o["Xe"](-np.log1p(z)) / cls[r]["Xe"](z) - 1
        num[f"{name}_vs_CLASS_{r}_Xe_maxdev"] = np.max(np.abs(dev))
        sel = (z > 800) & (z < 1600)
        num[f"{name}_vs_CLASS_{r}_Xe_maxdev_z800_1600"] = np.max(np.abs(dev[sel]))
for r in ("RECFAST", "HyRec"):
    num[f"CLASS_{r}_zstar"] = cls[r]["z_star"]
    num[f"CLASS_{r}_rs_star"] = cls[r]["rs_star"]

# --- spectra with reionization -------------------------------------------------
pf = read_params("data/rf/fid/params.txt")
cl_o = {"RECFAST": np.loadtxt("data/rf/fid/cells.txt"), "Peebles": np.loadtxt("data/rf/fid_peebles/cells.txt")}
ell = cl_o["RECFAST"][:, 0].astype(int)
lmax = int(ell.max())
f = ell * (ell + 1) / (2 * np.pi) * (pf["TCMB"] * 1e6) ** 2
ccl = {}
for r in ("RECFAST", "HyRec"):
    c = class_run(pf, r, {"output": "tCl,pCl", "l_max_scalars": 2500})
    raw = c.raw_cl(lmax)
    ccl[r] = {"TT": f * raw["tt"][ell], "EE": f * raw["ee"][ell]}
for name, o in cl_o.items():
    for r in ("RECFAST", "HyRec"):
        for j, s in [(1, "TT"), (2, "EE")]:
            dev = o[:, j] / ccl[r][s] - 1
            for lo, hi in [(2, 30), (30, 1000), (1000, 2000)]:
                sel = (ell >= lo) & (ell <= hi)
                num[f"{name}_vs_CLASS_{r}_{s}_rms_{lo}_{hi}"] = np.sqrt(np.mean(dev[sel] ** 2))
                num[f"{name}_vs_CLASS_{r}_{s}_max_{lo}_{hi}"] = np.max(np.abs(dev[sel]))

# --- figure --------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
a = ax[0]
for name, col in [("RECFAST", C_RF), ("Peebles", C_PB)]:
    a.semilogx(1 + z, ours[name]["Xe"](-np.log1p(z)) / cls["RECFAST"]["Xe"](z) - 1, color=col, label=f"ours, {name}")
a.semilogx(1 + z, cls["HyRec"]["Xe"](z) / cls["RECFAST"]["Xe"](z) - 1, color=C_HY, ls="--", label="CLASS HyRec")
a.set(xlabel="$1+z$", ylabel=r"$X_e/X_e^{\rm CLASS\,RECFAST}-1$", ylim=(-0.08, 0.2), xlim=(1, 8000))
a.legend(loc="lower left")
for j, (s, a) in enumerate([("TT", ax[1]), ("EE", ax[2])]):
    for name, col in [("RECFAST", C_RF), ("Peebles", C_PB)]:
        a.semilogx(ell, cl_o[name][:, j + 1] / ccl["RECFAST"][s] - 1, color=col, lw=0.8, label=f"ours, {name}")
    a.semilogx(ell, ccl["HyRec"][s] / ccl["RECFAST"][s] - 1, color=C_HY, ls="--", lw=0.8, label="CLASS HyRec")
    a.set(xlabel=r"$\ell$", ylabel=rf"$C_\ell^{{{s}}}/C_\ell^{{{s},\rm CLASS\,RECFAST}}-1$", xlim=(2, 2000),
          ylim=(-0.03, 0.03) if s == "TT" else (-0.06, 0.06))
ax[1].legend(loc="lower left")
fig.tight_layout()
fig.savefig("report/figures/recfast.pdf")
with open("report/data/recfast_numbers.txt", "w") as fp:
    for k, v in num.items():
        fp.write(f"{k} {v:.6g}\n")
print(open("report/data/recfast_numbers.txt").read())
