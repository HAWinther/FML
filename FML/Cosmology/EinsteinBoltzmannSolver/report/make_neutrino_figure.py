#!/usr/bin/env python3
"""Figure and numbers for the massive-neutrino section of the report.

Usage (from the project root):
  for m in 0 0.06 0.15 0.3; do
    ./cmb params/neutrinos.ini m_nu=$m output_dir=data/neutrinos_$m > data/log_nu_$m.txt; done
  python3 report/make_neutrino_figure.py
Reads data/neutrinos_<m>/, runs CLASS for each mass (with lensing), and writes
report/figures/neutrinos.pdf and report/data/neutrino_numbers.txt.
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from classy import Class

sys.path.insert(0, "python")
from params import class_params, read_params  # noqa: E402

MASSES = ["0.06", "0.15", "0.3"]
C = {"0.06": "#1baf7a", "0.15": "#2a78d6", "0.3": "#eb6834"}
LMAX = 2000
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm"})


def load(m):
    d = f"data/neutrinos_{m}"
    p = read_params(os.path.join(d, "params.txt"))
    cl = np.loadtxt(os.path.join(d, "cells.txt"))
    cl = cl[cl[:, 0] <= LMAX]
    o = {"p": p, "ell": cl[:, 0].astype(int), "TT": cl[:, 1], "EE": cl[:, 2],
         "len": np.loadtxt(os.path.join(d, "cells_lensed.txt"))[: len(cl)],
         "pk": np.loadtxt(os.path.join(d, "matter_power.txt")),
         "bg": np.loadtxt(os.path.join(d, "background.txt")),
         "pert": {k: np.loadtxt(os.path.join(d, f"perturbations_k{k}.txt")) for k in ("0.01", "0.1")},
         "log": open(f"data/log_nu_{m}.txt").read()}
    cp = class_params(p)
    cp.update({"output": "tCl,pCl,lCl,mPk", "lensing": "yes", "l_max_scalars": LMAX + 500, "P_k_max_1/Mpc": 1.0})
    c = Class()
    c.set(cp)
    c.compute()
    ell, h = o["ell"], p["h"]
    k = o["pk"][:, 0]
    mk = k * h < 0.9
    raw = c.raw_cl(LMAX)
    o["k"] = k[mk]
    o["Pm"], o["Pcb"] = o["pk"][mk, 1], o["pk"][mk, 2]
    o["class"] = {"TT": raw["tt"][ell], "EE": raw["ee"][ell], "TTlens": c.lensed_cl(LMAX)["tt"][ell],
                  "pp": raw["pp"][ell], "Pm": np.array([c.pk_lin(kk * h, 0.0) for kk in o["k"]])}
    o["class"]["Pcb"] = (np.array([c.pk_cb_lin(kk * h, 0.0) for kk in o["k"]]) if float(m) > 0 else o["class"]["Pm"])
    o["ours"] = {"TT": o["TT"], "EE": o["EE"], "TTlens": o["len"][:, 1], "pp": o["len"][:, 5],
                 "Pm": o["Pm"], "Pcb": o["Pcb"]}
    return o


ref = load("0")
runs = {m: load(m) for m in MASSES}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
# (a) background: Omega_ncdm(a) and w(a)
ax = axes[0, 0]
for m in MASSES:
    bg = runs[m]["bg"]
    z = np.exp(-bg[:, 0]) - 1
    ax.loglog(1 + z, bg[:, 17], color=C[m], label=rf"$w$, $m_\nu={m}$ eV")
    ax.loglog(1 + z, bg[:, 16] / (bg[:, 7] + bg[:, 8] + bg[:, 16]), color=C[m], ls="--")
ax.set(xlabel=r"$1+z$", ylabel=r"$w_\nu$ (solid), $\rho_\nu/\rho_m$ (dashed)", xlim=(1, 1e6), ylim=(1e-5, 1))
ax.invert_xaxis()
ax.legend(loc="lower left")
# (b) delta_ncdm / delta_cdm for m = 0.3 eV
ax = axes[0, 1]
for kk, ls in (("0.01", "-"), ("0.1", "--")):
    for m in MASSES:
        P = runs[m]["pert"][kk]
        ax.semilogx(np.exp(P[:, 0]), P[:, 20] / P[:, 1], color=C[m], ls=ls,
                    label=rf"$m_\nu={m}$ eV" if kk == "0.01" else None)
ax.set(xlabel=r"$a$", ylabel=r"$\delta_\nu/\delta_{\rm cdm}$", xlim=(1e-5, 1), ylim=(-0.1, 1.45))
ax.legend(loc="upper left")
# (c) matter power spectrum
ax = axes[1, 0]
for m in MASSES:
    o = runs[m]
    ax.semilogx(o["k"], o["Pm"] / ref["Pm"] - 1, color=C[m], label=rf"$m_\nu={m}$ eV")
    ax.semilogx(o["k"], o["class"]["Pm"] / ref["class"]["Pm"] - 1, color="#3d3d3a", ls=":", lw=0.9)
    ax.semilogx(o["k"], o["Pcb"] / ref["Pcb"] - 1, color=C[m], ls="--", lw=0.8)
ax.plot([], [], color="#3d3d3a", ls=":", lw=0.9, label="CLASS")
ax.plot([], [], color="#3d3d3a", ls="--", lw=0.8, label=r"CDM+baryons only")
ax.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P_m(k;m_\nu)/P_m(k;0)-1$", xlim=(1e-4, 0.6))
ax.legend(loc="lower left")
# (d) CMB
ax = axes[1, 1]
for m in MASSES:
    o = runs[m]
    ax.semilogx(o["ell"], o["TT"] / ref["TT"] - 1, color=C[m], label=rf"TT, $m_\nu={m}$ eV")
    ax.semilogx(o["ell"], o["ours"]["pp"] / ref["ours"]["pp"] - 1, color=C[m], ls="--")
    ax.semilogx(o["ell"], o["class"]["TT"] / ref["class"]["TT"] - 1, color="#3d3d3a", ls=":", lw=0.9)
    ax.semilogx(o["ell"], o["class"]["pp"] / ref["class"]["pp"] - 1, color="#3d3d3a", ls=":", lw=0.9)
ax.plot([], [], color="#3d3d3a", ls="--", lw=0.9, label=r"$C_\ell^{\phi\phi}$ (dashed)")
ax.plot([], [], color="#3d3d3a", ls=":", lw=0.9, label="CLASS")
ax.set(xlabel=r"$\ell$", ylabel=r"$C_\ell(m_\nu)/C_\ell(0)-1$", xlim=(2, LMAX))
ax.legend(loc="lower left")
for a_, lab in zip(axes.flat, "abcd"):
    a_.text(0.99, 0.99, f"({lab})", transform=a_.transAxes, ha="right", va="top", fontsize=8)
fig.tight_layout()
fig.savefig("report/figures/neutrinos.pdf")
fig.savefig("data/neutrinos.png", dpi=150)

# Numbers
with open("report/data/neutrino_numbers.txt", "w") as fp:
    for m in MASSES:
        o, p = runs[m], runs[m]["p"]
        h = p["h"]
        On = float(re.findall(r"OmegaNcdm\s+= ([\d.e+-]+)", o["log"])[0])
        Mt = float(re.findall(r"m/T today = ([\d.e+-]+)", o["log"])[0])
        f_nu = On / (p["OmegaB"] + p["OmegaCDM"] + On)
        tpert = float(re.findall(r"Perturbations::solve\] ([\d.]+)", o["log"])[0])
        i = np.argmin(np.abs(o["k"] - 0.5))
        sP = o["Pm"][i] / ref["Pm"][i] - 1
        sPc = o["class"]["Pm"][i] / ref["class"]["Pm"][i] - 1
        sPcb = o["Pcb"][i] / ref["Pcb"][i] - 1
        ipp = np.where(o["ell"] == 1000)[0][0]
        spp = o["ours"]["pp"][ipp] / ref["ours"]["pp"][ipp] - 1
        tag = m.replace(".", "p")
        fp.write(f"Omega_ncdm_{tag} {On:.6f}\nomega_ncdm_{tag} {On * h * h:.6f}\nf_nu_{tag} {f_nu:.5f}\n")
        fp.write(f"M_today_{tag} {Mt:.1f}\nz_nr_{tag} {Mt / 3 - 1:.0f}\n")
        fp.write(f"Pm_supp_k0.5_{tag} {sP:.4f}\nPm_supp_k0.5_CLASS_{tag} {sPc:.4f}\nPm_supp_over_fnu_{tag} {sP / f_nu:.2f}\n")
        fp.write(f"Pcb_supp_k0.5_{tag} {sPcb:.4f}\npp_supp_l1000_{tag} {spp:.4f}\n")
        for key in ("TT", "EE", "TTlens", "pp", "Pm", "Pcb"):
            ro, rc = o["ours"][key] / ref["ours"][key], o["class"][key] / ref["class"][key]
            fp.write(f"maxdiff_{key}_{tag} {np.max(np.abs(ro - rc)):.2e}\n")
        fp.write(f"t_pert_{tag} {tpert:.1f}\n")
    fp.write(f"t_pert_0 {float(re.findall(r'Perturbations::solve] ([0-9.]+)', ref['log'])[0]):.1f}\n")
print(open("report/data/neutrino_numbers.txt").read())
