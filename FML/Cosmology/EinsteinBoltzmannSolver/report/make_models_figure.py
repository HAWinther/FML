#!/usr/bin/env python3
"""Figures and numbers for the model-library section of the report.

Usage (from the project root): run all models into data/models/<name> (see README, "Models beyond LCDM"), then
  python3 report/make_models_figure.py
Writes report/figures/models_de.pdf, report/figures/models_mg.pdf and report/data/models_numbers.txt
(sigma_8, P(k) and C_ell ratios to LCDM, H(z) deviation).
"""
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.3, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm"})
BASE = "data/models"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#c03f8e", "#8a6d1f", "#5b4bd6", "#d1a21b"]
GROUPS = {
    "de": [("quintessence", r"quintessence ($\lambda=1$)"),
           ("coupled_quintessence", r"coupled quint. ($\lambda=0.5$, $\beta=0.05$)"),
           ("dark_scattering", r"dark scattering ($w=-0.9$, $\xi=10$ b/GeV)"),
           ("ede", r"EDE ($f_{\rm EDE}=0.12$, $z_c=3630$)"),
           ("ide", r"IDE ($\xi=-0.1$, $w=-0.95$)")],
    "mg": [("musigma", r"$\mu$-$\Sigma$ ($\mu_0=0.3$, $\Sigma_0=0.1$)"),
           ("ndgp", r"nDGP ($H_0r_c/c=1$, $\Lambda$CDM bg.)"),
           ("sdgp", "sDGP (no dark energy)"),
           ("jbd", r"Brans-Dicke ($\omega=100$)"),
           ("fr", r"$f(R)$ ($|f_{R0}|=10^{-5}$)"),
           ("cubic_galileon", "cubic Galileon (tracker)"),
           ("horndeski", r"Horndeski ($c_K,c_B,c_M=1,0.6,0.3$)")],
}


def load(name):
    d = os.path.join(BASE, name)
    if not os.path.isfile(os.path.join(d, "cells_lensed.txt")):
        return None
    return {"bg": np.loadtxt(os.path.join(d, "background.txt")), "cl": np.loadtxt(os.path.join(d, "cells.txt")),
            "len": np.loadtxt(os.path.join(d, "cells_lensed.txt")), "pk": np.loadtxt(os.path.join(d, "matter_power.txt"))}


def sigma8(pk):
    k, P = pk[:, 0], pk[:, 1]
    kr = k * 8.0
    W = 3 * (np.sin(kr) - kr * np.cos(kr)) / kr**3
    return np.sqrt(np.trapezoid(k**3 * P * W**2 / (2 * np.pi**2), np.log(k)))


ref = load("lcdm")
num = {"lcdm_sigma8": sigma8(ref["pk"])}
for g, models in GROUPS.items():
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for i, (name, lab) in enumerate(models):
        m = load(name)
        if m is None:
            print("missing", name)
            continue
        st = {"color": COL[i % len(COL)], "label": lab}
        x = ref["bg"][:, 0]
        sel = (x > -np.log(1 + 3e4)) & (x <= 0)
        zz = np.exp(-x[sel]) - 1
        Hr = np.interp(x[sel], m["bg"][:, 0], m["bg"][:, 3]) / ref["bg"][sel, 3]
        ax[0, 0].semilogx(1 + zz, Hr, **st)
        n = min(len(m["cl"]), len(ref["cl"]))
        ell = ref["cl"][:n, 0]
        rtt = m["cl"][:n, 1] / ref["cl"][:n, 1]
        ax[0, 1].semilogx(ell, rtt, **st)
        n2 = min(len(m["len"]), len(ref["len"]))
        rpp = m["len"][:n2, 5] / ref["len"][:n2, 5]
        ax[1, 0].semilogx(ref["len"][:n2, 0], rpp, **st)
        k = ref["pk"][:, 0]
        rpk = np.interp(k, m["pk"][:, 0], m["pk"][:, 1]) / ref["pk"][:, 1]
        ax[1, 1].semilogx(k, rpk, **st)
        num[f"{name}_sigma8_ratio"] = sigma8(m["pk"]) / num["lcdm_sigma8"]
        num[f"{name}_Pk_ratio_k0.1"] = np.interp(0.1, k, rpk)
        num[f"{name}_Pk_ratio_k0.001"] = np.interp(0.001, k, rpk)
        num[f"{name}_TT_ratio_l10"] = np.interp(10, ell, rtt)
        num[f"{name}_TT_maxdev_l100_2000"] = np.max(np.abs(rtt[(ell >= 100) & (ell <= 2000)] - 1))
        num[f"{name}_phiphi_ratio_l100"] = np.interp(100, ref["len"][:n2, 0], rpp)
        num[f"{name}_H_maxdev"] = np.max(np.abs(Hr - 1))
    ax[0, 0].set(xlabel="$1+z$", ylabel=r"$H/H_{\Lambda\rm CDM}$")
    ax[0, 1].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{TT}/C_\ell^{TT,\Lambda\rm CDM}$ (unlensed)", xlim=(2, 2500))
    ax[1, 0].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{\phi\phi}/C_\ell^{\phi\phi,\Lambda\rm CDM}$", xlim=(2, 2000))
    ax[1, 1].set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k)/P_{\Lambda\rm CDM}(k)$, $z=0$")
    for a in ax.flat:
        a.axhline(1.0, color="k", lw=0.5)
    ax[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"report/figures/models_{g}.pdf")
with open("report/data/models_numbers.txt", "w") as fp:
    for k, v in num.items():
        fp.write(f"{k} {v:.5g}\n")
print(open("report/data/models_numbers.txt").read())
