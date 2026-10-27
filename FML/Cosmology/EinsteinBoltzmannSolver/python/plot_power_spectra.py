#!/usr/bin/env python3
"""Plot data/cells.txt (TT, EE, TE) and data/matter_power.txt; overlay CLASS if classy is installed.

CLASS is run with the parameters in data/params.txt (same tanh reionization) and no lensing,
so the physics matches ours except for the recombination model (HyRec vs our Peebles
equation + Saha helium).

Usage:  python3 python/plot_power_spectra.py [--no-class] [data/power_spectra.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

use_class = "--no-class" not in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
outfile = args[0] if args else "data/power_spectra.png"

C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#3d3d3a", "#6b6a63"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "font.size": 10})

ell, TT, EE, TE = np.loadtxt("data/cells.txt", unpack=True)
kP, P = np.loadtxt("data/matter_power.txt", unpack=True)
has_pol = np.any(EE != 0)

cls = None
if use_class:
    try:
        from classy import Class

        sys.path.insert(0, "python")
        from params import class_params, read_params

        p = read_params()
        cp = class_params(p)
        cp.update({"output": "tCl,pCl,mPk", "lensing": "no", "l_max_scalars": int(ell[-1]) + 100,
                   "P_k_max_1/Mpc": 1.0})
        cl = Class()
        cl.set(cp)
        cl.compute()
        raw = cl.raw_cl(int(ell[-1]))
        f = raw["ell"][2:] * (raw["ell"][2:] + 1) / (2 * np.pi) * (p["TCMB"] * 1e6) ** 2
        cls = {"TT": f * raw["tt"][2:], "EE": f * raw["ee"][2:], "TE": f * raw["te"][2:],
               "P": np.array([cl.pk(k * p["h"], 0.0) * p["h"] ** 3 for k in kP])}
        print("CLASS overlay: dashed")
    except Exception as e:
        print(f"No CLASS overlay ({e})")

fig, axes = plt.subplots(3, 2, figsize=(11, 11.5))
kw_class = dict(color=INK, lw=1, ls="--", label="CLASS")

def spectrum(ax, y, key, col, title, log_x=True, legend_loc="upper left"):
    ax.plot(ell, y, color=col, label="ours")
    if cls:
        ax.plot(ell, cls[key], **kw_class)
    ax.set(xscale="log" if log_x else "linear", xlabel=r"$\ell$", xlim=(2, ell[-1]), title=title,
           ylabel=r"$\ell(\ell+1)C_\ell/2\pi$  [$\mu$K$^2$]")
    ax.legend(loc=legend_loc)

def ratio(ax, y, key, col, title):
    if not cls:
        ax.set_axis_off()
        return
    ax.plot(ell, y / cls[key], color=col)
    ax.axhline(1.0, color="#9a9a94", lw=1, ls="--", zorder=0)
    ax.set(xscale="log", xlabel=r"$\ell$", ylabel="ours / CLASS", xlim=(2, ell[-1]), title=title, ylim=(0.9, 1.1))

spectrum(axes[0, 0], TT, "TT", C[0], "TT (unlensed)")
ratio(axes[0, 1], TT, "TT", C[0], "TT ratio")
if has_pol:
    spectrum(axes[1, 0], EE, "EE", C[1], "EE")
    axes[1, 0].set_yscale("log")
    ratio(axes[1, 1], EE, "EE", C[1], "EE ratio")
    spectrum(axes[2, 0], TE, "TE", C[2], "TE", log_x=False, legend_loc="lower right")
    axes[2, 0].axhline(0.0, color="#9a9a94", lw=0.8, zorder=0)
else:
    for ax in (axes[1, 0], axes[1, 1], axes[2, 0]):
        ax.set_axis_off()
for ax in (axes[0, 1], axes[1, 1]):
    if cls:
        ax.text(0.03, 0.06, "Residuals of ~1-2%: recombination model\n(our Peebles + Saha helium vs HyRec)",
                transform=ax.transAxes, fontsize=8, color=MUTED)

ax = axes[2, 1]
ax.loglog(kP, P, color=C[0], label="ours")
if cls:
    ax.loglog(kP, cls["P"], **kw_class)
    r = P / cls["P"]
    ax.text(0.03, 0.05, f"ours/CLASS - 1 within [{r.min() - 1:+.1e}, {r.max() - 1:+.1e}]",
            transform=ax.transAxes, fontsize=8, color=MUTED)
ax.set(xlabel=r"$k$  [$h$/Mpc]", ylabel=r"$P(k)$  [(Mpc/$h$)$^3$]", title="Matter power spectrum today")
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(outfile, dpi=120)
print(f"Wrote {outfile}")
