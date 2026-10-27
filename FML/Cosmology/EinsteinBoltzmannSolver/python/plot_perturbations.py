#!/usr/bin/env python3
"""Plot data/perturbations_k*.txt (written by main) and, if classy is installed, overlay CLASS.

CLASS is run in Newtonian gauge with the parameters in data/params.txt. Its normalisation
(curvature R = 1) coincides with ours (Phi = 2/3 initially, no neutrinos); signs differ:
  delta_ours = -delta_CLASS,  Phi_ours = phi_CLASS,  Psi_ours = -psi_CLASS,  Theta0 = -delta_g / 4.
CLASS includes polarization and helium, and switches photons to a radiation-streaming
approximation after recombination, so Theta_0 differs at late times (not used by the LOS integral).

Usage:  python3 python/plot_perturbations.py [--no-class] [data/perturbations.png]
"""
import glob
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

use_class = "--no-class" not in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
outfile = args[0] if args else "data/perturbations.png"

files = sorted(glob.glob("data/perturbations_k*.txt"), key=lambda f: float(re.findall(r"k([\d.]+)\.txt", f)[0]))
cols = ["x", "dc", "vc", "db", "vb", "Phi", "Psi", "dPhi", "Th0", "Th1", "Th2", "S"]
data = []
for f in files:
    k = float(re.findall(r"k([\d.]+)\.txt", f)[0])
    data.append((k, dict(zip(cols, np.loadtxt(f, unpack=True)))))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "font.size": 10})

cls = {}
if use_class:
    try:
        from classy import Class

        sys.path.insert(0, "python")
        from params import class_params, read_params

        cp = class_params(read_params())
        cp.update({"output": "mTk", "gauge": "newtonian", "k_output_values": ",".join(str(k) for k, _ in data)})
        cl = Class()
        cl.set(cp)
        cl.compute()
        for (k, _), pk in zip(data, cl.get_perturbations()["scalar"]):
            cls[k] = {"x": np.log(pk["a"]), "dc": -pk["delta_cdm"], "db": -pk["delta_b"],
                      "Phi": pk["phi"], "Psi": -pk["psi"], "Th0": -0.25 * pk["delta_g"]}
        print("CLASS overlay: dashed lines")
    except Exception as e:  # classy missing or failed
        print(f"No CLASS overlay ({e})")

# Last scattering from the recombination output
rec = np.loadtxt("data/recombination.txt")
x_ls = np.interp(1.0, rec[::-1, 3], rec[::-1, 0])

fig, axes = plt.subplots(3, 2, figsize=(11, 12))
panels = [
    (axes[0, 0], r"$|\delta_{\rm CDM}|$ (solid), $|\delta_b|$ (thin)", "log"),
    (axes[0, 1], r"$|v_{\rm CDM}|$ (solid), $|v_b|$ (thin)", "log"),
    (axes[1, 0], r"$\Theta_0$", "linear"),
    (axes[1, 1], r"$\Theta_1$", "linear"),
    (axes[2, 0], r"$\Phi$ (solid), $\Phi + \Psi$ (thin)", "linear"),
    (axes[2, 1], r"Source $\tilde S(k, x)$", "linear"),
]
for i, (k, d) in enumerate(data):
    col, lab = C[i % len(C)], f"k = {k:g}/Mpc"
    x = d["x"]
    ax = axes[0, 0]; ax.semilogy(x, np.abs(d["dc"]), color=col, label=lab); ax.semilogy(x, np.abs(d["db"]), color=col, lw=0.9)
    ax = axes[0, 1]; ax.semilogy(x, np.abs(d["vc"]), color=col, label=lab); ax.semilogy(x, np.abs(d["vb"]), color=col, lw=0.9)
    axes[1, 0].plot(x, d["Th0"], color=col, label=lab)
    axes[1, 1].plot(x, d["Th1"], color=col, label=lab)
    ax = axes[2, 0]; ax.plot(x, d["Phi"], color=col, label=lab); ax.plot(x, d["Phi"] + d["Psi"], color=col, lw=0.9)
    m = x > -8
    axes[2, 1].plot(x[m], d["S"][m], color=col, label=lab)
    if k in cls:
        c = cls[k]
        kw = dict(color="#3d3d3a", lw=0.9, ls="--")
        axes[0, 0].semilogy(c["x"], np.abs(c["dc"]), **kw)
        axes[1, 0].plot(c["x"], c["Th0"], **kw)
        axes[2, 0].plot(c["x"], c["Phi"], **kw)

for ax, title, _ in panels:
    ax.set(title=title, xlabel=r"$x = \ln a$")
    ax.axvline(x_ls, color="#c3c2b7", lw=0.8, zorder=0)
axes[1, 0].set_xlim(-12, 0); axes[1, 1].set_xlim(-12, 0)
axes[2, 1].set_xlim(-8, 0)
axes[0, 0].legend(loc="upper left")
if cls:
    axes[1, 0].plot([], [], color="#3d3d3a", lw=0.9, ls="--", label="CLASS")
    axes[1, 0].legend(loc="lower left")

fig.suptitle("Perturbations (temperature only, Newtonian gauge; vertical line: last scattering)", y=0.995)
fig.tight_layout()
fig.savefig(outfile, dpi=120)
print(f"Wrote {outfile}")

if cls:
    # CLASS starts small-k modes late (x ~ -8.7 for k = 0.001/Mpc) with approximate initial
    # conditions that relax within ~1 e-fold, so compare after x = -7.
    print("\nmax |ours - CLASS| / max |CLASS| for x > -7:")
    for k, d in data:
        c = cls[k]
        m = d["x"] > max(-7.0, c["x"][0] + 1.5)
        out = []
        for key in ["dc", "db", "Phi"]:
            cv = np.interp(d["x"][m], c["x"], c[key])
            out.append(f"{key} {np.max(np.abs(d[key][m] - cv)) / np.max(np.abs(cv)):.1e}")
        print(f"  k = {k:g}/Mpc: " + "  ".join(out))
