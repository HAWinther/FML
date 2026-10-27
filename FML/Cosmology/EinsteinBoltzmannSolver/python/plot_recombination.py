#!/usr/bin/env python3
"""Plot the output of RecombinationHistory::output.

Usage:  python3 python/plot_recombination.py [data/recombination.txt] [data/recombination.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

infile = sys.argv[1] if len(sys.argv) > 1 else "data/recombination.txt"
outfile = sys.argv[2] if len(sys.argv) > 2 else "data/recombination.png"

cols = ["x", "Xe", "ne", "tau", "dtau", "ddtau", "g", "dg", "ddg", "Tb", "cs", "s", "Xe_saha"]
d = dict(zip(cols, np.loadtxt(infile, unpack=True)))
x = d["x"]
z = np.expm1(-x)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#6b6a63"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "font.size": 10})

# Last scattering (tau = 1) and the Saha -> Peebles switch
x_ls = np.interp(1.0, d["tau"][::-1], x[::-1])
z_ls = np.expm1(-x_ls)

def mark_ls(ax, label=True):
    ax.axvline(x_ls, color="#c3c2b7", lw=0.8, zorder=0)
    if label:
        ax.text(x_ls + 0.05, 0.97, f"$\\tau=1$\n$z_*={z_ls:.0f}$", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=MUTED)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# 1. Free electron fraction
ax = axes[0, 0]
ax.semilogy(x, d["Xe"], color=C[0], label="Saha + Peebles (+ reionization)")
m = d["Xe_saha"] > 1e-6
ax.semilogy(x[m], d["Xe_saha"][m], color=C[1], lw=1.5, ls="--", label="Saha only")
mark_ls(ax)
ax.set(xlabel=r"$x = \ln a$", ylabel=r"$X_e = n_e / n_H$", title="Free electron fraction", ylim=(1e-4, 3))
Xe30 = np.interp(-np.log(31.0), x, d["Xe"])
ax.text(-4.3, 2.5 * Xe30, f"freeze-out\n$X_e(z=30) = {Xe30:.2e}$", fontsize=8, color=MUTED, ha="center", va="bottom")
if d["Xe"][-1] > 0.5:
    ax.text(-1.2, 0.35, "reionization", fontsize=8, color=MUTED, ha="center")
ax.legend(loc="center left")

# 2. Optical depth and derivatives
ax = axes[0, 1]
ax.semilogy(x, d["tau"], color=C[0], label=r"$\tau$")
ax.semilogy(x, -d["dtau"], color=C[1], label=r"$-\tau'$")
ax.semilogy(x, np.abs(d["ddtau"]), color=C[2], label=r"$|\tau''|$")
ax.axhline(1.0, color="#9a9a94", lw=1, ls="--", zorder=0)
mark_ls(ax, label=False)
ax.set(xlabel=r"$x$", ylabel="dimensionless", title="Optical depth", ylim=(1e-8, 1e8))
ax.legend(loc="upper right")

# 3. Visibility function and derivatives (scaled to comparable amplitude)
ax = axes[1, 0]
w = (x > -7.8) & (x < -6.0)
s1 = np.max(np.abs(d["g"][w])) / np.max(np.abs(d["dg"][w]))
s2 = np.max(np.abs(d["g"][w])) / np.max(np.abs(d["ddg"][w]))
ax.plot(x[w], d["g"][w], color=C[0], label=r"$\tilde g$")
ax.plot(x[w], d["dg"][w] * s1, color=C[1], label=rf"$\tilde g' \times {s1:.3g}$")
ax.plot(x[w], d["ddg"][w] * s2, color=C[2], label=rf"$\tilde g'' \times {s2:.3g}$")
ax.axhline(0.0, color="#9a9a94", lw=0.8, zorder=0)
mark_ls(ax, label=False)
ax.set(xlabel=r"$x$", ylabel=r"$\tilde g$ [per unit $x$]",
       title=r"Visibility function $\tilde g = -\tau' e^{-\tau}$")
ax.legend(loc="upper left")

# 4. Baryon temperature
ax = axes[1, 1]
Tg = 2.7255 * (1 + z)
ax.semilogx(1 + z, d["Tb"] / Tg, color=C[0])
ax.axvline(1 + z_ls, color="#c3c2b7", lw=0.8, zorder=0)
ax.set(xlabel=r"$1 + z$", ylabel=r"$T_b / T_\gamma$", title="Baryon temperature", xlim=(1, 1e4))
ax.invert_xaxis()
ax.text(0.03, 0.08, r"$T_b \propto a^{-2}$ after thermal decoupling", transform=ax.transAxes,
        fontsize=8, color=MUTED)

fig.suptitle("RecombinationHistory (hydrogen + helium, reionization)", y=0.995)
fig.tight_layout()
fig.savefig(outfile, dpi=130)
print(f"Wrote {outfile}")
