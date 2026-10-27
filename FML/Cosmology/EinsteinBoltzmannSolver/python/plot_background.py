#!/usr/bin/env python3
"""Plot the output of BackgroundCosmology::output.

Usage:  python3 python/plot_background.py [data/background.txt] [data/background.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

infile = sys.argv[1] if len(sys.argv) > 1 else "data/background.txt"
outfile = sys.argv[2] if len(sys.argv) > 2 else "data/background.png"

cols = ["x", "eta", "t", "H", "Hp", "dHp", "ddHp", "OmB", "OmCDM", "OmG", "OmNu", "OmK", "OmDE",
        "chi", "dA", "dL"]
d = dict(zip(cols, np.loadtxt(infile, unpack=True)))
x = d["x"]
c_kms = 299792.458
Gyr_per_Mpc_over_c = 3.0856775814913673e19 / c_kms / (365.25 * 24 * 3600 * 1e9)  # (Mpc / c) in Gyr

# Fixed categorical order (blue, orange, aqua, yellow), muted reference lines
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REF = dict(color="#9a9a94", lw=1, ls="--", zorder=0)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "font.size": 10})

OmR = d["OmG"] + d["OmNu"]
OmM = d["OmB"] + d["OmCDM"]
# Equality epochs from the table
x_rm = x[np.argmin(np.abs(np.log(OmM / OmR)))]
x_mL = x[np.argmin(np.abs(np.log(OmM / np.maximum(d["OmDE"], 1e-300))))]

def mark_epochs(ax):
    for xe in (x_rm, x_mL):
        ax.axvline(xe, color="#c3c2b7", lw=0.8, zorder=0)

fig, axes = plt.subplots(3, 2, figsize=(11, 11))

# 1. Density parameters
ax = axes[0, 0]
ax.plot(x, OmR, color=C[0], label=r"$\Omega_r = \Omega_\gamma + \Omega_\nu$")
ax.plot(x, OmM, color=C[1], label=r"$\Omega_m = \Omega_b + \Omega_{\rm CDM}$")
ax.plot(x, d["OmDE"], color=C[2], label=r"$\Omega_\Lambda$")
if np.any(np.abs(d["OmK"]) > 1e-10):
    ax.plot(x, d["OmK"], color=C[3], label=r"$\Omega_K$")
mark_epochs(ax)
ax.set(xlabel=r"$x = \ln a$", ylabel=r"$\Omega_i(x)$", title="Density parameters", ylim=(-0.02, 1.05))
ax.legend(loc="center left")

# 2. Conformal Hubble rate
ax = axes[0, 1]
ax.semilogy(x, d["Hp"], color=C[0])
mark_epochs(ax)
ax.set(xlabel=r"$x$", ylabel=r"$\mathcal{H}(x)$  [100 km/s/Mpc]", title=r"Conformal Hubble rate $\mathcal{H} = aH$")

# 3. Logarithmic derivatives of Hp, with the analytic values in each era
ax = axes[1, 0]
ax.plot(x, d["dHp"] / d["Hp"], color=C[0], label=r"$\mathcal{H}'/\mathcal{H}$")
ax.plot(x, d["ddHp"] / d["Hp"], color=C[1], label=r"$\mathcal{H}''/\mathcal{H}$")
for v in (-1, -0.5, 0.25, 1):
    ax.axhline(v, **REF)
mark_epochs(ax)
ax.text(x[0] + 0.5, -0.92, "radiation: $-1$, $1$", fontsize=8, color="#6b6a63")
ax.text(x_rm + 0.4, -0.42, r"matter: $-\frac{1}{2}$, $\frac{1}{4}$", fontsize=8, color="#6b6a63")
ax.set(xlabel=r"$x$", ylabel="dimensionless", title=r"Derivatives of $\mathcal{H}$ (dashed: analytic limits)")
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)

# 4. eta * Hp / c
ax = axes[1, 1]
ax.plot(x, d["eta"] * d["Hp"] * 100.0 / c_kms, color=C[0])
for v in (1, 2):
    ax.axhline(v, **REF)
mark_epochs(ax)
ax.set(xlabel=r"$x$", ylabel=r"$\eta\,\mathcal{H}/c$", title=r"$\eta\mathcal{H}/c$  (1 in RD, 2 in MD)")
ax.set_yscale("log")

# 5. Cosmic time and conformal time, both in Gyr
ax = axes[2, 0]
ax.semilogy(x, d["t"], color=C[0], label=r"$t$")
ax.semilogy(x, d["eta"] * Gyr_per_Mpc_over_c, color=C[1], label=r"$\eta/c$")
mark_epochs(ax)
t0, eta0 = np.interp(0.0, x, d["t"]), np.interp(0.0, x, d["eta"]) * Gyr_per_Mpc_over_c
ax.text(0.03, 0.95, f"$t_0$ = {t0:.3f} Gyr\n$\\eta_0/c$ = {eta0:.2f} Gyr", transform=ax.transAxes,
        va="top", fontsize=9, color="#3d3d3a")
ax.set(xlabel=r"$x$", ylabel="time [Gyr]", title="Cosmic and conformal time")
ax.legend(loc="lower right")

# 6. Distances vs redshift
ax = axes[2, 1]
m = (x <= 0) & (x >= -np.log(1 + 5.0))
z = np.exp(-x[m]) - 1
ax.plot(z, d["dL"][m] / 1e3, color=C[0], label=r"$d_L$")
ax.plot(z, d["chi"][m] / 1e3, color=C[1], label=r"$\chi$")
ax.plot(z, d["dA"][m] / 1e3, color=C[2], label=r"$d_A$")
ax.set(xlabel=r"redshift $z$", ylabel="distance [Gpc]", title="Distances", xlim=(0, 5))
ax.legend(loc="upper left")

fig.suptitle("BackgroundCosmology  (vertical lines: matter-radiation and matter-$\\Lambda$ equality)", y=0.995)
fig.tight_layout()
fig.savefig(outfile, dpi=130)
print(f"Wrote {outfile}")
