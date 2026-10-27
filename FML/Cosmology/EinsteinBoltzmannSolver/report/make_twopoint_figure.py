#!/usr/bin/env python3
"""Figure and numbers for the two-point-function section of the report.

Usage (from the project root):
  ./cmb params/twopoint.ini > data/log_twopoint.txt
  python3 report/make_twopoint_figure.py
Reads data/twopoint/, runs CLASS (sigma_8, P(k) for xi(r), number-count C_ell) and writes
report/figures/twopoint.pdf and report/data/twopoint_numbers.txt.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

sys.path.insert(0, "python")
from params import class_params, read_params  # noqa: E402

D = "data/twopoint/"
p = read_params(D + "params.txt")
h = p["h"]
ZS = [0.5, 1.0]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#3d3d3a"]
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm"})

cu = np.loadtxt(D + "cmb_correlation_unlensed.txt")
cl = np.loadtxt(D + "cmb_correlation_lensed.txt")
cb = np.loadtxt(D + "cmb_correlation_lensed_beam1deg.txt")
xo = np.loadtxt(D + "matter_correlation.txt")
s8 = float(open(D + "matter_correlation.txt").readline().split("sigma_8 =")[1].split()[0])

cp = class_params(p)
cp.update({"output": "mPk,nCl", "P_k_max_1/Mpc": 50.0, "selection": "gaussian",
           "selection_mean": ", ".join(str(z) for z in ZS), "selection_width": 0.1,
           "number_count_contributions": "density", "non_diagonal": 0, "l_max_lss": 2000,
           "l_switch_limber_for_nc_local_over_z": 1000.0, "l_switch_limber_for_nc_los_over_z": 1000.0})
c = Class()
c.set(cp)
c.compute()
kk = np.logspace(-5, np.log10(49.0), 3000)
f = CubicSpline(np.log(kk), np.log(kk * np.array([c.pk_lin(k, 0.0) for k in kk])))
r_cl = np.array([20, 40, 60, 80, 90, 95, 100, 105, 110, 120, 140, 160, 180, 200.0])
xi_cl = []
for r_h in r_cl:
    r = r_h / h
    edges = np.concatenate([[kk[0]], np.logspace(-3, np.log10(kk[-1]), 400)])
    xi_cl.append(sum(quad(lambda k: np.exp(f(np.log(k))), a, b, weight="sin", wvar=r, limit=200)[0]
                     for a, b in zip(edges[:-1], edges[1:])) / (2 * np.pi**2 * r))
xi_cl = np.array(xi_cl)

fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8))
a = ax[0, 0]
m = cu[:, 0] > 0
a.semilogx(cu[m, 0], cu[m, 1], color=C[0], label="unlensed")
a.semilogx(cl[m, 0], cl[m, 1], color=C[1], ls="--", label="lensed")
a.semilogx(cb[m, 0], cb[m, 1], color=C[2], label=r"lensed, $1^\circ$ beam")
a.axhline(0, color="#3d3d3a", lw=0.6)
a.set(xlabel=r"$\theta$ [deg]", ylabel=r"$C^{TT}(\theta)$ [$\mu$K$^2$]", xlim=(0.01, 180))
a.legend(loc="upper right")
a = ax[0, 1]
m = cu[:, 0] <= 5
a.plot(cu[m, 0], cu[m, 2], color=C[0], label=r"$\xi_+=\langle QQ\rangle+\langle UU\rangle$")
a.plot(cu[m, 0], cu[m, 3], color=C[1], label=r"$\xi_-=\langle QQ\rangle-\langle UU\rangle$")
a.plot(cu[m, 0], cu[m, 4], color=C[2], label=r"$\xi_X=\langle TQ\rangle$")
a.axhline(0, color="#3d3d3a", lw=0.6)
a.set(xlabel=r"$\theta$ [deg]", ylabel=r"[$\mu$K$^2$]", xlim=(0, 5))
a.legend(loc="upper right")
a = ax[1, 0]
m = (xo[:, 0] >= 20)
a.plot(xo[m, 0], xo[m, 2], color=C[0], label=r"this work, $z=0$")
a.plot(r_cl, r_cl**2 * xi_cl, "o", ms=2.5, color=C[3], label=r"from CLASS $P(k)$")
a.set(xlabel=r"$r$ [Mpc/$h$]", ylabel=r"$r^2\xi(r)$ [(Mpc/$h)^2$]", xlim=(20, 200))
a.legend(loc="upper right")
a = ax[1, 1]
for z0, col in zip(ZS, C):
    w = np.loadtxt(D + f"angular_w_z{z0:g}.txt")
    a.semilogx(w[:, 0], w[:, 0] ** 2 * w[:, 1] * 1e3, color=col, label=rf"$z={z0:g}\pm0.1$")
a.axhline(0, color="#3d3d3a", lw=0.6)
a.set(xlabel=r"$\theta$ [deg]", ylabel=r"$10^3\,\theta^2 w(\theta)$ [deg$^2$]", xlim=(0.05, 10))
a.legend(loc="lower left")
for a_, lab in zip(ax.flat, "abcd"):
    a_.text(0.02, 0.98, f"({lab})", transform=a_.transAxes, ha="left", va="top", fontsize=8)
fig.tight_layout()
fig.savefig("report/figures/twopoint.pdf")
fig.savefig("data/twopoint.png", dpi=150)

# Numbers
th = cu[:, 0]
zc = th[np.where(np.diff(np.sign(cu[:, 1])) != 0)[0]]
win = (xo[:, 0] > 80) & (xo[:, 0] < 120)
ipk = np.argmax(xo[win, 2])
peak = xo[win, 2][ipk]
xi_o = np.interp(r_cl, xo[:, 0], xo[:, 1])
dpk = np.max(np.abs(r_cl**2 * (xi_o - xi_cl))[r_cl >= 60]) / peak
dcl = c.density_cl(2000)["dd"]
with open("report/data/twopoint_numbers.txt", "w") as fp:
    fp.write(f"CTT0_unlensed {cu[0, 1]:.1f}\nCTT0_lensed {cl[0, 1]:.1f}\nCTT0_beam1deg {cb[0, 1]:.1f}\n")
    fp.write(f"sigmaT {np.sqrt(cu[0, 1]):.1f}\nCTT_zero1 {zc[0]:.1f}\nCTT_zero2 {zc[1]:.1f}\n")
    fp.write(f"CTT_min {cu[:, 1].min():.1f}\nCTT_min_theta {th[np.argmin(cu[:, 1])]:.1f}\nCTT_180 {cu[-1, 1]:.1f}\n")
    dl = np.abs(cl[:, 1] - cu[:, 1])
    fp.write(f"lens_maxdiff {dl.max():.2f}\nlens_maxdiff_theta {th[np.argmax(dl)]:.2f}\n")
    fp.write(f"xiplus0 {cu[0, 2]:.2f}\nxiX_max {cu[:, 4].max():.2f}\nxiX_max_theta {th[np.argmax(cu[:, 4])]:.2f}\n")
    fp.write(f"xiX_min {cu[:, 4].min():.2f}\nxiX_min_theta {th[np.argmin(cu[:, 4])]:.2f}\n")
    fp.write(f"sigma8 {s8:.5f}\nsigma8_CLASS {c.sigma8():.5f}\n")
    fp.write(f"xi_peak_r {xo[win, 0][ipk]:.1f}\nxi_peak_r2xi {peak:.2f}\nxi_maxdiff_over_peak_r60 {dpk:.2e}\n")
    fp.write(f"xi_rel_r20 {xi_o[0] / xi_cl[0] - 1:+.2e}\n")
    for i, z0 in enumerate(ZS):
        o = np.loadtxt(D + f"angular_cl_z{z0:g}.txt")
        ell = o[:, 0].astype(int)
        r = o[:, 1] / dcl[f"dens[{i + 1}]-dens[{i + 1}]"][ell] - 1
        for lo, hi in ((2, 200), (200, 500), (500, 2000)):
            s = (ell >= lo) & (ell <= hi)
            fp.write(f"ncl_maxrel_z{z0:g}_l{lo}-{hi} {np.max(np.abs(r[s])):.2e}\n")
        w = np.loadtxt(D + f"angular_w_z{z0:g}.txt")
        t, y = w[:, 0], w[:, 0] ** 2 * w[:, 1]
        mm = (t > 1.5) & (t < 4)
        loc = [t[mm][j] for j in range(1, mm.sum() - 1) if y[mm][j] > y[mm][j - 1] and y[mm][j] > y[mm][j + 1]]
        fp.write(f"w_bao_theta_z{z0:g} {loc[0] if loc else float('nan'):.2f}\n")
        fp.write(f"w_zero_z{z0:g} {t[np.where(np.diff(np.sign(w[:, 1])) != 0)[0][0]]:.2f}\n")
print(open("report/data/twopoint_numbers.txt").read())
