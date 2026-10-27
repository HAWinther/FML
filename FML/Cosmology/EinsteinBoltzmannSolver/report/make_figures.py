#!/usr/bin/env python3
"""Make all figures for the report from report/data/ (written by ./milestones).

Usage (from the project root):  python3 report/make_figures.py [--no-class]
The validation figure needs classy (pip install classy); skipped with --no-class.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

D = "report/data/"
F = "report/figures/"
os.makedirs(F, exist_ok=True)

# ---------------------------------------------------------------------------
# Style: one-column figures for REVTeX (3.4 in wide), fixed categorical colour order
# ---------------------------------------------------------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6b6a63"]
REF = dict(color="#9a9a94", lw=0.8, ls="--", zorder=0)
EPOCH = dict(color="#c3c2b7", lw=0.8, zorder=0)
W1, W2 = 3.4, 7.0
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "font.family": "serif",
    "mathtext.fontset": "cm",
})

def numbers(name):
    out = {}
    for line in open(D + name):
        k, v = line.split()
        out[k] = float(v)
    return out

def save(fig, name):
    fig.savefig(F + name + ".pdf")
    plt.close(fig)
    print("wrote", F + name + ".pdf")

N1, N2, N3, N4 = (numbers(f"m{i}_numbers.txt") for i in (1, 2, 3, 4))
c_kms = 299792.458
Mpc_over_c_Gyr = 3.0856775814913673e19 / c_kms / (365.25 * 24 * 3600 * 1e9)
x_rm, x_ml, x_acc = N1["radiation_matter_equality_x"], N1["matter_lambda_equality_x"], N1["acceleration_onset_x"]

# Epoch markers (same in all Milestone I figures): solid = radiation-matter equality,
# dashed = matter-Lambda equality, dotted = onset of acceleration
EPOCH_STYLE = {"rm": "-", "ml": "--", "acc": ":"}
def mark_epochs(ax, labels=False, which=("rm", "ml", "acc")):
    for key, xe in [("rm", x_rm), ("ml", x_ml), ("acc", x_acc)]:
        if key in which:
            ax.axvline(xe, color="#8f8e86", lw=0.8, ls=EPOCH_STYLE[key], zorder=0)

# ===========================================================================
# Milestone I
# ===========================================================================
cols = ["x", "eta", "t", "H", "Hp", "dHp", "ddHp", "OmB", "OmCDM", "OmG", "OmNu", "OmK", "OmDE", "chi", "dA", "dL"]
bg = dict(zip(cols, np.loadtxt(D + "background.txt", unpack=True)))
x = bg["x"]

# Test plot 1: derivatives of Hp
fig, ax = plt.subplots(figsize=(W1, 2.4))
ax.plot(x, bg["dHp"] / bg["Hp"], color=C[0], label=r"$\mathcal{H}'/\mathcal{H}$")
ax.plot(x, bg["ddHp"] / bg["Hp"], color=C[1], label=r"$\mathcal{H}''/\mathcal{H}$")
for xa, xb, v1, v2 in [(-20, x_rm, -1, 1), (x_rm, x_ml, -0.5, 0.25), (x_ml, 5, 1, 1)]:
    ax.hlines([v1, v2], xa, xb, colors="#3d3d3a", linestyles=":", lw=1)
mark_epochs(ax, labels=True)
ax.set(xlabel=r"$x=\ln a$", ylabel="dimensionless", xlim=(-20, 5), ylim=(-1.2, 1.4))
ax.text(-19, -0.88, "radiation: $-1$, $1$", fontsize=6.5, color="#3d3d3a")
ax.text(-5.9, -0.38, r"matter: $-\frac{1}{2}$, $\frac{1}{4}$", fontsize=6.5, color="#3d3d3a")
ax.text(1.0, 0.82, r"$\Lambda$: $1$, $1$", fontsize=6.5, color="#3d3d3a")
ax.legend(loc="center left")
save(fig, "m1_Hp_derivatives")

# Test plot 2: eta Hp / c and eta' Hp / c from the spline derivative
eta_s = CubicSpline(x, bg["eta"])
Hp_SI_over_c_perMpc = bg["Hp"] * 100.0 / c_kms  # Hp/c in 1/Mpc
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.semilogy(x, bg["eta"] * Hp_SI_over_c_perMpc, color=C[0], label=r"$\eta\mathcal{H}/c$")
ax.semilogy(x, eta_s(x, 1) * Hp_SI_over_c_perMpc, color=C[1], ls="--", label=r"$\eta'\mathcal{H}/c$ (spline derivative)")
ax.axhline(1, **REF)
ax.axhline(2, **REF)
mark_epochs(ax)
ax.text(-19.5, 1.12, "RD: 1", fontsize=6.5, color="#3d3d3a")
ax.text(-6.0, 2.25, "MD: 2", fontsize=6.5, color="#3d3d3a")
ax.set(xlabel=r"$x=\ln a$", ylabel=r"$\eta\mathcal{H}/c$", xlim=(-20, 5))
ax.legend(loc="upper left")
save(fig, "m1_etaHp")
print("  max |eta' Hp/c - 1| =", np.max(np.abs(eta_s(x, 1) * Hp_SI_over_c_perMpc - 1)))

# Conformal Hubble factor
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.semilogy(x, bg["Hp"], color=C[0])
mark_epochs(ax, labels=True)
ax.set(xlabel=r"$x=\ln a$", ylabel=r"$\mathcal{H}(x)$ [100 km/s/Mpc]", xlim=(-20, 5))
save(fig, "m1_Hp")

# Times
fig, ax = plt.subplots(figsize=(W1, 2.3))
l1, = ax.semilogy(x, bg["t"], color=C[0], label=r"cosmic time $t$")
l2, = ax.semilogy(x, bg["eta"] * Mpc_over_c_Gyr, color=C[1], label=r"conformal time $\eta/c$")
mark_epochs(ax)
ax.set(xlabel=r"$x=\ln a$", ylabel="time [Gyr]", xlim=(-20, 5))
ax.legend(handles=[l1, l2], loc="lower right")
save(fig, "m1_times")

# Density parameters
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(x, bg["OmB"] + bg["OmCDM"], color=C[1], label=r"$\Omega_b+\Omega_{\rm CDM}$")
ax.plot(x, bg["OmG"] + bg["OmNu"], color=C[0], label=r"$\Omega_\gamma+\Omega_\nu$")
ax.plot(x, bg["OmDE"], color=C[2], label=r"$\Omega_\Lambda$")
mark_epochs(ax, which=("rm", "ml"))
ax.set(xlabel=r"$x=\ln a$", ylabel=r"$\Omega_i(x)$", xlim=(-20, 5), ylim=(-0.02, 1.05))
ax.legend(loc="center left")
save(fig, "m1_Omegas")

# Supernovae: d_L / z
sn = np.loadtxt(D + "../../external/supernovadata.txt")
zsn, dlsn, err = sn[:, 0], sn[:, 1], sn[:, 2]
m = (x <= 0) & (x > -np.log(1 + 2.0))
z = np.exp(-x[m]) - 1
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(z, bg["dL"][m] / 1e3 / z, color=C[0], label="fiducial model ($h=0.67$)")
ax.errorbar(zsn, dlsn / zsn, yerr=err / zsn, fmt="o", ms=2.5, color=C[1], elinewidth=0.8, capsize=0,
            label="Betoule et al. (2014)")
dl_sn_model = np.interp(-np.log1p(zsn), x, bg["dL"]) / 1e3
chi2 = np.sum(((dlsn - dl_sn_model) / err) ** 2)
ax.set(xscale="log", xlabel="redshift $z$", ylabel=r"$d_L/z$ [Gpc]", ylim=(3.5, 8))
ax.text(0.03, 0.06, rf"fiducial: $\chi^2/N = {chi2:.1f}/{len(zsn)}$", transform=ax.transAxes, fontsize=7)
print(f"  SN chi2 = {chi2:.2f} for N = {len(zsn)}")
hs = np.linspace(0.60, 0.80, 2001)
chis = [np.sum(((dlsn - dl_sn_model * 0.67 / hh) / err) ** 2) for hh in hs]
with open(D + "sn_numbers.txt", "w") as fp:
    fp.write(f"chi2_fiducial {chi2:.3f}\nN {len(zsn)}\nh_best_fixed_Omegas {hs[np.argmin(chis)]:.4f}\nchi2_best {min(chis):.3f}\n")
print(open(D + "sn_numbers.txt").read())
h_best = hs[np.argmin(chis)]
ax.plot(z, bg["dL"][m] / 1e3 / z * 0.67 / h_best, color=C[2], ls="--", lw=1.0,
        label=rf"same $\Omega$'s, $h={h_best:.3f}$ ($\chi^2={min(chis):.1f}$)")
ax.legend(loc="upper left")
save(fig, "m1_supernovae")

# ===========================================================================
# Milestone II
# ===========================================================================
rcols = ["x", "Xe", "ne", "tau", "dtau", "ddtau", "g", "dg", "ddg", "Tb", "cs", "s", "Xe_saha"]
def rec(name):
    return dict(zip(rcols, np.loadtxt(D + name, unpack=True)))
r_fid, r_nr, r_sa, r_h = rec("recombination_fiducial.txt"), rec("recombination_noreion.txt"), \
    rec("recombination_saha.txt"), rec("recombination_Honly.txt")
xr = r_fid["x"]
x_dec = N2["peebles_decoupling_tau1_x"]
x_rec = N2["peebles_recombination_Xe0.1_x"]

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axes[0]
ax.semilogy(xr, r_fid["Xe"], color=C[0], label="Peebles + He + reionization (fiducial)")
ax.semilogy(xr, r_nr["Xe"], color=C[2], ls="--", label="Peebles + He, no reionization")
ax.semilogy(xr, r_h["Xe"], color=C[3], ls=":", lw=1.6, label="Peebles, H only")
ax.semilogy(xr, r_sa["Xe"], color=C[1], ls="-.", label="Saha (H + He)")
ax.axvline(x_dec, **EPOCH)
ax.set(xlabel=r"$x=\ln a$", ylabel=r"$X_e=n_e/n_H$", xlim=(-12, 0), ylim=(1e-4, 4))
ax.legend(loc="lower left", fontsize=6)
ax = axes[1]
for d, col, ls, lab in [(r_nr, C[2], "--", "Peebles + He"), (r_h, C[3], ":", "Peebles, H only"), (r_sa, C[1], "-.", "Saha (H + He)")]:
    ax.plot(d["x"], d["Xe"], color=col, ls=ls, lw=1.4 if ls == ":" else 1.2, label=lab)
fHe = N2["f_He"]
for v in (1.0, 1.0 + fHe, 1.0 + 2 * fHe):
    ax.axhline(v, **REF)
ax.text(-11.9, 1 + 2 * fHe + 0.01, r"$1+2f_{\rm He}$", fontsize=6.5)
ax.text(-11.9, 1 + fHe + 0.01, r"$1+f_{\rm He}$", fontsize=6.5)
ax.axvline(x_dec, **EPOCH)
ax.set(xlabel=r"$x=\ln a$", xlim=(-12, -6.5), ylim=(0, 1.25), title="helium recombination (linear scale)")
ax.legend(loc="lower left", fontsize=6)
save(fig, "m2_Xe")

fig, ax = plt.subplots(figsize=(W1, 2.5))
ax.semilogy(xr, r_fid["tau"], color=C[0], label=r"$\tau$")
ax.semilogy(xr, -r_fid["dtau"], color=C[1], label=r"$-\tau'$")
ax.semilogy(xr, np.abs(r_fid["ddtau"]), color=C[2], label=r"$|\tau''|$")
ax.axhline(1.0, **REF)
ax.axvline(x_dec, **EPOCH)
ax.set(xlabel=r"$x=\ln a$", ylabel="dimensionless", xlim=(-12, 0), ylim=(1e-8, 1e8))
ax.annotate("reionization", xy=(-2.1, 3e-1), xytext=(-5.5, 3e2), fontsize=6.5, color="#6b6a63",
            arrowprops=dict(arrowstyle="-", color="#9a9a94", lw=0.6))
ax.legend(loc="upper right")
save(fig, "m2_tau")

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1.4, 1]))
for ax, (lo, hi) in zip(axes, [(-7.6, -6.2), (-2.8, -1.4)]):
    mm = (xr > lo) & (xr < hi)
    s1 = np.max(np.abs(r_fid["g"][mm])) / np.max(np.abs(r_fid["dg"][mm]))
    s2 = np.max(np.abs(r_fid["g"][mm])) / np.max(np.abs(r_fid["ddg"][mm]))
    ax.plot(xr[mm], r_fid["g"][mm], color=C[0], label=r"$\tilde g$")
    ax.plot(xr[mm], r_fid["dg"][mm] * s1, color=C[1], label=rf"$\tilde g'\times{s1:.3g}$")
    ax.plot(xr[mm], r_fid["ddg"][mm] * s2, color=C[2], label=rf"$\tilde g''\times{s2:.3g}$")
    ax.axhline(0, color="#9a9a94", lw=0.6)
    ax.set(xlabel=r"$x=\ln a$")
    ax.legend(loc="upper left" if lo < -5 else "lower right")
axes[0].axvline(x_dec, **EPOCH)
axes[0].set(ylabel=r"$\tilde g$", title="recombination")
axes[1].set(title=r"reionization ($z_{\rm re}=8$)")
save(fig, "m2_g")

# ===========================================================================
# Milestone III
# ===========================================================================
pcols = ["x", "dc", "vc", "db", "vb", "Phi", "Psi", "dPhi", "Th0", "Th1", "Th2", "ST", "P0", "P2", "Pi", "SE",
         "N0", "N2", "P1", "N1"]
ks = [0.001, 0.01, 0.1, 0.3]
P = {k: dict(zip(pcols, np.loadtxt(D + f"perturbations_k{k:g}.txt", unpack=True))) for k in ks}
xp = P[ks[0]]["x"]
klab = {k: rf"$k={k:g}$/Mpc" for k in ks}

def style_legend(ax, entries, loc):
    """Second legend explaining line styles (solid/dashed) without colours."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#3d3d3a", ls=ls, lw=1.0) for ls, _ in entries]
    leg = ax.legend(handles, [lab for _, lab in entries], loc=loc, fontsize=6)
    ax.add_artist(leg)

def mark_entry(ax, k, col, y=None):
    xe = N3[f"k{k:g}_horizon_entry_x"]
    ax.axvline(xe, color=col, lw=0.6, ls=":")

def m3_epochs(ax):
    ax.axvline(x_rm, **EPOCH)
    ax.axvline(x_dec, **EPOCH)

# Densities
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
for i, k in enumerate(ks):
    d = P[k]
    axes[0].semilogy(xp, np.abs(d["dc"]), color=C[i], label=klab[k])
    axes[0].semilogy(xp, np.abs(d["db"]), color=C[i], ls="--", lw=0.9)
    axes[1].plot(xp, 4 * d["Th0"], color=C[i], label=klab[k])
    axes[1].plot(xp, 4 * d["N0"], color=C[i], ls="--", lw=0.9)
    mark_entry(axes[0], k, C[i])
for ax in axes:
    m3_epochs(ax)
    ax.set(xlabel=r"$x=\ln a$", xlim=(-18, 0))
axes[0].set(ylabel=r"$|\delta|$", title="matter")
axes[1].set(ylabel=r"$\delta$", ylim=(-3, 4), title="radiation")
style_legend(axes[0], [("-", r"$\delta_{\rm CDM}$"), ("--", r"$\delta_b$")], "center left")
style_legend(axes[1], [("-", r"$\delta_\gamma=4\Theta_0$"), ("--", r"$\delta_\nu=4\mathcal{N}_0$")], "lower left")
axes[0].legend(loc="upper left")
save(fig, "m3_delta")

# Velocities
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
for i, k in enumerate(ks):
    d = P[k]
    axes[0].semilogy(xp, np.abs(d["vc"]), color=C[i], label=klab[k])
    axes[0].semilogy(xp, np.abs(d["vb"]), color=C[i], ls="--", lw=0.9)
    axes[1].plot(xp, -3 * d["Th1"], color=C[i], label=klab[k])
    axes[1].plot(xp, -3 * d["N1"], color=C[i], ls="--", lw=0.9)
    mark_entry(axes[0], k, C[i])
for ax in axes:
    m3_epochs(ax)
    ax.set(xlabel=r"$x=\ln a$", xlim=(-18, 0))
axes[0].set(ylabel=r"$|v|$", title="matter")
axes[1].set(ylabel=r"$v$", ylim=(-2.2, 2.2), title="radiation")
style_legend(axes[0], [("-", r"$v_{\rm CDM}$"), ("--", r"$v_b$")], "lower right")
style_legend(axes[1], [("-", r"$v_\gamma=-3\Theta_1$"), ("--", r"$v_\nu=-3\mathcal{N}_1$")], "lower left")
axes[0].legend(loc="upper left")
save(fig, "m3_velocity")

# Quadrupoles
fig, ax = plt.subplots(figsize=(W1, 2.4))
for i, k in enumerate(ks):
    d = P[k]
    ax.plot(xp, d["Th2"], color=C[i], label=klab[k])
    ax.plot(xp, d["N2"], color=C[i], ls="--", lw=0.9)
m3_epochs(ax)
ax.set(xlabel=r"$x=\ln a$", ylabel="quadrupole", xlim=(-18, 0))
style_legend(ax, [("-", r"$\Theta_2$"), ("--", r"$\mathcal{N}_2$")], "upper left")
ax.legend(loc="lower left", fontsize=6)
save(fig, "m3_quadrupole")

# Potentials
fig, ax = plt.subplots(figsize=(W1, 2.4))
for i, k in enumerate(ks):
    d = P[k]
    ax.plot(xp, d["Phi"], color=C[i], label=klab[k])
    ax.plot(xp, d["Phi"] + d["Psi"], color=C[i], ls="--", lw=0.9)
    mark_entry(ax, k, C[i])
m3_epochs(ax)
ax.axvline(x_ml, **EPOCH)
ax.set(xlabel=r"$x=\ln a$", ylabel="potential", xlim=(-18, 0))
style_legend(ax, [("-", r"$\Phi$"), ("--", r"$\Phi+\Psi$")], "upper right")
ax.legend(loc="center left", fontsize=6)
save(fig, "m3_Phi")

# Polarization multipoles
fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), sharex=True)
for j, (key, lab) in enumerate([("P0", r"$\Theta^P_0$"), ("P1", r"$\Theta^P_1$"), ("P2", r"$\Theta^P_2$")]):
    for i, k in enumerate(ks):
        axes[j].plot(xp, P[k][key], color=C[i], label=klab[k])
    axes[j].axvline(x_dec, **EPOCH)
    axes[j].set(xlabel=r"$x=\ln a$", title=lab, xlim=(-9, 0))
axes[0].legend(loc="lower left", fontsize=6)
save(fig, "m3_polarization")

# Tests: (a) Theta_0 + Psi ~ cos(k r_s) in the radiation era, (b) Phi for a mode entering in RD,
# (c) growth rate in matter domination
eta_x = CubicSpline(x, bg["eta"])  # Mpc
sound = CubicSpline(r_fid["x"], r_fid["s"])  # comoving sound horizon in Mpc
Hp_x = CubicSpline(x, bg["Hp"] * 100.0 / c_kms)  # Hp/c [1/Mpc]
OmM = CubicSpline(x, bg["OmB"] + bg["OmCDM"])
fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3))
k = 0.3
d = P[k]
mm = xp < -7.0
# After horizon entry the decaying potential drives the oscillator, so the amplitude A is
# ~2x the initial value (and of opposite sign); the phase k r_s is the test
win = (xp > -11.0) & (xp < -9.5)
A = np.max(np.abs((d["Th0"] + d["Psi"])[win]))
axes[0].plot(xp[mm], (d["Th0"] + d["Psi"])[mm], color=C[3], label=r"$\Theta_0+\Psi$, $k=0.3$/Mpc")
axes[0].plot(xp[mm], -A * np.cos(k * sound(xp[mm])), color="#3d3d3a", ls=":", lw=1,
             label=rf"$-A\cos(k r_s)$, $A={A:.2f}$")
axes[0].set(xlabel=r"$x=\ln a$", xlim=(-14, -7), title="acoustic oscillations")
axes[0].legend(loc="lower left", fontsize=6)
for i, kk in [(2, 0.1), (3, 0.3)]:
    d = P[kk]
    y = kk * eta_x(xp) / np.sqrt(3)
    Phi_RD = d["Phi"][0] * 3 * (np.sin(y) - y * np.cos(y)) / y**3
    mm = xp < -9.0
    axes[1].plot(xp[mm], d["Phi"][mm], color=C[i], label=klab[kk])
    axes[1].plot(xp[mm], Phi_RD[mm], color=C[i], ls=":", lw=1.2)
axes[1].set(xlabel=r"$x=\ln a$", ylabel=r"$\Phi$", xlim=(-14, -9), title=r"$\Phi$ in RD (dotted: $3(\sin y-y\cos y)/y^3$)")
axes[1].legend(loc="lower left", fontsize=6)
for i, kk in [(2, 0.1), (3, 0.3)]:
    d = P[kk]
    f = CubicSpline(xp, np.log(np.abs(d["dc"])))(xp, 1)
    mm = xp > -6.5
    axes[2].plot(xp[mm], f[mm], color=C[i], label=rf"$d\ln\delta_{{\rm CDM}}/dx$, {klab[kk]}")
axes[2].plot(xp[xp > -6.5], OmM(xp[xp > -6.5]) ** 0.55, color="#3d3d3a", ls=":", lw=1.2, label=r"$\Omega_M(x)^{0.55}$")
axes[2].set(xlabel=r"$x=\ln a$", xlim=(-6.5, 0), ylim=(0.4, 1.4), title="growth rate")
axes[2].legend(loc="lower left", fontsize=6)
save(fig, "m3_tests")

# ===========================================================================
# Milestone IV
# ===========================================================================
eta0 = N1["eta0_Mpc"]
Tcols = open(D + "transfer_fiducial.txt").readline().split()[3:]
tr = np.loadtxt(D + "transfer_fiducial.txt")
kt = tr[:, 0]
ells_tr = [int(s.split("_")[1]) for s in Tcols]
fig, axes = plt.subplots(2, 1, figsize=(W1, 4.2), sharex=True)
for i, l in enumerate(ells_tr):
    T = tr[:, i + 1]
    axes[0].plot(kt * eta0, T, color=C[i], lw=0.8, label=rf"$\ell={l}$")
    axes[1].plot(kt * eta0, l * (l + 1) * T**2 / kt, color=C[i], lw=0.8, label=rf"$\ell={l}$")
axes[0].set(ylabel=r"$\Theta_\ell(k)$", xscale="log", xlim=(1, 3e3))
axes[1].set(xlabel=r"$k\eta_0$", ylabel=r"$\ell(\ell+1)|\Theta_\ell|^2/k$ [Mpc]", xscale="log", yscale="log",
            ylim=(1e-1, 1e3))
axes[0].legend(loc="lower left", fontsize=6, ncol=3)
save(fig, "m4_transfer")

cl = np.loadtxt(D + "cells_fiducial.txt")
ell, TT, EE, TE = cl[:, 0], cl[:, 1], cl[:, 2], cl[:, 3]
low = np.loadtxt(D + "../../external/planck_cell_low.txt")
hTT = np.loadtxt(D + "../../external/COM_PowerSpect_CMB-TT-binned_R3.01.txt")
hEE = np.loadtxt(D + "../../external/COM_PowerSpect_CMB-EE-binned_R3.02.txt")
hTE = np.loadtxt(D + "../../external/COM_PowerSpect_CMB-TE-binned_R3.02.txt")

fig, axes = plt.subplots(2, 1, figsize=(W1, 4.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
lz = np.loadtxt(D + "cells_lensed_fiducial.txt")  # ell, lensed TT, EE, TE, BB, [l(l+1)]^2 C_pp/2pi
ax = axes[0]
ax.plot(ell, TT, color=C[0], label="this work (unlensed)", zorder=3)
ax.plot(lz[:, 0], lz[:, 1], color="#3d3d3a", lw=0.8, ls="--", label="this work (lensed)", zorder=4)
ax.errorbar(low[:, 0], low[:, 1], yerr=[low[:, 3], low[:, 2]], fmt="o", ms=1.8, color=C[1], elinewidth=0.6,
            label=r"Planck 2018, low $\ell$")
ax.errorbar(hTT[:, 0], hTT[:, 1], yerr=[hTT[:, 2], hTT[:, 3]], fmt="o", ms=1.8, color=C[2], elinewidth=0.6,
            label=r"Planck 2018, binned high $\ell$")
ax.set(ylabel=r"$D_\ell^{TT}$ [$\mu$K$^2$]", xscale="log", xlim=(2, 2500))
ax.legend(loc="upper left")
ax = axes[1]
model_hi = np.interp(hTT[:, 0], ell, TT)
ax.errorbar(hTT[:, 0], hTT[:, 1] / model_hi - 1, yerr=hTT[:, 2] / model_hi, fmt="o", ms=1.8, color=C[2], elinewidth=0.6)
ax.plot(lz[:, 0], lz[:, 1] / np.interp(lz[:, 0], ell, TT) - 1, color="#3d3d3a", lw=0.9, ls="--",
        label="lensed/unlensed $-1$ (this work)")
ax.legend(loc="lower left", fontsize=6)
ax.axhline(0, color="#9a9a94", lw=0.8)
ax.set(xlabel=r"$\ell$", ylabel="data/model $-1$", ylim=(-0.1, 0.1))
axes[0].set_xlim(2, 2500)
save(fig, "m4_TT_planck")
res_hi = (hTT[:, 1] - model_hi) / hTT[:, 2]
print(f"  TT high-ell: chi2 = {np.sum(res_hi**2):.1f} for {len(res_hi)} bins")
mh = hTT[:, 0] <= 2500
chi_unl = np.sum(res_hi[mh] ** 2)
chi_len = np.sum(((hTT[:, 1] - np.interp(hTT[:, 0], lz[:, 0], lz[:, 1])) / hTT[:, 2])[mh] ** 2)
with open(D + "lensing_numbers.txt", "w") as fp:
    fp.write(f"planck_TT_chi2_unlensed {chi_unl:.1f}\nplanck_TT_chi2_lensed {chi_len:.1f}\nplanck_TT_nbins {np.sum(mh)}\n")
    i_pk = np.argmax(lz[:, 5])
    fp.write(f"pp_peak_ell {lz[i_pk, 0]:.0f}\npp_peak_value {lz[i_pk, 5]:.4e}\n")
    fp.write(f"BB_l1000 {lz[lz[:, 0] == 1000, 4][0]:.4f}\nBB_peak_ell {lz[np.argmax(lz[:, 4]), 0]:.0f}\n")
    fp.write(f"BB_peak {np.max(lz[:, 4]):.4f}\n")

# Source-term decomposition
fig, ax = plt.subplots(figsize=(W1, 2.6))
ax.plot(ell, TT, color="#3d3d3a", lw=1.5, label="total")
for i, name in enumerate(["SW", "ISW", "Doppler", "Quadrupole"]):
    d = np.loadtxt(D + f"cells_{name}.txt")
    ax.plot(d[:, 0], d[:, 1], color=C[i], lw=1.0,
            label={"SW": "Sachs-Wolfe", "ISW": "ISW", "Doppler": "Doppler", "Quadrupole": "quadrupole (pol.)"}[name])
ax.set(xscale="log", yscale="log", xlabel=r"$\ell$", ylabel=r"$D_\ell^{TT}$ [$\mu$K$^2$]", xlim=(2, 2500), ylim=(1, 1e4))
ax.legend(loc="lower left", fontsize=6)
save(fig, "m4_TT_components")

# Polarization
fig, axes = plt.subplots(2, 1, figsize=(W1, 4.0), sharex=True)
ax = axes[0]
ax.plot(ell, EE, color=C[1], label="this work")
ax.errorbar(hEE[:, 0], hEE[:, 1], yerr=[hEE[:, 2], hEE[:, 3]], fmt="o", ms=1.8, color=C[2], elinewidth=0.6,
            label="Planck 2018, binned")
ax.set(ylabel=r"$D_\ell^{EE}$ [$\mu$K$^2$]", yscale="log", ylim=(1e-3, 80))
ax.legend(loc="lower right")
ax = axes[1]
ax.plot(ell, TE, color=C[0], label="this work")
ax.errorbar(hTE[:, 0], hTE[:, 1], yerr=[hTE[:, 2], hTE[:, 3]], fmt="o", ms=1.8, color=C[2], elinewidth=0.6,
            label="Planck 2018, binned")
ax.axhline(0, color="#9a9a94", lw=0.6)
ax.set(xlabel=r"$\ell$", ylabel=r"$D_\ell^{TE}$ [$\mu$K$^2$]", xlim=(2, 2500))
ax.legend(loc="lower right")
save(fig, "m4_EE_TE")

# Matter power spectrum
pk = np.loadtxt(D + "matter_power_fiducial.txt")
reid = np.loadtxt(D + "../../external/reid_DR7.txt")
wa = np.loadtxt(D + "../../external/wmap_act.txt")
fig, ax = plt.subplots(figsize=(W1, 2.6))
ax.loglog(pk[:, 0], pk[:, 1], color=C[0], label="this work, $z=0$")
ax.errorbar(reid[:, 0], reid[:, 1], yerr=reid[:, 2], fmt="o", ms=1.8, color=C[1], elinewidth=0.6, label="SDSS DR7 LRG")
ax.errorbar(wa[:, 0], wa[:, 1], yerr=wa[:, 2] - wa[:, 1], fmt="s", ms=2, color=C[2], elinewidth=0.6,
            label="CMB (WMAP+ACT)")
ax.axvline(N1["k_eq_hMpc"], color="#3d3d3a", ls=":", lw=1)
ax.text(N1["k_eq_hMpc"] * 0.92, 3.5e4, r"$k_{\rm eq}$", fontsize=7, ha="right")
ax.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k)$ [(Mpc/$h$)$^3$]", ylim=(1e2, 5e4))
ax.legend(loc="lower right", fontsize=6)
save(fig, "m4_Pk")

# Toy cosmology (course reference)
ct = np.loadtxt(D + "cells_toy.txt")
pt = np.loadtxt(D + "matter_power_toy.txt")
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
axes[0].loglog(ct[:, 0], ct[:, 1], color=C[0])
axes[0].set(xlabel=r"$\ell$", ylabel=r"$D_\ell^{TT}$ [$\mu$K$^2$]", xlim=(2, 2000), title="toy cosmology")
axes[1].loglog(pt[:, 0], pt[:, 1], color=C[0])
axes[1].set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k)$ [(Mpc/$h$)$^3$]", title="toy cosmology")
save(fig, "m4_toy")

# ===========================================================================
# Validation against CLASS (all physics matched: same parameters, tanh reionization, no lensing)
# ===========================================================================
if "--no-class" not in sys.argv:
    from classy import Class
    sys.path.insert(0, "python")
    from params import class_params, read_params

    p = read_params(D + "params.txt")
    h = p["h"]
    cp = class_params(p)
    cp.update({"output": "tCl,pCl,mPk,mTk", "lensing": "no", "l_max_scalars": 2100, "P_k_max_1/Mpc": 1.0,
               "gauge": "newtonian", "k_output_values": "0.001,0.01,0.1"})
    cls = Class()
    cls.set(cp)
    cls.compute()
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.2))
    # Background
    b = cls.get_background()
    zb = b["z"][::-1]
    zz = np.logspace(-2, 6, 200)
    Hs = CubicSpline(x, bg["H"] * 100.0)
    ts = CubicSpline(x, bg["t"])
    es = CubicSpline(x, bg["eta"])
    ax = axes[0, 0]
    ax.semilogx(zz, Hs(-np.log1p(zz)) / (np.array([cls.Hubble(z) for z in zz]) * c_kms) - 1, color=C[0], label=r"$H$")
    ax.semilogx(zz, ts(-np.log1p(zz)) / np.interp(zz, zb, b["proper time [Gyr]"][::-1]) - 1, color=C[1], label=r"$t$")
    ax.semilogx(zz, es(-np.log1p(zz)) / np.interp(zz, zb, b["conf. time [Mpc]"][::-1]) - 1, color=C[2], label=r"$\eta$")
    ax.set(xlabel="$z$", ylabel="ours/CLASS $-1$", title="background", ylim=(-1e-5, 1e-5))
    ax.legend(loc="upper right")
    # X_e
    th = cls.get_thermodynamics()
    o = np.argsort(th["z"])
    Xc = CubicSpline(th["z"][o], th["x_e"][o])
    zr = np.logspace(0, 3.9, 400)
    ax = axes[0, 1]
    ax.semilogx(zr, np.interp(-np.log1p(zr), r_fid["x"], r_fid["Xe"]) / Xc(zr) - 1, color=C[0])
    ax.axhline(0, color="#9a9a94", lw=0.6)
    ax.set(xlabel="$z$", ylabel="ours/CLASS $-1$", title=r"$X_e$ (CLASS: HyRec)", ylim=(-0.08, 0.2))
    # C_ell
    raw = cls.raw_cl(2000)
    lc = raw["ell"][2:]
    f = lc * (lc + 1) / (2 * np.pi) * (p["TCMB"] * 1e6) ** 2
    ax = axes[1, 0]
    ax.semilogx(lc, TT[: len(lc)] / (f * raw["tt"][2:]) - 1, color=C[0], label="TT")
    ax.semilogx(lc, EE[: len(lc)] / (f * raw["ee"][2:]) - 1, color=C[1], label="EE")
    teamp = np.max(np.abs(f * raw["te"][2:]))
    ax.semilogx(lc, (TE[: len(lc)] - f * raw["te"][2:]) / teamp, color=C[2], label=r"TE (diff./max$|$TE$|$)")
    ax.axhline(0, color="#9a9a94", lw=0.6)
    ax.set(xlabel=r"$\ell$", ylabel="ours/CLASS $-1$", title=r"$C_\ell$", ylim=(-0.06, 0.08), xlim=(2, 2000))
    ax.legend(loc="upper left", ncol=3, fontsize=6)
    # P(k)
    ax = axes[1, 1]
    pc = np.array([cls.pk(kk * h, 0.0) * h**3 for kk in pk[:, 0]])
    ax.semilogx(pk[:, 0], pk[:, 1] / pc - 1, color=C[0])
    ax.axhline(0, color="#9a9a94", lw=0.6)
    ax.set(xlabel=r"$k$ [$h$/Mpc]", ylabel="ours/CLASS $-1$", title=r"$P(k)$, $z=0$", ylim=(-0.01, 0.01))
    fig.tight_layout()
    save(fig, "validation_class")
    # Numbers for the text
    TTr = TT[: len(lc)] / (f * raw["tt"][2:]) - 1
    EEr = EE[: len(lc)] / (f * raw["ee"][2:]) - 1
    TEr = (TE[: len(lc)] - f * raw["te"][2:]) / teamp
    with open(D + "validation_numbers.txt", "w") as fp:
        fp.write(f"TT_maxabs {np.max(np.abs(TTr)):.4e}\nTT_maxabs_l1200 {np.max(np.abs(TTr[lc<=1200])):.4e}\n")
        fp.write(f"EE_maxabs_l100 {np.max(np.abs(EEr[lc>=100])):.4e}\nEE_l2to8_maxabs {np.max(np.abs(EEr[lc<=8])):.4e}\n")
        fp.write(f"TE_maxabs {np.max(np.abs(TEr)):.4e}\nPk_maxabs {np.max(np.abs(pk[:,1]/pc-1)):.4e}\n")
        der = cls.get_current_derived_parameters(["tau_reio", "z_star", "rs_star", "age", "conformal_age", "z_d", "rs_d"])
        for kname, val in der.items():
            fp.write(f"class_{kname} {val:.6f}\n")
        # Lensing: CLASS with lensing on (same physics), compared with our lensed spectra
        cpl = dict(cp)
        cpl.update({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": 2500})
        for key in ("k_output_values", "gauge", "P_k_max_1/Mpc"): cpl.pop(key, None)
        cl2 = Class(); cl2.set(cpl); cl2.compute()
        lensed, unlensed = cl2.lensed_cl(2500), cl2.raw_cl(2500)
        le = lz[:, 0].astype(int)
        fL = le * (le + 1) / (2 * np.pi) * (p["TCMB"] * 1e6) ** 2
        ratio_c = lensed["tt"][le] / unlensed["tt"][le]
        ratio_o = lz[:, 1] / np.interp(lz[:, 0], ell, TT)
        pp_c = (le * (le + 1.0)) ** 2 * unlensed["pp"][le] / (2 * np.pi)
        bb_c = fL * lensed["bb"][le]
        for lo, hi, tag in [(2, 1000, "l1000"), (1000, 2000, "l2000"), (2000, 2500, "l2500")]:
            s_ = (le >= lo) & (le <= hi)
            fp.write(f"lens_ratio_TT_maxdiff_{tag} {np.max(np.abs(ratio_o[s_] - ratio_c[s_])):.3e}\n")
            fp.write(f"pp_maxrel_{tag} {np.max(np.abs(lz[s_, 5] / pp_c[s_] - 1)):.3e}\n")
            fp.write(f"BB_maxrel_{tag} {np.max(np.abs(lz[s_, 4] / bb_c[s_] - 1)):.3e}\n")
        np.savetxt(D + "class_lensing.txt", np.column_stack([le, ratio_c, pp_c, bb_c]),
                   header="ell CLASS lensed/unlensed TT, [l(l+1)]^2 C_pp/2pi, lensed D_BB [muK^2]")
    print(open(D + "validation_numbers.txt").read())

    # Lensing figure: lensing potential and lensed BB, ours vs CLASS
    cll = np.loadtxt(D + "class_lensing.txt")
    fig, axes = plt.subplots(2, 1, figsize=(W1, 4.0), sharex=True)
    ax = axes[0]
    ax.loglog(lz[:, 0], lz[:, 5], color=C[0], label="this work")
    ax.loglog(cll[:, 0], cll[:, 2], color="#3d3d3a", ls="--", lw=0.9, label="CLASS")
    ax.set(ylabel=r"$[\ell(\ell+1)]^2C_\ell^{\phi\phi}/2\pi$")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.loglog(lz[:, 0], lz[:, 4], color=C[1], label="this work")
    ax.loglog(cll[:, 0], cll[:, 3], color="#3d3d3a", ls="--", lw=0.9, label="CLASS")
    ax.set(xlabel=r"$\ell$", ylabel=r"lensed $D_\ell^{BB}$ [$\mu$K$^2$]", xlim=(2, 2500))
    ax.legend(loc="lower right")
    save(fig, "m4_lensing")
