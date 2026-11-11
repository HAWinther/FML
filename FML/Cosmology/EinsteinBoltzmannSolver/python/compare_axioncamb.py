#!/usr/bin/env python3
"""Compare mixed axion dark matter (model = axion) with axionCAMB (Hlozek et al. 2015, github.com/dgrin1/axionCAMB):
the effect of the axion relative to LCDM on the cold (CDM + baryon) power spectrum and C_ell^TT, and the axion
transfer function delta_ax / delta_cdm (synchronous gauge), at z = 0.

Usage:  python3 python/compare_axioncamb.py LCDMDIR AXDIR /path/to/axionCAMB/camb [template.ini]
e.g.
    ./cmb transfer_ells= output_dir=data/lcdm
    ./cmb params/axion.ini transfer_ells= ax_switch_mH=3 output_dir=data/axion
The axionCAMB input is built from AXDIR/params.txt on the template (default: params.ini next to the executable);
its LCDM reference is a run with axfrac = 1e-6. axionCAMB replaces the field by its fluid where m = 3H: run ours
with ax_switch_mH = 3 to compare like with like (with the default switch, m / H ~ 125, they differ by up to a few
percent near the axion Jeans scale: the m = 3H approximation). Two things to know about axionCAMB (2019 master):
without massive neutrinos its axion background has no massless neutrinos (set P%Nu_massless_degeneracy =
P%Num_Nu_massless before w_evolve in inidriver_axion.F90), which makes a_osc 12% early; and its phiinit is h times
the initial field (its density table is right), so its axion isocurvature amp_i is too large by 1/h^2.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from params import read_params  # noqa: E402

if len(sys.argv) not in (4, 5):
    sys.exit(__doc__)
dl, da, exe = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
template = sys.argv[4] if len(sys.argv) == 5 else os.path.join(os.path.dirname(exe), "params.ini")
p = read_params(os.path.join(da, "params.txt"))
if p.get("model") != "axion":
    sys.exit(f"{da} is not an axion run")
if p.get("m_nu", 0.0) > 0 or p.get("OmegaK", 0.0) != 0:
    sys.exit("the comparison is set up for massless neutrinos and a flat universe")
h = p["h"]


def run(axfrac, tag):
    keys = {"output_root": None, "get_scalar_cls": "T", "get_vector_cls": "F", "get_tensor_cls": "F",
            "get_transfer": "T", "do_lensing": "F", "do_nonlinear": "0", "l_max_scalar": "2500",
            "ombh2": p["OmegaB"] * h**2, "omnuh2": 0, "omk": 0, "hubble": 100 * h, "use_axfrac": "T",
            "omdah2": p["OmegaCDM"] * h**2, "axfrac": axfrac, "m_ax": 10 ** p["ax_log10_m"],
            "axion_isocurvature": "F", "temp_cmb": p["TCMB"], "helium_fraction": p["Yp"],
            "massless_neutrinos": p["Neff"], "massive_neutrinos": 0, "scalar_amp(1)": p["A_s"],
            "scalar_spectral_index(1)": p["n_s"], "pivot_scalar": p["k_pivot"], "re_use_optical_depth": "F",
            "re_redshift": p["z_reion"], "re_delta_redshift": p["delta_z_reion"], "transfer_kmax": 5,
            "transfer_num_redshifts": 1, "transfer_redshift(1)": 0, "accurate_reionization": "T"}
    tmp = tempfile.mkdtemp(prefix="axioncamb_")
    keys["output_root"] = os.path.join(tmp, tag)
    keys["transfer_filename(1)"] = "transfer_z0.dat"
    keys["transfer_matterpower(1)"] = "matterpower_z0.dat"
    lines, seen = [], set()
    for line in open(template):
        m = re.match(r"\s*([^#=\s][^=]*?)\s*=", line)
        if m and m.group(1) in keys:
            line = f"{m.group(1)} = {keys[m.group(1)]}\n"
            seen.add(m.group(1))
        lines.append(line)
    lines += [f"{k} = {v}\n" for k, v in keys.items() if k not in seen]
    ini = os.path.join(tmp, tag + ".ini")
    open(ini, "w").writelines(lines)
    subprocess.run([exe, ini], cwd=os.path.dirname(exe), check=True, stdout=subprocess.DEVNULL)
    root = keys["output_root"]
    return np.loadtxt(root + "_transfer_z0.dat"), np.loadtxt(root + "_scalCls.dat")


def cold(T):  # axionCAMB transfer columns: k/h, cdm, b, g, r, nu, axion, growth, tot
    wc, wb = (1 - float(p["ax_fraction"])) * p["OmegaCDM"], p["OmegaB"]
    return (wc * T[:, 1] + wb * T[:, 2]) / (wc + wb)


Tl, cll = run(1e-6, "lcdm")
Ta, cla = run(p["ax_fraction"], "axion")
k = Ta[:, 0]
R_camb = (cold(Ta) / cold(Tl)) ** 2   # P_cb / P_cb,LCDM (the same primordial spectrum)
ml, ma = np.loadtxt(os.path.join(dl, "matter_power.txt")), np.loadtxt(os.path.join(da, "matter_power.txt"))
R_ours = np.interp(np.log(k), np.log(ma[:, 0]), ma[:, 2] / np.interp(ma[:, 0], ml[:, 0], ml[:, 2]))
ok = (k > ma[0, 0]) & (k < ma[-1, 0])
print(f"axion: m = 10^{p['ax_log10_m']} eV, f = {p['ax_fraction']}, switch m/H = {p.get('ax_switch_mH', 0)} "
      f"(0: fast oscillation)")
print("P_cb / P_cb(LCDM) at z = 0:")
print("   k [h/Mpc]     ours    axionCAMB   ours/axionCAMB - 1")
for kk in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
    i = np.argmin(abs(k - kk))
    if ok[i]:
        print(f"   {k[i]:8.3f}  {R_ours[i]:9.5f}  {R_camb[i]:9.5f}   {R_ours[i] / R_camb[i] - 1:+.2e}")
print(f"   max |ours/axionCAMB - 1| = {np.max(abs(R_ours[ok] / R_camb[ok] - 1)):.2e}")
if os.path.exists(os.path.join(da, "matter_power.txt")) and ma.shape[1] > 3:
    Pax = ma[:, -1] / ma[:, 2]   # P_axion / P_cb
    r_camb = (Ta[:, 6] / cold(Ta)) ** 2
    r_ours = np.interp(np.log(k), np.log(ma[:, 0]), Pax)
    print(f"P_axion / P_cb: max |ours/axionCAMB - 1| = {np.max(abs(r_ours[ok] / r_camb[ok] - 1)):.2e} "
          f"(min ratio {r_camb[ok].min():.3g})")
co = {d: np.loadtxt(os.path.join(d, "cells.txt")) for d in (dl, da)}
lmax = int(min(co[dl][-1, 0], co[da][-1, 0], cla[-1, 0]))
ell = np.arange(2, lmax + 1)
ours = np.interp(ell, co[da][:, 0], co[da][:, 1]) / np.interp(ell, co[dl][:, 0], co[dl][:, 1])
camb = np.interp(ell, cla[:, 0], cla[:, 1]) / np.interp(ell, cll[:, 0], cll[:, 1])
print("C_ell^TT / C_ell^TT(LCDM):")
for l in (2, 10, 30, 100, 200, 500, 1000, 2000):
    if l <= lmax:
        print(f"   ell = {l:5d}: ours {ours[l - 2]:.5f}  axionCAMB {camb[l - 2]:.5f}  diff {ours[l - 2] - camb[l - 2]:+.1e}")
print(f"   max |difference of the ratios| = {np.max(abs(ours - camb)):.2e}")
