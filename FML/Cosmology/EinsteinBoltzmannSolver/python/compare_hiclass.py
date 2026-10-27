#!/usr/bin/env python3
"""Compare the Horndeski alpha model (model = horndeski) with hi_class: the effect of the alphas (and of the
expansion) relative to LCDM, for C_ell^TT and P(k) at z = 0, in both codes.

Usage:  python3 python/compare_hiclass.py LCDMDIR HORNDESKIDIR /path/to/hi_class/class
e.g.
    ./cmb transfer_ells= output_dir=data/lcdm
    ./cmb transfer_ells= model=horndeski hd_cK=1 hd_cB=0.6 hd_cM=0.3 output_dir=data/hd
hi_class (https://github.com/miguelzuma/hi_class_public) is run with the matching gravity_model (propto_omega,
propto_scale or constant_alphas), parameters_smg = c_K, c_B, c_M, 0, M*^2_ini and expansion_model = lcdm or
wowa. Both codes use the Bellini-Sawicki alphas. hi_class starts the scalar on its attractor; we start it at
zero (the difference has decayed by recombination).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

if len(sys.argv) != 4:
    sys.exit(__doc__)
dl, dh, exe = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
pl, ph = read_params(os.path.join(dl, "params.txt")), read_params(os.path.join(dh, "params.txt"))
if ph.get("model") != "horndeski":
    sys.exit(f"{dh} is not a horndeski run")
cl_ours = {d: np.loadtxt(os.path.join(d, "cells.txt")) for d in (dl, dh)}
lmax = int(min(cl_ours[dl][-1, 0], cl_ours[dh][-1, 0]))


def get(name, default):
    return ph.get(name, default)


def run_hiclass(extra, tag):
    cp = class_params(pl)
    cp.update({"output": "tCl,mPk", "l_max_scalars": lmax + 100, "P_k_max_1/Mpc": 1.0})
    cp.update(extra)
    tmp = tempfile.mkdtemp(prefix="hiclass_")
    cp.update({"root": os.path.join(tmp, tag), "overwrite_root": "yes"})
    ini = os.path.join(tmp, tag + ".ini")
    with open(ini, "w") as f:
        for k, v in cp.items():
            f.write(f"{k} = {v}\n")
    subprocess.run([exe, ini], cwd=os.path.dirname(exe), check=True, stdout=subprocess.DEVNULL)
    cl = np.loadtxt(os.path.join(tmp, tag + "_cl.dat"))
    pk = np.loadtxt(os.path.join(tmp, tag + "_pk.dat"))
    return cl, pk


models = {0: "propto_omega", 1: "propto_scale", 2: "constant_alphas"}
w0, wa = get("hd_w0", -1.0), get("hd_wa", 0.0)
smg = {"Omega_Lambda": 0, "Omega_fld": 0, "Omega_smg": -1,
       "gravity_model": models[int(get("hd_parametrization", 0))],
       "parameters_smg": f"{get('hd_cK', 1.0)}, {get('hd_cB', 0.0)}, {get('hd_cM', 0.0)}, 0., {get('hd_M2_ini', 1.0)}",
       "expansion_model": "lcdm" if (w0 == -1.0 and wa == 0.0) else "wowa",
       "expansion_smg": "0.5" if (w0 == -1.0 and wa == 0.0) else f"0.5, {w0}, {wa}"}
print("hi_class:", smg)
Lc, Lp = run_hiclass({}, "lcdm")
Hc, Hp = run_hiclass(smg, "smg")
ell = np.arange(2, lmax + 1)
rc = np.interp(ell, Hc[:, 0], Hc[:, 1]) / np.interp(ell, Lc[:, 0], Lc[:, 1])
a, b = cl_ours[dl], cl_ours[dh]
ro = np.interp(ell, b[:, 0], b[:, 1]) / np.interp(ell, a[:, 0], a[:, 1])
print("C_ell^TT(Horndeski)/C_ell^TT(LCDM):   ell   ours   hi_class   difference")
for l in [2, 5, 10, 20, 30, 50, 100, 200, 400, 600, 800, 1000]:
    if l <= lmax:
        print(f"  {l:5d}  {ro[l - 2]:.5f}  {rc[l - 2]:.5f}  {ro[l - 2] / rc[l - 2] - 1:+.1e}")
pa, pb = np.loadtxt(os.path.join(dl, "matter_power.txt")), np.loadtxt(os.path.join(dh, "matter_power.txt"))
print("P(Horndeski)/P(LCDM) at z = 0:   k [h/Mpc]   ours   hi_class   difference")
for k in [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    o = np.interp(k, pa[:, 0], pb[:, 1] / pa[:, 1])
    c = np.interp(k, Hp[:, 0], Hp[:, 1]) / np.interp(k, Lp[:, 0], Lp[:, 1])
    print(f"  {k:8.4f}  {o:.5f}  {c:.5f}  {o / c - 1:+.1e}")
