#!/usr/bin/env python3
"""Compare early dark energy (model = ede) with CLASS_EDE (Hill et al. 2020, github.com/mwt5345/class_ede): the
effect of EDE relative to LCDM for C_ell^TT and P(k) at z = 0, and the potential parameters found by the shooting.

Usage:  python3 python/compare_class_ede.py LCDMDIR EDEDIR /path/to/class_ede/class
e.g.
    ./cmb transfer_ells= output_dir=data/lcdm
    ./cmb transfer_ells= model=ede ede_fEDE=0.1 ede_log10zc=3.5 ede_theta_i=2.8 ede_n=3 output_dir=data/ede
Both codes define f_EDE as the maximum of rho_EDE/rho_tot and z_c as its redshift (CLASS_EDE takes the
maximum on its background table, we interpolate), and keep Lambda for the closure.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

if len(sys.argv) != 4:
    sys.exit(__doc__)
dl, de, exe = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
pl, pe = read_params(os.path.join(dl, "params.txt")), read_params(os.path.join(de, "params.txt"))
if pe.get("model") != "ede":
    sys.exit(f"{de} is not an ede run")
cl_ours = {d: np.loadtxt(os.path.join(d, "cells.txt")) for d in (dl, de)}
lmax = int(min(cl_ours[dl][-1, 0], cl_ours[de][-1, 0]))


def run(extra, tag):
    cp = class_params(pl)
    cp.update({"output": "tCl,mPk", "l_max_scalars": lmax + 100, "P_k_max_1/Mpc": 1.0})
    cp.update(extra)
    tmp = tempfile.mkdtemp(prefix="classede_")
    cp.update({"root": os.path.join(tmp, tag), "overwrite_root": "yes", "background_verbose": 2})
    ini = os.path.join(tmp, tag + ".ini")
    with open(ini, "w") as f:
        for k, v in cp.items():
            f.write(f"{k} = {v}\n")
    out = subprocess.run([exe, ini], cwd=os.path.dirname(exe), check=True, capture_output=True, text=True).stdout
    return np.loadtxt(os.path.join(tmp, tag + "_cl.dat")), np.loadtxt(os.path.join(tmp, tag + "_pk.dat")), out


ede = {"Omega_Lambda": 0, "Omega_fld": 0, "Omega_scf": -1, "scf_parameters": "1, 1, 1, 1, 1, 0.0",
       "scf_tuning_index": 3, "attractor_ic_scf": "no", "CC_scf": 1,
       "fEDE": pe.get("ede_fEDE", 0.1), "log10z_c": pe.get("ede_log10zc", 3.5),
       "thetai_scf": pe.get("ede_theta_i", 2.8), "n_scf": pe.get("ede_n", 3.0)}
print("CLASS_EDE:", ede)
Lc, Lp, _ = run({}, "lcdm")
Ec, Ep, out = run(ede, "ede")
for line in out.splitlines():
    if re.search(r"f_EDE|log10z_c|f_scf|m_scf", line):
        print("  CLASS_EDE:", line.strip())
for line in open(os.path.join(de, "..", os.path.basename(de) + ".log")) if os.path.exists(
        os.path.join(de, "..", os.path.basename(de) + ".log")) else []:
    if "early dark energy" in line:
        print("  ours:     ", line.strip())
ell = np.arange(2, lmax + 1)
rc = np.interp(ell, Ec[:, 0], Ec[:, 1]) / np.interp(ell, Lc[:, 0], Lc[:, 1])
a, b = cl_ours[dl], cl_ours[de]
ro = np.interp(ell, b[:, 0], b[:, 1]) / np.interp(ell, a[:, 0], a[:, 1])
print("C_ell^TT(EDE)/C_ell^TT(LCDM):   ell   ours   CLASS_EDE   difference")
for l in [2, 10, 30, 100, 200, 300, 500, 800, 1000, 1500, 2000]:
    if l <= lmax:
        print(f"  {l:5d}  {ro[l - 2]:.5f}  {rc[l - 2]:.5f}  {ro[l - 2] / rc[l - 2] - 1:+.1e}")
pa, pb = np.loadtxt(os.path.join(dl, "matter_power.txt")), np.loadtxt(os.path.join(de, "matter_power.txt"))
print("P(EDE)/P(LCDM) at z = 0:   k [h/Mpc]   ours   CLASS_EDE   difference")
for k in [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    o = np.interp(k, pa[:, 0], pb[:, 1] / pa[:, 1])
    c = np.interp(k, Ep[:, 0], Ep[:, 1]) / np.interp(k, Lp[:, 0], Lp[:, 1])
    print(f"  {k:8.4f}  {o:.5f}  {c:.5f}  {o / c - 1:+.1e}")
