#!/usr/bin/env python3
"""Compare perturbations_k*.txt in a directory (written by ./cmb) with CLASS.

Usage:  python3 python/compare_class_perturbations.py DIR
DIR must contain params.txt and perturbations_k*.txt. The exact k of each file is read from its
header. CLASS runs in Newtonian gauge with the same parameters (incl. Omega_k), and the
conventions are mapped as
  delta_ours = -delta_CLASS,  Phi_ours = phi_CLASS,  Psi_ours = -psi_CLASS.
Printed: max |ours - CLASS| / max|CLASS| over x in [-14, 0] (skipping the first e-fold of each CLASS mode) for Phi, Psi,
delta_cdm, delta_b (deltas relative to the local value, for x > -6),
and the relative difference today. Photon multipoles are not compared because CLASS switches
to approximations (tight coupling, radiation streaming) at different times.
"""
import glob
import os
import re
import sys

import numpy as np
from classy import Class
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "perturbations_k*.txt")),
               key=lambda f: float(re.findall(r"k([\d.e-]+)\.txt", f)[0]))
ks = [float(re.findall(r"# k = ([\d.eE+-]+)", open(f).readline())[0]) for f in files]

cp = class_params(read_params(os.path.join(d, "params.txt")))
cp.update({"output": "mTk", "gauge": "newtonian", "k_output_values": ",".join(f"{k:.10g}" for k in ks)})
cl = Class()
cl.set(cp)
cl.compute()
pts = cl.get_perturbations()["scalar"]

cols = ["x", "dc", "vc", "db", "vb", "Phi", "Psi"]
print(f"Omega_k = {cp['Omega_k']}")
print(f"{'k [1/Mpc]':>12} {'quantity':>10} {'max|diff|/max':>14} {'today rel':>11}")
worst = 0.0
for f, k, pk in zip(files, ks, pts):
    o = dict(zip(cols, np.loadtxt(f, unpack=True, usecols=range(7))))
    xc = np.log(pk["a"])
    # CLASS starts each mode late (x ~ -8.5 for k = 0.001/Mpc) with approximate initial conditions;
    # skip its initial transient
    m = (o["x"] > max(-14.0, xc[0] + 1.0)) & (o["x"] <= 0)
    for name, ours, theirs in [("Phi", o["Phi"], pk["phi"]), ("Psi", o["Psi"], -pk["psi"]),
                               ("delta_cdm", o["dc"], -pk["delta_cdm"]), ("delta_b", o["db"], -pk["delta_b"])]:
        th = interp1d(xc, theirs, bounds_error=False, fill_value=np.nan)(o["x"][m])
        ok = np.isfinite(th)
        if name.startswith("delta"):
            # relative to the local amplitude (deltas grow by orders of magnitude)
            diff = np.max(np.abs(ours[m][ok] / th[ok] - 1.0)[o["x"][m][ok] > -6])
        else:
            diff = np.max(np.abs(ours[m][ok] - th[ok])) / np.max(np.abs(th[ok]))
        today = ours[m][ok][-1] / th[ok][-1] - 1.0
        worst = max(worst, diff)
        print(f"{k:12.6g} {name:>10} {diff:14.2e} {today:+11.2e}")
print(f"worst: {worst:.2e}  (deltas: max relative difference for x > -6)")
