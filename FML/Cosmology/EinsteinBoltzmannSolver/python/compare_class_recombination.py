#!/usr/bin/env python3
"""Compare data/recombination.txt with CLASS (pip install classy).

CLASS is run without reionization (run ours with reionization=false). Both x_e = n_e/n_H include the helium
electrons. With our default RECFAST model compare with recombination=RECFAST (same physics) or HyRec.

Usage:  python3 python/compare_class_recombination.py [data/recombination.txt] [RECFAST|HyRec]
"""
import os
import sys

import numpy as np
from classy import Class
from scipy.interpolate import CubicSpline

from params import class_params, read_params

infile = sys.argv[1] if len(sys.argv) > 1 else "data/recombination.txt"
rec = sys.argv[2] if len(sys.argv) > 2 else "HyRec"
PARAMS = class_params(read_params(os.path.join(os.path.dirname(infile), "params.txt")), recombination=rec)


d = np.loadtxt(infile)
x = d[:, 0]
ours = {name: CubicSpline(x, d[:, i]) for i, name in
        [(1, "Xe"), (3, "tau"), (6, "g"), (9, "Tb"), (11, "s")]}

cl = Class()
cl.set(PARAMS)
cl.compute()
th = cl.get_thermodynamics()
z_c = th["z"]
order = np.argsort(z_c)
xe_c = CubicSpline(z_c[order], th["x_e"][order])
Tb_c = CubicSpline(z_c[order], th["Tb [K]"][order])
der = cl.get_current_derived_parameters(["z_star", "rs_star"])

# Our tau = 1 redshift and sound horizon there
from scipy.optimize import brentq
x_star = brentq(lambda xx: ours["tau"](xx) - 1.0, -8, -6)
print(f"CLASS ({rec}, no reionization) vs ours")
print(f"  z_*  (tau = 1)     CLASS {der['z_star']:.2f}   ours {np.expm1(-x_star):.2f}   "
      f"rel {np.expm1(-x_star) / der['z_star'] - 1:+.1e}")
print(f"  r_s(z_*) [Mpc]     CLASS {der['rs_star']:.3f}  ours {ours['s'](x_star):.3f}  "
      f"rel {ours['s'](x_star) / der['rs_star'] - 1:+.1e}")
print(f"\n{'z':>6} {'Xe CLASS':>11} {'Xe ours':>11} {'rel':>8}   {'Tb CLASS':>9} {'Tb ours':>9} {'rel':>8}")
for z in [7000, 6000, 5000, 4000, 3000, 2500, 2000, 1800, 1600, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 600, 400, 200, 100, 50, 20, 0]:
    xx = -np.log1p(z)
    a, b = xe_c(z), ours["Xe"](xx)
    ta, tb = Tb_c(z), ours["Tb"](xx)
    print(f"{z:6d} {a:11.4e} {b:11.4e} {b / a - 1:+9.1e}   {ta:9.3f} {tb:9.3f} {tb / ta - 1:+9.1e}")
