#!/usr/bin/env python3
"""Compare data/background.txt with CLASS (pip install classy) for the fiducial parameters.

Usage:  python3 python/compare_class_background.py [data/background.txt] [data/params.txt]
Parameters are read from data/params.txt (written by main) unless another file is given.
"""
import sys

import numpy as np
from classy import Class
from scipy.interpolate import CubicSpline

from params import class_params, read_params

PARAMS = class_params(read_params(sys.argv[2] if len(sys.argv) > 2 else "data/params.txt"))


infile = sys.argv[1] if len(sys.argv) > 1 else "data/background.txt"
d = np.loadtxt(infile)
x, eta, t, H, dL = d[:, 0], d[:, 1], d[:, 2], d[:, 3] * 100.0, d[:, 15]

def ours(arr, z):
    """Cubic-spline interpolation of the output table in x."""
    return CubicSpline(x, arr)(-np.log1p(z))

cl = Class()
cl.set(PARAMS)
cl.compute()
bg = cl.get_background()
zc = bg["z"][::-1]

print(f"Age            CLASS {cl.age():.6f} Gyr   ours {ours(t, 0.0):.6f} Gyr   "
      f"rel {ours(t, 0.0) / cl.age() - 1:+.1e}")
eta0_c = bg["conf. time [Mpc]"][-1]
print(f"eta0           CLASS {eta0_c:.4f} Mpc  ours {ours(eta, 0.0):.4f} Mpc  "
      f"rel {ours(eta, 0.0) / eta0_c - 1:+.1e}")
print(f"Omega_Lambda   CLASS {cl.Omega_Lambda():.8f}")
print("\nRelative differences (ours/CLASS - 1):")
print(f"{'z':>10} {'H':>10} {'t':>10} {'eta':>10} {'d_L':>10}")
for z in [0.1, 0.5, 1, 3, 10, 100, 1089, 1e4, 1e6]:
    H_c = cl.Hubble(z) * 299792.458
    t_c = np.interp(z, zc, bg["proper time [Gyr]"][::-1])
    eta_c = np.interp(z, zc, bg["conf. time [Mpc]"][::-1])
    row = [ours(H, z) / H_c - 1, ours(t, z) / t_c - 1, ours(eta, z) / eta_c - 1]
    row.append(ours(dL, z) / cl.luminosity_distance(z) - 1 if z < 2000 else np.nan)
    print(f"{z:10g} " + " ".join(f"{v:+10.1e}" for v in row))
