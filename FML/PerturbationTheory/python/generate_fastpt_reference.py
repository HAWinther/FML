#!/usr/bin/env python3
"""Regenerate the FAST-PT reference of tests/test_tns.cpp (TNS.h).

Run from PerturbationTheory/ after generate_classpt_reference.py (it uses tests/reference/classpt_linear_pk.txt).
Requires FAST-PT (pip install fast-pt; 4.1.1 used) and numpy. Writes tests/reference/tns_fastpt.txt: k, the one-loop
P22 + P13, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl (one_loop_dd_bias_b3nl) and the TNS A + B terms by powers
of mu (RSD_ABsum_components) for the growth rate f of the linear P(k) file.
"""
from pathlib import Path
import numpy as np
import fastpt

ref = Path(__file__).resolve().parents[1] / "tests/reference"
d = np.loadtxt(ref / "classpt_linear_pk.txt")
f = float(open(ref / "classpt_linear_pk.txt").readline().split("f = ")[1].split()[0])
k, P = d[:, 0], d[:, 1]
sel = (k >= 1e-4) & (k <= 10.0)
k, P = k[sel], P[sel]
k, P = k[: len(k) // 2 * 2], P[: len(k) // 2 * 2]
fpt = fastpt.FASTPT(k, to_do=["one_loop_dd", "dd_bias", "RSD"], low_extrap=-5, high_extrap=3, n_pad=len(k) // 2)
P1 = fpt.one_loop_dd(P, C_window=0.75)[0]
bias = fpt.one_loop_dd_bias_b3nl(P, C_window=0.75)
AB = fpt.RSD_ABsum_components(P, f, C_window=0.75)
kout = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
cols = [np.interp(np.log(kout), np.log(k), c * np.ones_like(k)) for c in [P1] + list(bias[2:]) + list(AB)]
np.savetxt(ref / "tns_fastpt.txt", np.c_[kout, np.array(cols).T], fmt="%.12e",
           header=f"FAST-PT {fastpt.__version__}, f = {f!r}. Columns: k (h/Mpc), P22+P13, Pd1d2, Pd2d2, Pd1s2, Pd2s2, "
                  "Ps2s2, sig4, sig3nl, (A+B) mu^2, mu^4, mu^6, mu^8")
