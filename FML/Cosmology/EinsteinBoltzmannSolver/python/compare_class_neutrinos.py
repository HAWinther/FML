#!/usr/bin/env python3
"""Compare massive-neutrino runs (./cmb params/neutrinos.ini m_nu=... output_dir=data/neutrinos_<m>) with CLASS.

Usage:  python3 python/compare_class_neutrinos.py REFDIR DIR [DIR ...]
REFDIR is the massless run (m_nu=0); each DIR contains params.txt, cells.txt, cells_lensed.txt
and matter_power.txt. CLASS runs with the same parameters (N_ncdm = 1, m_ncdm, T_ncdm, N_ur) and lensing.
To isolate the neutrino physics from the small differences in recombination, the effect of the mass,
X(m_nu)/X(0) - 1, is compared between the codes for C_ell^TT, EE, lensed TT, C_ell^phiphi, P_m and P_cb.
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

LMAX = 2000


def run(d):
    p = read_params(os.path.join(d, "params.txt"))
    cp = class_params(p)
    cp.update({"output": "tCl,pCl,lCl,mPk", "lensing": "yes", "l_max_scalars": LMAX + 500, "P_k_max_1/Mpc": 1.0})
    c = Class()
    c.set(cp)
    c.compute()
    o = {"cl": np.loadtxt(os.path.join(d, "cells.txt")), "len": np.loadtxt(os.path.join(d, "cells_lensed.txt")),
         "pk": np.loadtxt(os.path.join(d, "matter_power.txt"))}
    o["cl"] = o["cl"][o["cl"][:, 0] <= LMAX]  # with lensing cells.txt extends beyond ell_max
    ell = o["cl"][:, 0].astype(int)
    raw, lens = c.raw_cl(LMAX), c.lensed_cl(LMAX)
    h = p["h"]
    k = o["pk"][:, 0]
    m = k * h < 0.9
    cl = {"TT": raw["tt"][ell], "EE": raw["ee"][ell], "TTlens": lens["tt"][ell], "pp": raw["pp"][ell],
          "Pm": np.array([c.pk_lin(kk * h, 0.0) for kk in k[m]]),
          "Pcb": np.array([c.pk_cb_lin(kk * h, 0.0) for kk in k[m]]) if p.get("m_nu", 0) > 0 else None}
    if cl["Pcb"] is None:
        cl["Pcb"] = cl["Pm"]
    ours = {"TT": o["cl"][:, 1], "EE": o["cl"][:, 2], "TTlens": o["len"][: len(ell), 1], "pp": o["len"][: len(ell), 5],
            "Pm": o["pk"][m, 1], "Pcb": o["pk"][m, 2]}
    return p, ell, k[m], ours, cl


p0, ell, k, o0, c0 = run(sys.argv[1])
for d in sys.argv[2:]:
    p, ell_, k_, o, c = run(d)
    print(f"m_nu = {p['m_nu']} eV:  effect of the mass, max |ours - CLASS| of X(m)/X(0) - 1  [range of CLASS effect]")
    for key in ("TT", "EE", "TTlens", "pp"):
        ro, rc = o[key] / o0[key], c[key] / c0[key]
        s = (ell >= 2) & (ell <= LMAX)
        print(f"  {key:7s} ell 2-{LMAX}: {np.max(np.abs(ro[s] - rc[s])):.2e}   [{np.min(rc[s]) - 1:+.4f}, {np.max(rc[s]) - 1:+.4f}]")
    for key in ("Pm", "Pcb"):
        ro, rc = o[key] / o0[key], c[key] / c0[key]
        print(f"  {key:7s} k < 0.9/Mpc: {np.max(np.abs(ro - rc)):.2e}   [{np.min(rc) - 1:+.4f}, {np.max(rc) - 1:+.4f}]")
    h = p["h"]
    print(f"  absolute P_m vs CLASS: max {np.max(np.abs(o['Pm'] / (c['Pm'] * h**3) - 1)):.2e}")
