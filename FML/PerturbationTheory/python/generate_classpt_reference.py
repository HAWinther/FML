#!/usr/bin/env python3
"""Regenerate the CLASS-PT reference of tests/test_eftoflss.cpp (EFTofLSS.h) and the linear P(k) of test_tns.cpp.

Run from PerturbationTheory/. Requires CLASS-PT's python wrapper classy (github.com/Michalychforever/CLASS-PT, built
with `make classy`), numpy and scipy. Writes
  tests/reference/classpt_linear_pk.txt  the linear P(k) at z = 0.5 (k [h/Mpc], P [(Mpc/h)^3]; f, h, r_d in the header)
  tests/reference/classpt_eft_ir{Yes,No}.txt  for l = 0, 2, 4 the response of CLASS-PT's pk_gg_l0/l2/l4 to each bias
      monomial (1, b1, b1^2, b2, b1 b2, b2^2, bG2, b1 bG2, b2 bG2, bG2^2, bGamma3, b1 bGamma3), the counterterm template
      T_l = -(dP_l/dc_l)/2 and the b4 response (b1 = 0), with and without IR resummation
"""
from pathlib import Path
import numpy as np
from classy import Class

ref = Path(__file__).resolve().parents[1] / "tests/reference"
z = 0.5
cosmo = {"h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200, "n_s": 0.9649, "A_s": 2.1e-9, "tau_reio": 0.0544,
         "output": "mPk", "P_k_max_h/Mpc": 100.0, "z_pk": z}
kout = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3])

for ir in ["Yes", "No"]:
    c = Class()
    c.set({**cosmo, "non linear": "PT", "IR resummation": ir, "Bias tracers": "Yes", "RSD": "Yes", "AP": "No"})
    c.compute()
    h, f = float(c.h()), float(c.scale_independent_growth_factor_f(z))
    if ir == "Yes":
        k = np.logspace(-5, np.log10(50.0), 2000)
        P = np.array([c.pk_lin(kk * h, z) * h**3 for kk in k])
        np.savetxt(ref / "classpt_linear_pk.txt", np.c_[k, P], fmt="%.12e",
                   header=f"k (h/Mpc), P_lin (Mpc/h)^3 at z = {z} from CLASS-PT. f = {f!r} h = {h!r} "
                          f"r_d = {float(c.rs_drag())!r} Mpc")
    c.initialize_output(kout * h, z, len(kout))  # CLASS-PT's convention: k in 1/Mpc

    def F(b1=0, b2=0, bG2=0, bG3=0, cs0=0, cs2=0, cs4=0, b4=0):
        return np.array([c.pk_gg_l0(b1, b2, bG2, bG3, cs0, 0, 0, 0, b4),
                         c.pk_gg_l2(b1, b2, bG2, bG3, cs2, 0, b4),
                         c.pk_gg_l4(b1, b2, bG2, bG3, cs4, b4)])

    c0 = F()
    cb1, cb11 = (F(b1=1) - F(b1=-1)) / 2, (F(b1=1) + F(b1=-1)) / 2 - c0
    cb2, cb22 = (F(b2=1) - F(b2=-1)) / 2, (F(b2=1) + F(b2=-1)) / 2 - c0
    cg, cgg = (F(bG2=1) - F(bG2=-1)) / 2, (F(bG2=1) + F(bG2=-1)) / 2 - c0
    cb1b2 = F(b1=1, b2=1) - c0 - cb1 - cb11 - cb2 - cb22
    cb1g = F(b1=1, bG2=1) - c0 - cb1 - cb11 - cg - cgg
    cb2g = F(b2=1, bG2=1) - c0 - cb2 - cb22 - cg - cgg
    cG3 = F(bG3=1) - c0
    cb1G3 = F(b1=1, bG3=1) - c0 - cb1 - cb11 - cG3
    T = -(F(cs0=1, cs2=1, cs4=1) - c0) / 2
    B4 = F(b4=1) - c0
    cols = [c0, cb1, cb11, cb2, cb1b2, cb22, cg, cb1g, cb2g, cgg, cG3, cb1G3, T, B4]
    out = [kout] + [cc[l] for l in range(3) for cc in cols]
    np.savetxt(ref / f"classpt_eft_ir{ir}.txt", np.array(out).T, fmt="%.12e",
               header=f"CLASS-PT ({ir} IR resummation), z = {z}, f = {f!r}, h = {h!r}, r_d = {float(c.rs_drag())!r} Mpc. "
                      "Columns: k (h/Mpc), then for l = 0, 2, 4: the 12 bias monomials, T_l, the b4 response")
