#!/usr/bin/env python3
"""Regenerate the velocileptors reference of the scaled REPT tables in tests/test_rept.cpp
(EPTRedshiftSpace::table_at_mu_scaled, multipole_components_scaled).

velocileptors' ept_fullresum_varyDz_nu_fftw (used by DESI's DR1 full-shape likelihood): the one-loop terms of the input
P(k) scaled by Dz^4, the linear terms from pcb, pcb_nw at the redshift, and a scale-dependent f(k). Run from
PerturbationTheory/. Requires velocileptors 3.1, numpy, scipy (and pyFFTW). Inputs as generate_rept_reference.py.
"""
from pathlib import Path
import numpy as np
from velocileptors.Utils.loginterp import loginterp
from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT

ref = Path(__file__).resolve().parents[1] / "tests/reference"
k, p = np.loadtxt(ref / "lptrsd_linear_pk.txt", unpack=True)
knw, pnw = np.loadtxt(ref / "rept_nowiggle_pk.txt", unpack=True)
N, emin, emax = 4000, -4, 3
rept = REPT(k, p, pnw=pnw, kmin=1e-2, kmax=0.5, nk=60, rbao=110, cutoff=20, jn=5, N=N, extrap_min=emin,
            extrap_max=emax, threads=1)
kv = rept.kv
# another redshift: growth Dz, a scale-dependent suppression of the linear spectrum and a scale-dependent f(k)
Dz = 0.8
supp = 1.0 - 0.05 * kv
pcb = Dz**2 * loginterp(k, p)(kv) * supp
pcb_nw = Dz**2 * loginterp(knw, pnw)(kv) * supp
fk = 0.7 + 0.05 * kv
rows = []
for apar, aperp in [(1.0, 1.0), (1.03, 0.97)]:
    for mu in [0.15, 0.7]:
        tab = rept.compute_redshift_space_power_table_at_mu(fk, mu, pcb=pcb, pcb_nw=pcb_nw, apar=apar, aperp=aperp, Dz=Dz)
        for kk, r in zip(kv, tab):
            rows.append(["M", apar, aperp, mu, kk] + list(r))
    _, p0, p2, p4 = rept.compute_redshift_space_power_multipoles_tables(fk, pcb=pcb, pcb_nw=pcb_nw, ngauss=4, apar=apar,
                                                                        aperp=aperp, Dz=Dz)
    for ell, tab in [(0, p0), (2, p2), (4, p4)]:
        for kk, r in zip(kv, tab):
            rows.append([f"P{ell}", apar, aperp, kk] + list(r))
with open(ref / "rept_varydz_velocileptors.txt", "w") as fp:
    fp.write("# velocileptors 3.1 ept_fullresum_varyDz_nu_fftw REPT of lptrsd_linear_pk.txt (pnw from "
             "rept_nowiggle_pk.txt), N = 4000, kmin = 1e-2, kmax = 0.5, nk = 60, with Dz = 0.8, pcb = Dz^2 P (1 - 0.05 k), "
             "pcb_nw = Dz^2 Pnw (1 - 0.05 k), f(k) = 0.7 + 0.05 k. Rows: M (apar, aperp, mu, k, 19 columns), "
             "P0/P2/P4 (apar, aperp, k, 19 columns; ngauss = 4)\n")
    for r in rows:
        fp.write(" ".join(r[:1] + [f"{v:.15e}" if isinstance(v, float) else str(v) for v in r[1:]]) + "\n")
