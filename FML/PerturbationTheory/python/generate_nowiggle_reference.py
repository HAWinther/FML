#!/usr/bin/env python3
"""Regenerate the reference of tests/test_nowiggle.cpp (NoWiggle.h).

Run from PerturbationTheory/. Requires velocileptors (https://github.com/sfschen/velocileptors), colossus, numpy and
scipy. Rows: D (k, the no-wiggle P of velocileptors' pnw_dst for tests/reference/lptrsd_linear_pk.txt), G (k,
scipy savgol_filter pipeline of nowiggle_savgol with its default grid), E (k, Eisenstein & Hu 1998 transfer with
and without wiggles from colossus, for Omega_m = 0.3111, Omega_b = 0.049, h = 0.6766, TCMB = 2.7255).
"""
from pathlib import Path
import warnings
import numpy as np
from scipy.signal import savgol_filter
from velocileptors.Utils.pnw_dst import pnw_dst
from velocileptors.Utils.loginterp import loginterp
from colossus.cosmology import power_spectrum as ps

warnings.filterwarnings("ignore")
ref = Path(__file__).resolve().parents[1] / "tests/reference"
k, p = np.loadtxt(ref / "lptrsd_linear_pk.txt", unpack=True)
rows = []
_, pnw = pnw_dst(k, p)
sel = (k > 1e-3) & (k < 10)
for kk, pp in zip(k[sel][::5], pnw[sel][::5]):
    rows.append(["D", kk, pp])
kg = np.logspace(-5, 3, 6000)
W = int(np.ceil(np.log(7) / np.log(kg[-1] / kg[-2])) // 2 * 2 + 1)
pg = loginterp(kg, savgol_filter(loginterp(k, p)(kg), W, 4))
for kk in k[::10]:
    rows.append(["G", kk, pg(kk)])
ke = np.logspace(-4, 2, 61)
te = ps.transferFunction(ke, 0.6766, 0.3111, 0.049, 2.7255, model="eisenstein98")
tz = ps.transferFunction(ke, 0.6766, 0.3111, 0.049, 2.7255, model="eisenstein98_zb")
for a, b, c in zip(ke, te, tz):
    rows.append(["E", a, b, c])
with open(ref / "nowiggle_reference.txt", "w") as fp:
    fp.write("# NoWiggle.h reference (python/generate_nowiggle_reference.py): D pnw_dst, G savgol, E EH98 (T, T_nw)\n")
    for r in rows:
        fp.write(" ".join([r[0]] + [f"{v:.15e}" for v in r[1:]]) + "\n")

# A well-sampled linear P(k) (CAMB, Planck 2018-like, z = 0) for the wiggle-removal tests: the other test spectrum
# is too sparse at the BAO scales (dlnk ~ 0.13 at k ~ 0.2 h/Mpc)
import camb
pars = camb.set_params(H0=67.66, ombh2=0.02242, omch2=0.11933, mnu=0.0, num_massive_neutrinos=0, ns=0.9665,
                       As=2.105e-9, WantTransfer=True, kmax=200.0)
pars.set_matter_power(redshifts=[0.0], kmax=200.0, nonlinear=False)
res = camb.get_results(pars)
kc = np.logspace(np.log10(2e-5), 2, 1500)
interp = res.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True, extrap_kmax=1e3)
pc = interp.P(0.0, kc)
np.savetxt(ref / "nowiggle_camb_pk.txt", np.c_[kc, pc], fmt="%.12e",
           header=f"CAMB {camb.__version__} linear P(k) at z = 0 (h = 0.6766, ombh2 = 0.02242, omch2 = 0.11933, "
                  "n_s = 0.9665, As = 2.105e-9, no neutrinos); k [h/Mpc], P [(Mpc/h)^3]")
