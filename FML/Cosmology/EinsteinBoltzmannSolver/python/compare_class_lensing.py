#!/usr/bin/env python3
"""Compare the lensed spectra of ./cmb lensing=true with CLASS.

Usage:  python3 python/compare_class_lensing.py [DIR] [--plot]
DIR (default data/) must contain params.txt, cells.txt (unlensed) and cells_lensed.txt
(ell, lensed D_TT, D_EE, D_TE, D_BB in muK^2, [l(l+1)]^2 C_pp/2pi).
CLASS runs with the same parameters and lensing on. To separate lensing from the small differences
in the unlensed spectra (recombination), the lensed/unlensed ratios of the two codes are compared.
Also compares C_pp and the lensing B modes. Unit tests of the lensing module itself are in
tests/test_lensing.cpp.
With --plot, writes DIR/compare_class_lensing.png.
"""
import os
import sys

import numpy as np
from classy import Class

sys.path.insert(0, os.path.dirname(__file__))
from params import class_params, read_params  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d = args[0] if args else "data"
p = read_params(os.path.join(d, "params.txt"))
U = np.loadtxt(os.path.join(d, "cells.txt"))
L = np.loadtxt(os.path.join(d, "cells_lensed.txt"))
ell = L[:, 0].astype(int)
lmax = int(ell.max())
Uint = {k: np.interp(ell, U[:, 0], U[:, k]) for k in (1, 2)}

cp = class_params(p)
cp.update({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": lmax + 500})  # CLASS lenses to l_max - 500
cl = Class()
cl.set(cp)
cl.compute()
lc, uc = cl.lensed_cl(lmax), cl.raw_cl(lmax)
f = ell * (ell + 1) / (2 * np.pi) * (p["TCMB"] * 1e6) ** 2
rat_c = {"TT": lc["tt"][ell] / uc["tt"][ell], "EE": lc["ee"][ell] / uc["ee"][ell]}
rat_o = {"TT": L[:, 1] / Uint[1], "EE": L[:, 2] / Uint[2]}
pp_c = (ell * (ell + 1.0)) ** 2 * uc["pp"][ell] / (2 * np.pi)
bb_c = f * lc["bb"][ell]

print(f"Omega_k = {p['OmegaK']}, ell_max = {lmax}")
print("  ell range    |dratio TT|  |dratio EE|  C_pp rel    BB rel")
for lo, hi in [(2, 200), (200, 1000), (1000, 2000), (2000, 3000)]:
    s = (ell >= lo) & (ell <= min(hi, lmax))
    if not s.any():
        continue
    dt = np.max(np.abs(rat_o["TT"][s] - rat_c["TT"][s]))
    de = np.max(np.abs(rat_o["EE"][s] - rat_c["EE"][s]))
    rp = np.max(np.abs(L[s, 5] / pp_c[s] - 1))
    rb = np.max(np.abs(L[s, 4] / bb_c[s] - 1))
    print(f"  {lo:4d}-{min(hi, lmax):4d}    {dt:.2e}     {de:.2e}     {rp:.2e}    {rb:.2e}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].semilogx(ell, rat_o["TT"] - 1, label="ours")
    ax[0].semilogx(ell, rat_c["TT"] - 1, "--", label="CLASS")
    ax[0].set_ylabel("lensed/unlensed TT - 1")
    ax[0].legend()
    ax[1].loglog(ell, L[:, 5], label="ours")
    ax[1].loglog(ell, pp_c, "--", label="CLASS")
    ax[1].set_ylabel(r"$[\ell(\ell+1)]^2C_\ell^{\phi\phi}/2\pi$")
    ax[2].loglog(ell, L[:, 4], label="ours")
    ax[2].loglog(ell, bb_c, "--", label="CLASS")
    ax[2].set(xlabel=r"$\ell$", ylabel=r"lensed $D_\ell^{BB}$ [$\mu$K$^2$]")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "compare_class_lensing.png"), dpi=120)
