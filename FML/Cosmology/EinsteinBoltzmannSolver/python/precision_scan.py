#!/usr/bin/env python3
"""Accuracy of a ./cmb precision setting against a high-precision reference, as a chi^2 difference.

Usage:  python3 python/precision_scan.py REF_DIR RUN_DIR [RUN_DIR ...]

Each directory holds cells_lensed.txt (or cells.txt) from ./cmb; the wall time is read ("[Timer total]") from
RUN_DIR.log or RUN_DIR/*.log if present. For every run it prints

  dchi2_planck = sum_ell (2l+1) f_sky/2 tr[(C^-1 dC)^2],  C = C_ref + N_ell (TT, EE, TE jointly), ell = 2..2500
  dchi2_cv     = the same without noise (cosmic-variance limit)

with a Planck-like noise: f_sky = 0.6, Delta_T = 33 muK arcmin, Delta_P = sqrt(2) Delta_T, 7 arcmin FWHM beam.
A systematic error dC biases a parameter by at most sqrt(dchi2) standard deviations (in the worst direction of
parameter space), so dchi2 < 1 cannot be seen in a Planck fit. Also printed: the largest relative error of TT
and EE for 30 <= ell <= 2500, and of TE relative to sqrt(TT EE).

Reference used for the README numbers (Planck 2018 LCDM, lensing, ell_max = 2500; ~9 min, 3 GB):
  ./cmb params/planck2018.ini pert_n_k=700 pert_k_max=0.5 lmax=16 lmax_nu=32 ode_relerr=1e-10 \
        n_k_per_oscillation=10 dlnk_max=0.005 n_eta_per_oscillation=12 dx_recomb=0.001 \
        pert_n_x_recomb=2000 pert_n_x_late=900 bessel_points_per_2pi=48 output_dir=data/ref
"""
import glob
import os
import re
import sys

import numpy as np

LMAX = 2500
FSKY, DT, FWHM = 0.6, 33.0, 7.0   # muK arcmin, arcmin


def load(d):
    f = os.path.join(d, "cells_lensed.txt")
    if not os.path.exists(f):
        f = os.path.join(d, "cells.txt")
    a = np.loadtxt(f)   # ell, D_ell^TT, EE, TE, ... [muK^2]
    ell = a[:, 0].astype(int)
    tt, ee, te = a[:, 1], a[:, 2], a[:, 3]
    fac = ell * (ell + 1) / (2 * np.pi)   # files hold l(l+1)C_l/2pi [muK^2]
    out = {}
    for k, v in (("TT", tt), ("EE", ee), ("TE", te)):
        c = np.zeros(LMAX + 1)
        m = (ell >= 2) & (ell <= LMAX)
        c[ell[m]] = v[m] / fac[m]
        out[k] = c
    return out


def wall_time(d):
    logs = [d + ".log"] + glob.glob(os.path.join(d, "*.log"))
    for lf in logs:
        if os.path.exists(lf):
            m = re.findall(r"\[Timer total\]\D*([\d.]+)", open(lf).read())
            if m:
                return float(m[-1])
    return float("nan")


def dchi2(ref, run, noise):
    ell = np.arange(2, LMAX + 1)
    s = 0.0
    for l in ell:
        C = np.array([[ref["TT"][l] + noise[0][l], ref["TE"][l]], [ref["TE"][l], ref["EE"][l] + noise[1][l]]])
        D = np.array([[run["TT"][l] - ref["TT"][l], run["TE"][l] - ref["TE"][l]],
                      [run["TE"][l] - ref["TE"][l], run["EE"][l] - ref["EE"][l]]])
        M = np.linalg.solve(C, D)
        s += (2 * l + 1) * FSKY / 2 * np.trace(M @ M)
    return s


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = load(sys.argv[1])
    ell = np.arange(LMAX + 1)
    sig = np.radians(FWHM / 60) / np.sqrt(8 * np.log(2))
    b2 = np.exp(ell * (ell + 1) * sig ** 2)
    nT = (np.radians(DT / 60)) ** 2 * b2
    planck = (nT, 2 * nT)
    cv = (0 * nT, 0 * nT)
    print(f"{'run':40s} {'time[s]':>8s} {'dchi2_pl':>9s} {'dchi2_cv':>9s} {'maxTT':>8s} {'maxEE':>8s} {'maxTE':>8s}")
    l = np.arange(30, LMAX + 1)
    for d in sys.argv[2:]:
        try:
            run = load(d)
        except Exception as e:   # noqa: BLE001
            print(f"{d:40s} failed: {e}")
            continue
        eTT = np.max(np.abs(run["TT"][l] / ref["TT"][l] - 1))
        eEE = np.max(np.abs(run["EE"][l] / ref["EE"][l] - 1))
        eTE = np.max(np.abs(run["TE"][l] - ref["TE"][l]) / np.sqrt(ref["TT"][l] * ref["EE"][l]))
        print(f"{os.path.basename(d.rstrip('/')):40s} {wall_time(d):8.1f} {dchi2(ref, run, planck):9.3f} "
              f"{dchi2(ref, run, cv):9.2f} {eTT:8.1e} {eEE:8.1e} {eTE:8.1e}")


if __name__ == "__main__":
    main()
