#!/usr/bin/env python3
#==========================================================================
# Check of scripts/horndeski_frontend.py (the generic reduced Horndeski model, gravity_model = Horndeski) against
# theories with known answers:
#   * the cubic Galileon on its tracker (input/horndeski/cubic_galileon.json): E^4 - rho E^2 = Omega_phi0, and the
#     quasi-static force of QuasiStaticGravity_CubicGalileon.h, mu = 1 + Omega_phi / (1 - 2 eps - Omega_phi) and
#     (r_c H0/c)^2 = E^2 / (2 Omega_phi Omega_phi0) (a > 0.05: integrated back from today the frontend leaves the
#     tracker, an attractor, at early times where the Galileon is negligible)
#   * Brans-Dicke (input/horndeski/brans_dicke.json): mu = (4 + 2 omega) / ((3 + 2 omega) phi), Sigma = 1/phi, no
#     cubic term (c2 = 0), and dln phi/dlna -> 1/(1 + omega) in the matter era
# Usage: python3 check_horndeski_frontend.py <outdir>
#==========================================================================
import json
import os
import re
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
frontend = os.path.join(here, "..", "scripts", "horndeski_frontend.py")
indir = os.path.join(here, "..", "input", "horndeski")
outdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "horndeski_check")
os.makedirs(outdir, exist_ok=True)
failed = []


def check(ok, what, value):
    print(f"[{'PASS' if ok else 'FAIL'}] {what}: {value:.3e}")
    if not ok:
        failed.append(what)


def run(name):
    out = os.path.join(outdir, name)
    subprocess.run([sys.executable, frontend, os.path.join(indir, name + ".json"), "--out", out], check=True,
                   capture_output=True)
    d = np.loadtxt(out + "_gravity.txt")
    head = open(out + "_gravity.txt").readlines()[1]
    vals = dict(re.findall(r"(\w+) = ([-0-9.e+]+)", head))
    return d.T, {k: float(v) for k, v in vals.items()}


# Cubic Galileon
(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi), v = run("cubic_galileon")
Om, Or = v["Omega_m"], v["Omega_r"]
Ophi0 = 1 - Om - Or
rho = Om * a ** -3 + Or * a ** -4
E2 = 0.5 * (rho + np.sqrt(rho ** 2 + 4 * Ophi0))
Ophi = Ophi0 / E2 ** 2
D = 1 - 2 * dlnE - Ophi
s = a > 0.05
err = np.max(np.abs(E / np.sqrt(E2) - 1))
check(err < 1e-8, "cubic Galileon: H(a) of the tracker", err)
err = np.max(np.abs(mu[s] / (1 + Ophi[s] / D[s]) - 1))
check(err < 1e-6, "cubic Galileon: mu (a > 0.05)", err)
err = np.max(np.abs(c2[s] / (E2[s] / (2 * Ophi[s] * Ophi0) * (Ophi[s] / D[s]) ** 2) - 1))
check(err < 2e-4, "cubic Galileon: (r_c H0/c)^2 b5^2 (a > 0.05)", err)
check(np.max(np.abs(Sigma / mu - 1)) < 1e-12, "cubic Galileon: Sigma = mu", np.max(np.abs(Sigma / mu - 1)))

# Brans-Dicke
(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi), v = run("brans_dicke")
w = v["omega"]
err = np.max(np.abs(mu * phi * (3 + 2 * w) / (4 + 2 * w) - 1))
check(err < 1e-10, "Brans-Dicke: mu = (4 + 2 omega) / ((3 + 2 omega) phi)", err)
check(np.max(np.abs(Sigma * phi - 1)) < 1e-10, "Brans-Dicke: Sigma = 1/phi", np.max(np.abs(Sigma * phi - 1)))
check(np.max(np.abs(c2)) == 0.0, "Brans-Dicke: no cubic term", np.max(np.abs(c2)))
i = np.argmin(np.abs(a - 0.1))
err = abs(dphi[i] / phi[i] * (1 + w) - 1)
check(err < 0.01, "Brans-Dicke: dln phi/dlna = 1/(1 + omega) in the matter era (a = 0.1)", err)

print("All Horndeski frontend checks passed" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
