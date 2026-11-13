#!/usr/bin/env python3
#==========================================================================
# Check of scripts/horndeski_frontend.py (the generic reduced Horndeski model, gravity_model = Horndeski) against
# theories with known answers:
#   * the cubic Galileon on its tracker (input/horndeski/cubic_galileon.json): E^4 - rho E^2 = Omega_phi0, the
#     quasi-static force of QuasiStaticGravity_CubicGalileon.h, mu = 1 + Omega_phi / (1 - 2 eps - Omega_phi) and
#     (r_c H0/c)^2 = E^2 / (2 Omega_phi Omega_phi0), and the alphas alpha_K = 6 Omega_phi, alpha_B = 2 Omega_phi,
#     alpha_M = 0 (test_horndeski), Rx = a^2 (rho + p)_phi
#   * Brans-Dicke (input/horndeski/brans_dicke.json): mu = (4 + 2 omega) / ((3 + 2 omega) phi), Sigma = 1/phi, no
#     cubic term (c2 = 0), dln phi/dlna -> 1/(1 + omega) in the matter era, and M*^2 = phi, alpha_M = -alpha_B =
#     dln phi/dlna, alpha_K = omega alpha_M^2
#   * K-mouflage (input/horndeski/kmouflage.json): its Einstein-frame background, integrated independently here and
#     mapped to the Jordan frame, and mu = A^2 (1 + 2 beta^2 / K_X), G_G4/G = Sigma = A^2
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
(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi, M2, dM2, aK, aB, aM, daK, daB, Rx), v = run("cubic_galileon")
Om, Or = v["Omega_m"], v["Omega_r"]
Ophi0 = 1 - Om - Or
rho = Om * a ** -3 + Or * a ** -4
E2 = 0.5 * (rho + np.sqrt(rho ** 2 + 4 * Ophi0))
Ophi = Ophi0 / E2 ** 2
D = 1 - 2 * dlnE - Ophi
s = a > 1e-6
err = np.max(np.abs(E / np.sqrt(E2) - 1))
check(err < 1e-6, "cubic Galileon: H(a) of the tracker", err)
err = np.max(np.abs(mu[s] / (1 + Ophi[s] / D[s]) - 1))
check(err < 1e-6, "cubic Galileon: mu (a > 1e-6)", err)
err = np.max(np.abs(c2[s] / (E2[s] / (2 * Ophi[s] * Ophi0) * (Ophi[s] / D[s]) ** 2) - 1))
check(err < 1e-5, "cubic Galileon: (r_c H0/c)^2 b5^2 (a > 1e-6)", err)
check(np.max(np.abs(Sigma / mu - 1)) < 1e-12, "cubic Galileon: Sigma = mu", np.max(np.abs(Sigma / mu - 1)))
# alphas on the tracker (test_horndeski): alpha_K = 6 Omega_phi, alpha_B = 2 Omega_phi, alpha_M = 0, M*^2 = 1
err = max(np.max(np.abs(aK - 6 * Ophi)), np.max(np.abs(aB - 2 * Ophi))) / np.max(Ophi)
check(err < 5e-6, "cubic Galileon: alpha_K = 6 Omega_phi, alpha_B = 2 Omega_phi", err)
check(np.max(np.abs(aM)) + np.max(np.abs(dM2)) == 0.0, "cubic Galileon: alpha_M = 0, M*^2 = 1", np.max(np.abs(aM)))
x = np.log(a)
s = (a > 1e-6) & (a < 2.5)
err = np.max(np.abs(daK - np.gradient(aK, x))[s]) / np.max(np.abs(daK[s]))
check(err < 1e-4, "cubic Galileon: dalpha_K/dlna", err)
# Rx = a^2 (rho + p)_phi for M*^2 = 1, with rho_phi = 3 Omega_phi0 / E^2 (8 pi G = H0 = 1): 2 a^2 Omega_phi0 dlnE/dlna
# / E^2
Rx_an = 2 * a ** 2 * Ophi0 * dlnE / E ** 2
err = np.max(np.abs(Rx - Rx_an)[s] / np.max(np.abs(Rx_an[s])))
check(err < 1e-6, "cubic Galileon: Rx", err)


# Brans-Dicke
(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi, M2, dM2, aK, aB, aM, daK, daB, Rx), v = run("brans_dicke")
w = v["omega"]
err = np.max(np.abs(mu * phi * (3 + 2 * w) / (4 + 2 * w) - 1))
check(err < 1e-10, "Brans-Dicke: mu = (4 + 2 omega) / ((3 + 2 omega) phi)", err)
check(np.max(np.abs(Sigma * phi - 1)) < 1e-10, "Brans-Dicke: Sigma = 1/phi", np.max(np.abs(Sigma * phi - 1)))
check(np.max(np.abs(c2)) == 0.0, "Brans-Dicke: no cubic term", np.max(np.abs(c2)))
i = np.argmin(np.abs(a - 0.1))
err = abs(dphi[i] / phi[i] * (1 + w) - 1)
check(err < 0.01, "Brans-Dicke: dln phi/dlna = 1/(1 + omega) in the matter era (a = 0.1)", err)

# alphas (test_horndeski): M*^2 = phi, alpha_M = -alpha_B = dln phi/dlna, alpha_K = omega alpha_M^2
err = max(np.max(np.abs(M2 / phi - 1)), np.max(np.abs(aM - dphi / phi)), np.max(np.abs(aB + aM)),
          np.max(np.abs(aK - w * aM ** 2)) / np.max(np.abs(aK)))
check(err < 1e-10, "Brans-Dicke: M*^2 = phi, alpha_M = -alpha_B = dln phi/dlna, alpha_K = omega alpha_M^2", err)


# K-mouflage (kmouflage.json, n = 2) against its Einstein-frame equations (Brax & Valageas 2014; Sen Gupta, Fiorini &
# Baker 2024, eqs. 3.5-3.9 and 4.5-4.6), integrated here independently: 8 pi G = 1, X = (dvarphi/dt_E)^2 / (2 lam^2),
#   3 H_E^2 = 3 Omega_m A a_E^-3 + 3 Omega_r a_E^-4 + lam^2 (2 X K_X - K),   K = -1 + X + K0 X^2,  A = exp(beta varphi)
#   (K_X + 2 X K_XX) d^2varphi/dt_E^2 = -3 H_E K_X dvarphi/dt_E - beta 3 Omega_m A a_E^-3
# (densities of the Jordan frame times A^4; time in units of the Jordan 1/H0) and mapped to the Jordan frame:
# a_J = A a_E, H_J = (H_E + beta dvarphi/dt_E) / A, the frontend's field phi = 1/A^2. Then G_G4/G = A^2,
# mu = A^2 (1 + 2 beta^2 / K_X) and Sigma = A^2 (the conformal factor cancels in the lensing potential), and
# M*^2 = 1/A^2, alpha_B = -alpha_M (G3 = 0)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi, M2, dM2, aK, aB, aM, daK, daB, Rx), v = run("kmouflage")
Om, Or, be, K0, lam = v["Omega_m"], v["Omega_r"], v["beta"], v["K0"], v["lam"]


def km_rhs(N, y):
    vphi, vd = y
    aE, A = np.exp(N), np.exp(be * y[0])
    X = vd ** 2 / (2 * lam ** 2)
    rm = 3 * Om * A * aE ** -3
    H = np.sqrt((rm + 3 * Or * aE ** -4 + lam ** 2 * (1 + X + 3 * K0 * X ** 2)) / 3)
    KX = 1 + 2 * K0 * X
    vdd = (-3 * H * KX * vd - be * rm) / (KX + 4 * K0 * X)
    return [vd / H, vdd / H]


aE0 = 1e-8
H0E = np.sqrt(Om * aE0 ** -3 + Or * aE0 ** -4)
S = -be * 3 * Om * aE0 ** -3 / (2 * H0E)  # the attractor K_X dvarphi/dt = -beta rho_m / (2 H)
vd0 = brentq(lambda q: (1 + K0 * q ** 2 / lam ** 2) * q - S, S, 0.0)
sol = solve_ivp(km_rhs, (np.log(aE0), np.log(3.0)), [0.0, vd0], method="LSODA", rtol=1e-11, atol=1e-30,
                dense_output=True)
NE = np.linspace(np.log(1e-6), np.log(2.5), 4000)
vphi, vd = sol.sol(NE)
A = np.exp(be * vphi)
X = vd ** 2 / (2 * lam ** 2)
HE = np.sqrt((3 * Om * A * np.exp(-3 * NE) + 3 * Or * np.exp(-4 * NE) + lam ** 2 * (1 + X + 3 * K0 * X ** 2)) / 3)
aJ = A * np.exp(NE)
HJ = (HE + be * vd) / A
s = (aJ > 1e-5) & (aJ < 2.0)
at = lambda col: interp1d(np.log(a), col, kind="cubic")(np.log(aJ[s]))
err = np.max(np.abs(at(E) / HJ[s] - 1))
check(err < 1e-6, "K-mouflage: H(a) = the Einstein-frame solution in the Jordan frame (1e-5 < a < 2)", err)
err = max(np.max(np.abs(at(phi) * A[s] ** 2 - 1)), np.max(np.abs(at(GG4) / A[s] ** 2 - 1)),
          np.max(np.abs(at(Sigma) / A[s] ** 2 - 1)))
check(err < 1e-6, "K-mouflage: phi = 1/A^2, G_G4/G = Sigma = A^2", err)
mu_E = A[s] ** 2 * (1 + 2 * be ** 2 / (1 + 2 * K0 * X[s]))
err = np.max(np.abs(at(mu) / mu_E - 1))
check(err < 1e-6, "K-mouflage: mu = A^2 (1 + 2 beta^2 / K_X)", err)
err = max(np.max(np.abs(M2 / phi - 1)),
          np.max(np.abs(aB + aM)), np.max(np.abs(aM - dphi / phi)))
check(err < 1e-10, "K-mouflage: M*^2 = 1/A^2, alpha_M = -alpha_B = dln M*^2/dlna", err)
i = np.argmin(np.abs(a - 1.0))
print(f"K-mouflage today: A = {np.sqrt(GG4[i]):.6f}, mu = {mu[i]:.6f}, lam = {lam:.10g}")

print("All Horndeski frontend checks passed" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
