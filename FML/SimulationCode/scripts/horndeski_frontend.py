#!/usr/bin/env python3
# ======================================================================================================================
# Front-end of the generic Horndeski gravity model of SimulationCode (gravity_model = Horndeski): from a reduced
# Horndeski Lagrangian
#   S = int d^4x sqrt(-g) [ K(phi, X) - G3(phi, X) box phi + G4(phi) R ] + S_m,   X = -(1/2) (d phi)^2
# (luminal gravitational waves: no G4X, no G5; units M_pl = 1 for G4 = 1/2, i.e. 8 pi G = 1 with G4 = 1/2, and H0 = 1;
# phi dimensionless) it solves the background and writes the quasi-static force of the theory, in the DGP form that
# SimulationCode's screening approximation and multigrid solver use (as Hi-COLA, Wright et al. 2023, arXiv:2209.01666).
#
# Background: the equations are derived here with sympy from the minisuperspace action (lapse N, scale factor a,
# phi(t)) plus dust, radiation and a cosmological constant (rho = 3 Omega a^-3, 3 Omega_r a^-4, 3 Omega_Lambda):
# the N equation is the Friedmann constraint, the a and phi equations give dH/dt and the field acceleration. They are
# integrated in x = ln a (from today back to a_min, or from early times with a shooting for E(a = 1) = 1; see Input).
# The constraint is monitored (printed).
#
# Quasi-static perturbations (Kimura, Kobayashi & Yamamoto 2012, arXiv:1111.6749; Hi-COLA eqs. 2.13-2.20), Newtonian
# gauge ds^2 = -(1 + 2 Phi) dt^2 + a^2 (1 - 2 Psi) dx^2, Q = H dphi / dphi/dt:
#   2 G4 nabla^2 (Psi - Phi) = A1 nabla^2 Q,   2 G4 nabla^2 Psi = a^2 rho_m delta / 2 - A2 nabla^2 Q
#   A0 nabla^2 Q - A1 nabla^2 Psi - A2 nabla^2 Phi + B0 / (a^2 H^2) [(nabla^2 Q)^2 - (nabla_i nabla_j Q)^2] = 0
#   Theta = -dphi X G3X + 2 H G4 + dphi G4phi,   A1 = 2 dG4/dt / H,   A2 = 2 G4 - Theta / H,   B0 = X dphi G3X / H
#   A0 = dTheta/dt / H^2 + Theta / H - 2 G4 - 4 dG4/dt / H + (rho + p)_matter / (2 H^2)
# With alpha_i = A_i / (2 G4), beta0 = B0 / (2 G4), D = alpha0 + 2 alpha1 alpha2 + alpha2^2 and the fifth-force
# potential phi5 = -(alpha1 + alpha2) Q (Phi = GG4 Phi_N + phi5, GG4 = G_G4/G = 1 / (2 G4)), the scalar equation is the
# DGP brane-bending equation (QuasiStaticGravity_DGP.h, with phi = 2 phi5)
#   nabla^2 phi + (r_c^2 / (3 beta a^2)) [(nabla^2 phi)^2 - (nabla_i nabla_j phi)^2] = 8 pi G a^2 delta rho / (3 beta)
#   b5 = 1/(3 beta) = -GG4 (alpha1 + alpha2)^2 / D,   c2 = (r_c H0/c)^2 b5^2 = GG4 beta0 (alpha1 + alpha2) / (2 D^2 E^2)
#   mu = G_eff/G = GG4 + b5,   Sigma = GG4 [1 - (alpha1 + 2 alpha2) (alpha1 + alpha2) / (2 D)]
# (b5 = 0 without a fifth force; c2 = 0 without G3X: no Vainshtein screening, a linear fifth force). The coefficients
# are simplified symbolically with dH/dt and the field acceleration of the background equations and the matter density
# eliminated with the Friedmann constraint, so that their GR parts cancel exactly (at early times the fifth force is
# tiny and round-off would dominate it). Checked (tests/check_horndeski_frontend.py) against the cubic Galileon
# (QuasiStaticGravity_CubicGalileon.h) and Brans-Dicke (mu = (4 + 2 omega) / ((3 + 2 omega) phi), Sigma = 1/phi); the
# generic model reproduces SimulationCode's CubicGalileon gravity model to 4e-6 in P(k).
#
# Output (--out PREFIX):
#   PREFIX_background.txt   z, E = H/H0  (cosmology_model = Tabulated, cosmology_tabulated_format = H)
#   PREFIX_gravity.txt      a, E, dlnE/dlna, GG4, mu, Sigma, b5, c2, phi, dphi/dlna  (gravity_model = Horndeski,
#                           gravity_model_horndeski_filename)
# Use the same Omega_m (cosmology_OmegaCDM + cosmology_Omegab) and the radiation of the parameter file (Omega_r here:
# photons and massless neutrinos; default from h, T_CMB, Neff as FML does).
#
# Input: a JSON file (examples in input/horndeski: cubic_galileon.json, brans_dicke.json) with
#   K, G3, G4         expressions in phi and X (sympy syntax) and the named parameters (default 0, 0, 1/2)
#   parameters        {name: value}
#   Omega_m           today (the CDM + baryons of the parameter file)
#   Omega_r or h (T_CMB, Neff)    the radiation (photons + massless neutrinos)
#   Omega_Lambda      a cosmological constant (default 0)
#   a_min, n_points   the table: a_min - 1 (default 1e-4, 2000 points)
#   initial = "today" (default): phi0 and dphidlna0_guess: dphi/dlna today is the root of the Friedmann constraint
#                     with E = 1 closest to the guess (search_width: the range searched, in units of max(1, |guess|));
#                     integrated back in time (the attractor of a tracker is unstable backwards: its decaying modes
#                     grow, which matters only where the scalar is negligible)
#   initial = "early": a_ini (<= a_min), phi_ini, dphidlna_ini at a_ini, and shoot: the parameter (or Omega_Lambda)
#                     adjusted within shoot_bracket [lo, hi] so that E(a = 1) = 1
#
# Usage: python3 horndeski_frontend.py model.json --out PREFIX
#
# References:
#   Horndeski 1974, Int. J. Theor. Phys. 10, 363; Kobayashi 2019, arXiv:1901.07183 (review)
#   Kimura, Kobayashi & Yamamoto 2012, PRD 85, 024023, arXiv:1111.6749 (the quasi-static equations used here)
#   Wright, Sen Gupta, Baker, Valogiannis & Fiorini 2023 (Hi-COLA), JCAP 03, 040, arXiv:2209.01666, and its code
#       https://github.com/Hi-COLACode/Hi-COLA (the same approach: a symbolic front-end and fast simulations)
#   Sen Gupta, Fiorini & Baker 2024, JCAP 11, 052, arXiv:2407.00855 (Hi-COLA with K-mouflage; the G_G4 factor)
#   Brando, Koyama & Winther 2023, JCAP 06, 045, arXiv:2303.09549 (Vainshtein screening in COLA)
# ======================================================================================================================
import argparse
import json
import sys

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def derive(Kexpr, G3expr, G4expr, phis, Xs, consts):
    """The background equations and the quasi-static coefficients. consts: the symbols of the constants (Omega_m,
    Omega_r, Omega_Lambda and the model parameters), kept symbolic so that the GR parts of the QS coefficients cancel
    exactly (numerically they would leave round-off that dominates at early times, where the scalar is negligible).
    Returns functions of (a, H, phi, dphi/dt, *constants): constraint, dH/dt, field acceleration, and
    (G4, alpha0, alpha1, alpha2, beta0)"""
    t = sp.symbols("t")
    N, a, ph = sp.Function("N")(t), sp.Function("a")(t), sp.Function("phi")(t)
    X = ph.diff(t) ** 2 / (2 * N ** 2)
    Oms, Ors, OLs = consts[:3]

    def at(expr):
        return expr.subs({phis: ph, Xs: X}, simultaneous=True)

    box = -1 / (N * a ** 3) * sp.diff(a ** 3 * ph.diff(t) / N, t)
    R = 6 / N ** 2 * (a.diff(t, 2) / a + a.diff(t) ** 2 / a ** 2 - a.diff(t) * N.diff(t) / (a * N))
    rho = 3 * Oms * a ** -3 + 3 * Ors * a ** -4 + 3 * OLs
    L = N * a ** 3 * (at(Kexpr) - at(G3expr) * box + at(G4expr) * R) - N * a ** 3 * rho
    eqs = sp.calculus.euler.euler_equations(L, [N, a, ph], t)

    # Symbols: A (scale factor), H, Hd = dH/dt, P = phi, PD = dphi/dt, PDD, ...
    A, H, Hd, Hdd, P, PD, PDD, PDDD = sp.symbols("A H Hd Hdd P PD PDD PDDD")
    A4, P4 = sp.symbols("A4 P4")
    rules = [
        (N.diff(t, 3), 0), (N.diff(t, 2), 0), (N.diff(t), 0),
        (a.diff(t, 4), A4), (a.diff(t, 3), A * (Hdd + 3 * H * Hd + H ** 3)),
        (a.diff(t, 2), A * (Hd + H ** 2)), (a.diff(t), A * H),
        (ph.diff(t, 4), P4), (ph.diff(t, 3), PDDD), (ph.diff(t, 2), PDD), (ph.diff(t), PD),
    ]

    def symb(e):
        e = e.lhs - e.rhs
        for old, new in rules:
            e = e.subs(old, new)
        return sp.expand(e.subs({N: 1, a: A, ph: P}))

    eN, ea, ep = (symb(e) for e in eqs)
    for e, name in ((eN, "N"), (ea, "a"), (ep, "phi")):
        if any(e.has(s) for s in (Hdd, PDDD, A4, P4)):
            raise RuntimeError("The " + name + " equation has higher derivatives: not a Horndeski theory?")
    if eN.has(Hd) or eN.has(PDD):
        raise RuntimeError("The constraint has second derivatives")
    Om_onshell = sp.solve(eN, Oms)
    if len(Om_onshell) != 1:
        raise RuntimeError("Could not solve the Friedmann constraint for the matter density")
    Om_onshell = Om_onshell[0]
    sol = sp.solve([ea, ep], [Hd, PDD], dict=True)
    if len(sol) != 1:
        raise RuntimeError("Could not solve the a and phi equations for dH/dt and the field acceleration")
    Hd_expr, PDD_expr = sp.simplify(sol[0][Hd]), sp.simplify(sol[0][PDD])
    args = (A, H, P, PD) + tuple(consts)
    constraint = sp.lambdify(args, eN, "numpy")
    rhs_Hd = sp.lambdify(args, Hd_expr, "numpy")
    rhs_pdd = sp.lambdify(args, PDD_expr, "numpy")

    # Quasi-static coefficients: dH/dt and the field acceleration substituted, the matter density eliminated with the
    # constraint, simplified
    Xv = PD ** 2 / 2
    sub = {phis: P, Xs: Xv}
    G4 = G4expr.subs(sub)
    G4p = sp.diff(G4expr, phis).subs(sub)
    G3X = sp.diff(G3expr, Xs).subs(sub)
    Theta = -PD * Xv * G3X + 2 * H * G4 + PD * G4p
    dTheta = sp.diff(Theta, P) * PD + sp.diff(Theta, PD) * PDD + sp.diff(Theta, H) * Hd
    dG4 = G4p * PD
    rho_plus_p = 3 * Oms * A ** -3 + 4 * Ors * A ** -4
    on_shell = {Hd: Hd_expr, PDD: PDD_expr}

    def simplified(e):
        e = sp.together(sp.expand(e.subs(on_shell).subs(Oms, Om_onshell)))
        try:
            return sp.factor(e)
        except Exception:
            return e

    al0 = simplified((dTheta / H ** 2 + Theta / H - 2 * G4 - 4 * dG4 / H + rho_plus_p / (2 * H ** 2)) / (2 * G4))
    al1 = simplified(dG4 / (H * G4))
    al2 = simplified((2 * G4 - Theta / H) / (2 * G4))
    be0 = simplified(Xv * PD * G3X / (2 * H * G4))
    coeffs = sp.lambdify(args, [G4 + 0 * H, al0, al1, al2, be0], "numpy")
    return constraint, rhs_Hd, rhs_pdd, coeffs


def radiation(cfg):
    if "Omega_r" in cfg:
        return float(cfg["Omega_r"])
    h = float(cfg.get("h", 0.67))
    T = float(cfg.get("T_CMB", 2.7255))
    Neff = float(cfg.get("Neff", 3.046))
    Og = 2.4728e-5 / h ** 2 * (T / 2.7255) ** 4
    return Og * (1.0 + Neff * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0))


def main():
    ap = argparse.ArgumentParser(description="Background and quasi-static force of a reduced Horndeski theory")
    ap.add_argument("config", help="JSON file with K, G3, G4, parameters, Omega_m, ...")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    phis, Xs = sp.symbols("phi X")
    values = {"Omega_m": float(cfg["Omega_m"]), "Omega_r": radiation(cfg),
              "Omega_Lambda": float(cfg.get("Omega_Lambda", 0.0))}
    values.update({k: float(v) for k, v in cfg.get("parameters", {}).items()})
    names = list(values)
    syms = [sp.Symbol(n) for n in names]
    loc = {"phi": phis, "X": Xs, **dict(zip(names, syms))}
    K, G3, G4 = (sp.sympify(str(cfg.get(n, d)), locals=loc) for n, d in (("K", "0"), ("G3", "0"), ("G4", "1/2")))
    a_min = float(cfg.get("a_min", 1e-4))
    npts = int(cfg.get("n_points", 2000))
    print(f"K = {K}\nG3 = {G3}\nG4 = {G4}\n" + ", ".join(f"{n} = {v:.10g}" for n, v in values.items()))

    constraint, rhs_Hd, rhs_pdd, coeffs = derive(K, G3, G4, phis, Xs, syms)

    def consts():
        return [values[n] for n in names]

    def rhs(x, y):
        a = np.exp(x)
        phi, dphi, E = y
        pd = E * dphi
        c = consts()
        dE = rhs_Hd(a, E, phi, pd, *c) / E
        return [dphi, (rhs_pdd(a, E, phi, pd, *c) / E - dE * dphi) / E, dE]

    def roots_of(f, guess, width):
        grid = guess + np.linspace(-1.0, 1.0, 4001) * width
        vals = np.array([f(y) for y in grid])
        roots = [grid[i] for i in range(len(grid)) if vals[i] == 0.0]
        roots += [brentq(f, grid[i], grid[i + 1], xtol=1e-14) for i in range(len(grid) - 1)
                  if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0]
        return sorted(roots)

    def integrate(x0, y0, x1, t_eval=None):
        sol = solve_ivp(rhs, (x0, x1), y0, t_eval=t_eval, method="LSODA", rtol=1e-10, atol=1e-12)
        if not sol.success:
            sys.exit("Background integration failed: " + sol.message)
        return sol

    initial = cfg.get("initial", "today")
    if initial == "today":
        # dphi/dlna today from the Friedmann constraint with E = 1: the root closest to the guess; integrated back
        # in time (fine unless the early-time solution is an attractor: backwards its decaying modes grow)
        phi0 = float(cfg.get("phi0", 0.0))
        guess = float(cfg.get("dphidlna0_guess", 1.0))
        width = max(1.0, abs(guess)) * float(cfg.get("search_width", 2.0))
        roots = roots_of(lambda y: float(constraint(1.0, 1.0, phi0, y, *consts())), guess, width)
        if not roots:
            sys.exit("No root of the Friedmann constraint near dphidlna0_guess (no background with E(0) = 1)")
        dphi0 = min(roots, key=lambda r: abs(r - guess))
        print(f"dphi/dlna today = {dphi0:.10g} (roots: {', '.join(f'{r:.6g}' for r in roots)})")
        xs = np.linspace(0.0, np.log(a_min), npts)
        sol = integrate(0.0, [phi0, dphi0, 1.0], np.log(a_min), xs)
        x, (phi, dphi, E) = sol.t[::-1], (v[::-1] for v in sol.y)
    elif initial == "early":
        # phi and dphi/dlna given at a_ini (e.g. the field at rest deep in the matter or radiation era); E there from
        # the constraint (the root closest to the GR value) and the parameter `shoot` adjusted so that E(a = 1) = 1
        x_ini = np.log(float(cfg.get("a_ini", a_min)))
        phi_i, dphi_i = float(cfg["phi_ini"]), float(cfg.get("dphidlna_ini", 0.0))
        shoot = cfg["shoot"]
        if shoot not in values:
            sys.exit(f"shoot = {shoot}: not Omega_Lambda or a parameter")

        def E_ini():
            a = np.exp(x_ini)
            EGR = np.sqrt(values["Omega_m"] * a ** -3 + values["Omega_r"] * a ** -4 + values["Omega_Lambda"])
            f = lambda E: float(constraint(a, E, phi_i, E * dphi_i, *consts())) / (3 * E ** 2)
            r = roots_of(f, EGR, 0.5 * EGR)
            if not r:
                raise RuntimeError("no root of the constraint at a_ini")
            return min(r, key=lambda v: abs(v - EGR))

        def E_today(q):
            values[shoot] = q
            sol = integrate(x_ini, [phi_i, dphi_i, E_ini()], 0.0)
            return sol.y[2, -1] - 1.0

        lo, hi = cfg["shoot_bracket"]
        q = brentq(E_today, float(lo), float(hi), xtol=1e-13)
        values[shoot] = q
        print(f"shooting: {shoot} = {q:.12g} gives E(a = 1) = 1")
        xs = np.linspace(np.log(a_min), 0.0, npts)
        if x_ini > xs[0]:
            sys.exit("a_ini must be <= a_min")
        sol = integrate(x_ini, [phi_i, dphi_i, E_ini()], 0.0, xs)
        x, (phi, dphi, E) = sol.t, sol.y
    else:
        sys.exit("initial must be today or early")

    a = np.exp(x)
    pd = E * dphi
    c = consts()
    cons = np.array([constraint(ai, Ei, p, q, *c) for ai, Ei, p, q in zip(a, E, phi, pd)]) / (3 * E ** 2)
    print(f"Friedmann constraint |C| / (3 H^2): max {np.max(np.abs(cons)):.2e}")
    Hd = np.array([rhs_Hd(ai, Ei, p, q, *c) for ai, Ei, p, q in zip(a, E, phi, pd)])

    G4v, al0, al1, al2, be0 = (np.broadcast_to(np.asarray(v, dtype=float), a.shape)
                               for v in coeffs(a, E, phi, pd, *c))
    D = al0 + 2 * al1 * al2 + al2 ** 2
    GG4 = 1.0 / (2.0 * G4v)
    s12 = al1 + al2
    with np.errstate(divide="ignore", invalid="ignore"):
        b5 = np.where(s12 != 0.0, -GG4 * s12 ** 2 / D, 0.0)
        # c2 = (r_c H0/c)^2 b5^2 (finite when the fifth force vanishes at early times)
        c2 = np.where(s12 != 0.0, GG4 * be0 * s12 / (2 * D ** 2 * E ** 2), 0.0)
        Sigma = GG4 * (1.0 - np.where(s12 != 0.0, (al1 + 2 * al2) * s12 / (2 * D), 0.0))
    mu = GG4 + b5
    dlnE = Hd / E ** 2
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(c2)) and np.all(np.isfinite(Sigma))):
        print("WARNING: non-finite quasi-static coefficients (D = alpha0 + 2 alpha1 alpha2 + alpha2^2 = 0?)")

    head = (f"# Reduced Horndeski (horndeski_frontend.py {args.config}): K = {K}, G3 = {G3}, G4 = {G4}\n# " +
            ", ".join(f"{n} = {v:.12g}" for n, v in values.items()) + "\n")
    with open(args.out + "_background.txt", "w") as fp:
        fp.write(head + "# z  H/H0\n")
        for ai, Ei in zip(a[::-1], E[::-1]):
            fp.write(f"{1.0 / ai - 1.0:.12e} {Ei:.12e}\n")
    with open(args.out + "_gravity.txt", "w") as fp:
        fp.write(head + "# a  E  dlnE/dlna  GG4  mu  Sigma  b5  c2  phi  dphi/dlna\n")
        for row in zip(a, E, dlnE, GG4, mu, Sigma, b5, c2, phi, dphi):
            fp.write(" ".join(f"{v:.12e}" for v in row) + "\n")
    print(f"Today: mu = {mu[-1]:.6g}, Sigma = {Sigma[-1]:.6g}, G_G4/G = {GG4[-1]:.6g}, 1/(3 beta) = {b5[-1]:.6g}, "
          f"(r_c H0/c)^2 = {c2[-1] / b5[-1] ** 2 if b5[-1] != 0 else 0.0:.6g}")
    print(f"Wrote {args.out}_background.txt and {args.out}_gravity.txt")


if __name__ == "__main__":
    main()
