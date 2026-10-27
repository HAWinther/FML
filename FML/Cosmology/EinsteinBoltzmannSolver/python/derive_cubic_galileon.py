#!/usr/bin/env python3
"""Derive the background and linear perturbation equations of the cubic Galileon in Newtonian gauge (used in
src/models/CubicGalileon.cpp). Run: python3 python/derive_cubic_galileon.py   (needs sympy)

Action (mostly-plus signature, kappa = 8 pi G = 1):
    S = int sqrt(-g) [ R/2 + K(X) - G(X) box phi ] + S_m,   X = -(1/2) g^{mn} d_m phi d_n phi,
    cubic Galileon: K = c2 X, G = g3 X
Scalar stress tensor and field equation (kinetic gravity braiding; Deffayet et al. 2010, Kobayashi et al. 2010):
    T_mn = K_X phi_m phi_n + K g_mn - G_X box phi phi_m phi_n - (G_m phi_n + G_n phi_m) + g_mn G^l phi_l
    nabla^m J_m = 0,   J_m = (K_X - G_X box phi) phi_m - G_X X_m   (Euler-Lagrange equation of the action)
The script first checks nabla^m T_mn = (nabla^l J_l) phi_n identically (to first order on the perturbed metric),
which validates T_mn and the field equation together, then prints the Einstein equations
G^m_n = T^m_n(phi) + T^m_n(matter) and the field equation for
    ds^2 = a^2 [ -(1 + 2 Psi) dtau^2 + (1 - 2 Phi_P) dx^2 ],   phi = F(tau) + dF(tau) e^{ikx}
(Phi_P = -Phi of the code), in the same form as python/derive_brans_dicke.py.
"""
import sympy as sp

tau, x, y, z, k, eps, c2, g3 = sp.symbols("tau x y z k epsilon c2 g3")
a = sp.Function("a")(tau)
F = sp.Function("F")(tau)
Psi, PhiP, dF = (sp.Function(n)(tau) for n in ("Psi", "PhiP", "dF"))
e = sp.exp(sp.I * k * x)
X4 = [tau, x, y, z]

g = sp.diag(-a**2 * (1 + 2 * eps * Psi * e), *(3 * [a**2 * (1 - 2 * eps * PhiP * e)]))
ginv = sp.diag(*[1 / g[i, i] for i in range(4)])
phi = F + eps * dF * e


def s1(expr):
    """Series to first order in eps"""
    expr = sp.expand(expr)
    return sp.expand(expr.subs(eps, 0) + eps * sp.diff(expr, eps).subs(eps, 0))


def lin(expr):
    return sp.simplify(sp.diff(expr, eps).subs(eps, 0))


def bg(expr):
    return sp.simplify(expr.subs(eps, 0))


Gam = [[[s1(sum(ginv[l, m] * (sp.diff(g[m, i], X4[j]) + sp.diff(g[m, j], X4[i]) - sp.diff(g[i, j], X4[m]))
                for m in range(4)) / 2) for j in range(4)] for i in range(4)] for l in range(4)]


def cov_dd(f):
    """nabla_i nabla_j f"""
    df = [sp.diff(f, X4[i]) for i in range(4)]
    return [[s1(sp.diff(f, X4[i], X4[j]) - sum(Gam[l][i][j] * df[l] for l in range(4))) for j in range(4)] for i in range(4)]


ph = [sp.diff(phi, X4[i]) for i in range(4)]
nn = cov_dd(phi)
box = s1(sum(ginv[i, i] * nn[i][i] for i in range(4)))
X = s1(-sum(ginv[i, i] * ph[i] ** 2 for i in range(4)) / 2)
K, KX = c2 * X, c2
G, GX = g3 * X, g3
Gd = [s1(sp.diff(G, X4[i])) for i in range(4)]
Xd = [s1(sp.diff(X, X4[i])) for i in range(4)]
Gphi = s1(sum(ginv[i, i] * Gd[i] * ph[i] for i in range(4)))

T = [[s1(KX * ph[i] * ph[j] + K * g[i, j] - GX * box * ph[i] * ph[j] - (Gd[i] * ph[j] + Gd[j] * ph[i]) + g[i, j] * Gphi)
      for j in range(4)] for i in range(4)]
J = [s1((KX - GX * box) * ph[i] - GX * Xd[i]) for i in range(4)]
# divergence of a vector J^m = g^{mm} J_m: (1/sqrt(-g)) d_m (sqrt(-g) J^m)
sqrtg = sp.sqrt(-g.det())
divJ = s1(sum(sp.diff(sqrtg * ginv[i, i] * J[i], X4[i]) for i in range(4)) / sqrtg)
# nabla^m T_mn = g^{mm} (d_m T_mn - Gam^l_mm T_ln - Gam^l_mn T_ml)
divT = [s1(sum(ginv[m, m] * (sp.diff(T[m][n], X4[m]) - sum(Gam[l][m][m] * T[l][n] + Gam[l][m][n] * T[m][l]
                                                             for l in range(4))) for m in range(4))) for n in range(4)]
ok = True
for n in range(2):   # background and first order separately (simplify does not reduce the sum to 0)
    r = sp.expand(divT[n] - divJ * ph[n])
    ok = ok and sp.simplify(r.subs(eps, 0)) == 0 and sp.simplify(sp.expand(sp.diff(r, eps).subs(eps, 0) / e)) == 0
print("Check nabla^m T_mn = (nabla.J) phi_n:", ok)
if not ok:
    raise SystemExit("stress tensor / field equation inconsistent")

# Einstein tensor
def ricci(i, j):
    r = 0
    for l in range(4):
        r += sp.diff(Gam[l][i][j], X4[l]) - sp.diff(Gam[l][i][l], X4[j])
        for m in range(4):
            r += Gam[l][l][m] * Gam[m][i][j] - Gam[l][j][m] * Gam[m][i][l]
    return s1(r)


R = [[ricci(i, j) for j in range(4)] for i in range(4)]
Rs = s1(sum(ginv[i, i] * R[i][i] for i in range(4)))
Gmix = [[s1(ginv[i, i] * (R[i][j] - g[i, j] * Rs / 2)) for j in range(4)] for i in range(4)]
Tmix = [[s1(ginv[i, i] * T[i][j]) for j in range(4)] for i in range(4)]
E = [[s1(Gmix[i][j] - Tmix[i][j]) for j in range(4)] for i in range(4)]   # = T^m_n(matter)

print("\nBackground (cosmic quantities in conformal time):")
print("  rho_phi = -T^0_0 =", sp.simplify(-bg(Tmix[0][0])))
print("  p_phi   =  T^x_x =", sp.simplify(bg(Tmix[1][1])))
print("  nabla.J =", sp.simplify(bg(divJ)))
print("\nPerturbations (divided by e^{ikx}); matter side as in derive_brans_dicke.py:")
print("  delta(G - T_phi)^0_0 [= -delta rho_m] =", sp.simplify(lin(E[0][0]) / e))
print("  delta(G - T_phi)^0_x [= (rho+p)_m v_x] =", sp.simplify(lin(E[0][1]) / e))
print("  delta(G - T_phi)^x_x - ^y_y [= -(3/2)(rho+p)_m sigma] =", sp.simplify((lin(E[1][1]) - lin(E[2][2])) / e))
print("  delta(nabla.J) [= 0] =", sp.simplify(lin(divJ) / e))

# -----------------------------------------------------------------------------
# Reduction to a first-order system for the code. Matter (kappa = 1):
#   a^2 delta rho = drho, a^2 delta p = dp, Q = a^2 (rho+p) theta / k^2, Sig = (3/2) a^2 (rho+p) sigma,
#   R = a^2 (rho+p)
# Total matter momentum conservation (Thomson exchange cancels):
#   Q' = -2 H Q + dp - (2/3) Sig + R Psi        (H = a'/a)
# Variables: Phi_P, dF, u = dF' - F' Psi. Phi_P'' and Psi' in the field equation are eliminated with the
# tau-derivative of the 0i equation.
# -----------------------------------------------------------------------------
print("\nReduced system:")
t = tau
H, Hd, F1, F2, F3 = sp.symbols("H Hd F1 F2 F3")         # a'/a, (a'/a)', F', F'', F'''
P, P1, Ps, Ps1, D, D1, D2, U, U1 = sp.symbols("P P1 Psi Psi1 D D1 D2 U U1")
P2 = sp.Symbol("P2")
drho, dp, Q, Sig, Rm = sp.symbols("drho dp Q Sig R")
subs_bg = {}


def to_sym(expr):
    """Replace functions of tau by symbols (a'/a = H etc.)"""
    expr = expr.subs({sp.Derivative(PhiP, (tau, 2)): P2, sp.Derivative(dF, (tau, 2)): D2})
    expr = expr.subs({sp.Derivative(PhiP, tau): P1, sp.Derivative(Psi, tau): Ps1, sp.Derivative(dF, tau): D1})
    expr = expr.subs({sp.Derivative(F, (tau, 3)): F3, sp.Derivative(F, (tau, 2)): F2, sp.Derivative(F, tau): F1})
    expr = expr.subs({PhiP: P, Psi: Ps, dF: D})
    asym = sp.Symbol("asym")
    expr = expr.subs({sp.Derivative(a, (tau, 2)): asym * (Hd + H**2), sp.Derivative(a, tau): asym * H}).subs(a, asym)
    return sp.simplify(expr), asym


E00, asym = to_sym(lin(E[0][0]) / e)
E0x, _ = to_sym(lin(E[0][1]) / e)
EJ, _ = to_sym(lin(divJ) / e)
A = asym
# 00:  E00 = -drho / a^2
sol_P1 = sp.solve(sp.Eq(E00, -drho / A**2), P1)[0]
# 0i: E0x = (rho+p) v_x = -(rho+p) i theta / k  ->  E0x = -i k Q / a^2
eq0i = sp.Eq(E0x, -sp.I * k * Q / A**2)
# d/dtau of the 0i equation: differentiate with respect to all symbols (chain rule)
dt = {A: A * H, H: Hd, F1: F2, F2: F3, P: P1, P1: P2, Ps: Ps1, D: D1, D1: D2, Q: sp.Symbol("Qd")}
def ddt(expr):
    return sum(sp.diff(expr, s) * v for s, v in dt.items())
Hdd = sp.Symbol("Hdd")
dt[Hd] = Hdd
eq0i_d = sp.Eq(ddt(eq0i.lhs), ddt(eq0i.rhs))
Qd = -2 * H * Q + dp - sp.Rational(2, 3) * Sig + Rm * Ps
eq0i_d = eq0i_d.subs(sp.Symbol("Qd"), Qd)
sol_P2 = sp.solve(eq0i_d, P2)[0]
# field equation with u: D1 = U + F1 Psi, D2 = U1 + F2 Psi + F1 Psi1
eqJ = EJ.subs(P2, sol_P2).subs(D2, U1 + F2 * Ps + F1 * Ps1).subs(D1, U + F1 * Ps)
eqJ = sp.expand(eqJ)
print("  Psi' remaining in the field equation:", sp.simplify(eqJ.coeff(Ps1)))
sol_U1 = sp.solve(sp.Eq(eqJ, 0), U1)[0]
print("  Phi_P' (00) =", sp.simplify(sol_P1.subs(D1, U + F1 * Ps)))
print("  u'         =", sp.simplify(sol_U1))
print("\nC code:")
print("  P1 =", sp.ccode(sp.simplify(sol_P1.subs(D1, U + F1 * Ps))), ";")
print("  U1 =", sp.ccode(sp.simplify(sol_U1)), ";")
print("  0i residual (should vanish):", sp.ccode(sp.simplify((eq0i.lhs - eq0i.rhs).subs(D1, U + F1 * Ps) / (sp.I * k))), ";")
