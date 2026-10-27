#!/usr/bin/env python3
"""Derive the Newtonian-gauge linear equations of Horndeski gravity (with alpha_T = 0) in terms of the
Bellini-Sawicki functions alpha_K, alpha_B, alpha_M and M*^2 (used in src/models/HorndeskiAlpha.cpp).
Run: python3 python/derive_horndeski.py   (needs sympy; a few minutes)

Family (kappa = 1, mostly-plus signature), general enough for independent alpha_K, alpha_B, alpha_M:
    S = int sqrt(-g) [ F(phi) R/2 + G2(phi, X) - G3(phi, X) box phi ] + S_m,   X = -(1/2)(d phi)^2
Field equations:  F G_mn - (nabla_m nabla_n F - g_mn box F) = T_mn(matter) + T_mn(phi),
    T(phi) = G2_X phi_m phi_n + G2 g_mn - G3_X box phi phi_m phi_n - (nabla_m G3 phi_n + nabla_n G3 phi_m)
             + g_mn nabla G3 . nabla phi
and, from the Bianchi identity, the scalar equation  E = (nabla^m T_mn(phi))/phi_n + (R/2) F_phi = 0.
Check: the linear spatial component of nabla^m T_mn must equal E_T(background) d_x(delta phi) (it is
proportional to phi_n for the correct stress tensor).

Alphas (Bellini & Sawicki 2014, with M*^2 = F, cosmic time):
    alpha_M = Fdot/(H F),   H F alpha_B = 2 phidot (X G3_X - F_phi/2),
    H^2 F alpha_K = 2 X (G2_X + 2 X G2_XX - 2 G3_phi - 2 X G3_phiX) + 12 phidot X H (G3_X + X G3_XX)
The G functions are eliminated in favour of the alphas, their derivatives and the background (H, H', rho_m, p_m).
The equations are then written for v = -delta phi/phi' (conformal time), and reduced to a first-order system.
"""
import sympy as sp

tau, x, y, z, k, eps = sp.symbols("tau x y z k epsilon")
a = sp.Function("a")(tau)
p0 = sp.Function("p0")(tau)            # background field
Psi, PhiP, dp = (sp.Function(n)(tau) for n in ("Psi", "PhiP", "dphi"))
e = sp.exp(sp.I * k * x)
X4 = [tau, x, y, z]
g = sp.diag(-a**2 * (1 + 2 * eps * Psi * e), *(3 * [a**2 * (1 - 2 * eps * PhiP * e)]))
ginv = sp.diag(*[1 / g[i, i] for i in range(4)])
phi = p0 + eps * dp * e


def s1(expr):
    expr = sp.expand(expr)
    return sp.expand(expr.subs(eps, 0) + eps * sp.diff(expr, eps).subs(eps, 0))


# Background values of G and its partial derivatives along the trajectory: name_i_j(tau) = d^i_phi d^j_X G
_fns = {}
def gfun(base, i, j):
    key = (base, i, j)
    if key not in _fns:
        _fns[key] = sp.Function(f"{base}_{i}_{j}")(tau)
    return _fns[key]

X0 = p0.diff(tau)**2 / (2 * a**2)
dX0 = sp.diff(X0, tau)

def chain(expr):
    """d/dtau with the chain rule for the background G functions"""
    d = sp.diff(expr, tau)
    rep = {}
    for (base, i, j), f in list(_fns.items()):
        rep[sp.Derivative(f, tau)] = gfun(base, i + 1, j) * p0.diff(tau) + gfun(base, i, j + 1) * dX0
    return d.subs(rep)

def Gval(base, i, j, Xfull):
    """G derivative (i, j) at the perturbed point, to first order"""
    dphi_full = eps * dp * e
    dX = Xfull - X0
    return gfun(base, i, j) + gfun(base, i + 1, j) * dphi_full + gfun(base, i, j + 1) * dX


def deriv(expr, i):
    """partial derivative along coordinate i, with the chain rule in tau"""
    if i == 0:
        return s1(chain(expr))
    return s1(sp.diff(expr, X4[i]))


Gam = [[[s1(sum(ginv[l, m] * (sp.diff(g[m, i], X4[j]) + sp.diff(g[m, j], X4[i]) - sp.diff(g[i, j], X4[m]))
                for m in range(4)) / 2) for j in range(4)] for i in range(4)] for l in range(4)]
ph = [sp.diff(phi, X4[i]) for i in range(4)]
nn = [[s1(sp.diff(phi, X4[i], X4[j]) - sum(Gam[l][i][j] * ph[l] for l in range(4))) for j in range(4)] for i in range(4)]
box = s1(sum(ginv[i, i] * nn[i][i] for i in range(4)))
X = s1(-sum(ginv[i, i] * ph[i]**2 for i in range(4)) / 2)

G2, G2X = Gval("G2", 0, 0, X), Gval("G2", 0, 1, X)
G3, G3X = Gval("G3", 0, 0, X), Gval("G3", 0, 1, X)
Fv, Fp, Fpp = Gval("F", 0, 0, X), Gval("F", 1, 0, X), Gval("F", 2, 0, X)
# F depends on phi only
for j in range(1, 6):
    for i in range(0, 6):
        _fns[("F", i, j)] = sp.Integer(0)

dG3 = [deriv(G3, i) for i in range(4)]
G3phi = s1(sum(ginv[i, i] * dG3[i] * ph[i] for i in range(4)))
T = [[s1(G2X * ph[i] * ph[j] + G2 * g[i, j] - G3X * box * ph[i] * ph[j] - (dG3[i] * ph[j] + dG3[j] * ph[i]) + g[i, j] * G3phi)
      for j in range(4)] for i in range(4)]

# divergence nabla^m T_mn
def divT(n):
    return s1(sum(ginv[m, m] * (deriv(T[m][n], m) - sum(Gam[l][m][m] * T[l][n] + Gam[l][m][n] * T[m][l] for l in range(4)))
                  for m in range(4)))

dT0, dTx = divT(0), divT(1)
ET_bg = sp.simplify(dT0.subs(eps, 0) / p0.diff(tau))
check = sp.simplify(sp.expand(sp.diff(dTx, eps).subs(eps, 0) / e - ET_bg * sp.I * k * dp))
print("Check: linear nabla^m T_mx = E_T(bg) d_x delta phi:", check == 0)
if check != 0:
    raise SystemExit("stress tensor inconsistent")

# Einstein tensor, Ricci scalar
def ricci(i, j):
    r = 0
    for l in range(4):
        r += sp.diff(Gam[l][i][j], X4[l]) - sp.diff(Gam[l][i][l], X4[j])
        for m in range(4):
            r += Gam[l][l][m] * Gam[m][i][j] - Gam[l][j][m] * Gam[m][i][l]
    return s1(r)

R = [[ricci(i, j) for j in range(4)] for i in range(4)]
Rs = s1(sum(ginv[i, i] * R[i][i] for i in range(4)))
Gmn = [[s1(R[i][j] - g[i, j] * Rs / 2) for j in range(4)] for i in range(4)]
nnF = [[s1(deriv(deriv(Fv, j), i) - sum(Gam[l][i][j] * deriv(Fv, l) for l in range(4))) for j in range(4)] for i in range(4)]
boxF = s1(sum(ginv[i, i] * nnF[i][i] for i in range(4)))
# gravity side minus scalar stress, mixed components: E^m_n = (matter T)^m_n
E = [[s1(ginv[i, i] * (Fv * Gmn[i][j] - (nnF[i][j] - g[i, j] * boxF) - T[i][j])) for j in range(4)] for i in range(4)]
# scalar equation: E_T + (R/2) F_phi = 0, E_T from the time component of the divergence
ET_lin = (sp.diff(dT0, eps).subs(eps, 0) / e - ET_bg * dp.diff(tau)) / p0.diff(tau)
Escal_bg = sp.simplify(ET_bg + (Rs.subs(eps, 0) / 2) * gfun("F", 1, 0))
Escal_lin = sp.expand(ET_lin + sp.diff(s1(Rs * Fp / 2), eps).subs(eps, 0) / e)

def lin(expr):
    return sp.expand(sp.diff(expr, eps).subs(eps, 0) / e)

eq = {
    "00": lin(E[0][0]),                       # = -delta rho
    "0x": lin(E[0][1]),                       # = (rho+p) v_x = -i (rho+p) theta / k
    "tl": lin(E[1][1]) - lin(E[2][2]),        # = -(3/2)(rho+p) sigma
    "S": Escal_lin,                           # = 0
}
bg = {"00": sp.simplify(E[0][0].subs(eps, 0)), "xx": sp.simplify(E[1][1].subs(eps, 0)), "S": Escal_bg}
print("Background and linear equations derived; eliminating the G functions ...")

import pickle
with open("/tmp/horndeski_eqs.pkl", "wb") as f:
    pickle.dump({"eq": eq, "bg": bg, "fns": list(_fns.keys())}, f)
print({kk: len(str(v)) for kk, v in eq.items()})
