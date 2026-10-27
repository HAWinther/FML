#!/usr/bin/env python3
"""Derive the linear perturbation equations of Brans-Dicke gravity in Newtonian gauge (used in
src/models/BransDicke.cpp). Run: python3 python/derive_brans_dicke.py   (needs sympy)

Jordan frame, field equations (kappa = 8 pi G*):
    phi G_mn = kappa T_mn + (omega/phi)(d_m phi d_n phi - g_mn (dphi)^2 / 2) + nabla_m nabla_n phi - g_mn box phi
    box phi  = kappa T / (3 + 2 omega)
Metric (conformal time tau, Planck/Ma-Bertschinger signs; Phi_P = -Phi of the code):
    ds^2 = a^2 [ -(1 + 2 Psi) dtau^2 + (1 - 2 Phi_P) dx^2 ],   phi = F(tau) + dF(tau) e^{ikx}
Matter: delta T^0_0 = -delta rho, delta T^0_x = (rho + p) v_x with theta = i k v_x, delta T^i_j = delta p delta_ij +
Sigma^i_j with Sigma^x_x - Sigma^y_y = -(3/2)(rho + p) sigma (k along x).
Prints the gravity side of each equation, divided by the plane wave, as LHS = kappa * (matter).
"""
import sympy as sp

tau, x, y, z, k, om, eps = sp.symbols("tau x y z k omega epsilon")
a = sp.Function("a")(tau)
F = sp.Function("F")(tau)
Psi, PhiP, dF = (sp.Function(n)(tau) for n in ("Psi", "PhiP", "dF"))
e = sp.exp(sp.I * k * x)
X = [tau, x, y, z]

g = sp.diag(-a**2 * (1 + 2 * eps * Psi * e), *(3 * [a**2 * (1 - 2 * eps * PhiP * e)]))
ginv = sp.diag(*[1 / g[i, i] for i in range(4)])
phi = F + eps * dF * e


def lin(expr):
    """First-order part in eps"""
    return sp.simplify(sp.diff(expr, eps).subs(eps, 0))


def bg(expr):
    return sp.simplify(expr.subs(eps, 0))


# Christoffel symbols, Ricci tensor, Einstein tensor (series in eps to first order is enough)
def series1(expr):
    return expr.subs(eps, 0) + eps * sp.diff(expr, eps).subs(eps, 0)


Gam = [[[series1(sum(ginv[l, m] * (sp.diff(g[m, i], X[j]) + sp.diff(g[m, j], X[i]) - sp.diff(g[i, j], X[m]))
                     for m in range(4)) / 2) for j in range(4)] for i in range(4)] for l in range(4)]


def ricci(i, j):
    r = 0
    for l in range(4):
        r += sp.diff(Gam[l][i][j], X[l]) - sp.diff(Gam[l][i][l], X[j])
        for m in range(4):
            r += Gam[l][l][m] * Gam[m][i][j] - Gam[l][j][m] * Gam[m][i][l]
    return series1(sp.expand(r))


R = [[ricci(i, j) for j in range(4)] for i in range(4)]
Rs = series1(sp.expand(sum(ginv[i, i] * R[i][i] for i in range(4))))
G = [[series1(R[i][j] - g[i, j] * Rs / 2) for j in range(4)] for i in range(4)]

# Scalar field terms
dphi = [sp.diff(phi, X[i]) for i in range(4)]
nn = [[series1(sp.diff(phi, X[i], X[j]) - sum(Gam[l][i][j] * dphi[l] for l in range(4))) for j in range(4)]
      for i in range(4)]
box = series1(sp.expand(sum(ginv[i, i] * nn[i][i] for i in range(4))))
kin = series1(sp.expand(sum(ginv[i, i] * dphi[i] ** 2 for i in range(4))))

# Gravity side with lower indices, then mixed E^m_n = g^{mm} E_mn
E = [[series1(phi * G[i][j] - (om / phi) * (dphi[i] * dphi[j] - g[i, j] * kin / 2) - nn[i][j] + g[i, j] * box)
      for j in range(4)] for i in range(4)]
Emix = [[series1(sp.expand(ginv[i, i] * E[i][j])) for j in range(4)] for i in range(4)]

print("Background (kappa rho = -E^0_0, kappa p = E^x_x):")
print("  E^0_0 =", sp.simplify(bg(Emix[0][0])))
print("  E^x_x =", sp.simplify(bg(Emix[1][1])))
print("  box phi =", sp.simplify(bg(box)))
print()
print("Perturbations (divided by e^{ikx}):")
print("  delta E^0_0  [= -kappa delta rho]                 =", sp.simplify(lin(Emix[0][0]) / e))
print("  delta E^0_x  [= kappa (rho+p) v_x]                =", sp.simplify(lin(Emix[0][1]) / e))
print("  delta E^x_x - delta E^y_y [= -(3/2) kappa (rho+p) sigma] =", sp.simplify((lin(Emix[1][1]) - lin(Emix[2][2])) / e))
print("  delta box phi [= kappa (-delta rho + 3 delta p)/(3 + 2 omega)] =", sp.simplify(lin(box) / e))
