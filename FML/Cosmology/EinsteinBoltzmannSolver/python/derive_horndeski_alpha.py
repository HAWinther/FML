#!/usr/bin/env python3
"""Second step of the Horndeski derivation (run python/derive_horndeski.py first): eliminate the G functions in
favour of alpha_K, alpha_B, alpha_M (and their conformal-time derivatives), M*^2 = F and the background
(a, H = a'/a, H', rho_m, p_m), write delta phi = -phi' v, and reduce the system to first order:
  * the scalar-field equation still contains higher derivatives of the G functions; these are removed with the
    conformal-time derivatives of the alpha_K definition and of the three background equations;
  * Phi_P'' is removed with the time derivative of the 0i equation and matter momentum conservation,
        Q' = -2 H Q + dp - (2/3) Sig + R Psi;
    Psi' then only appears in u = v' + Psi, which is the evolved variable;
  * the matter background enters only through R = a^2 (rho + p) and R' (so p'_m appears only via R').
Units kappa = 1, conformal time; matter: drho = a^2 delta rho, dp = a^2 delta p, Q = a^2 (rho+p) theta/k^2,
Sig = (3/2) a^2 (rho+p) sigma. Prints C expressions used in src/models/HorndeskiAlpha.cpp.
"""
import pickle

import sympy as sp

d = pickle.load(open("/tmp/horndeski_eqs.pkl", "rb"))
eq, bg = d["eq"], d["bg"]
tau, k = sp.symbols("tau k")
a_f = sp.Function("a")(tau)
p0 = sp.Function("p0")(tau)
Psi_f, PhiP_f, dp_f = (sp.Function(n)(tau) for n in ("Psi", "PhiP", "dphi"))

# --- symbols at one instant -------------------------------------------------
a, H, Hd, Hdd = sp.symbols("a H Hd Hdd", positive=True)
p1, p2, p3, p4 = sp.symbols("p1 p2 p3 p4")            # phi', phi'', phi''', phi''''
Psi, Psi1, Psi2, P, P1, P2, D, D1, D2, D3 = sp.symbols("Psi Psi1 Psi2 P P1 P2 D D1 D2 D3")
gs = {}
def G(base, i, j):
    if base == "F" and j > 0:
        return sp.Integer(0)
    key = (base, i, j)
    if key not in gs:
        gs[key] = sp.Symbol(f"{base}{i}{j}")
    return gs[key]

def to_sym(expr):
    expr = sp.expand(expr)
    rep = {}
    for f in expr.atoms(sp.Function):
        pass
    # derivatives of the perturbations and of the background, highest order first
    for (f, s0, s1_, s2_, s3_) in [(dp_f, D, D1, D2, D3), (PhiP_f, P, P1, P2, None), (Psi_f, Psi, Psi1, Psi2, None),
                                   (p0, None, p1, p2, p3)]:
        for n, s in [(3, s3_), (2, s2_), (1, s1_)]:
            if s is not None:
                expr = expr.subs(sp.Derivative(f, (tau, n)), s)
        if s0 is not None:
            expr = expr.subs(f, s0)
    expr = expr.subs(sp.Derivative(p0, (tau, 4)), p4)
    # G functions of tau and their tau-derivatives (none should remain un-chained, but replace values)
    for at in list(expr.atoms(sp.Function)):
        name = type(at).__name__
        if "_" in name and name.split("_")[0] in ("G2", "G3", "F"):
            base, i, j = name.split("_")
            expr = expr.subs(at, G(base, int(i), int(j)))
    expr = expr.subs({sp.Derivative(a_f, (tau, 3)): a * (Hdd + 3 * H * Hd + H**3),
                      sp.Derivative(a_f, (tau, 2)): a * (Hd + H**2),
                      sp.Derivative(a_f, tau): a * H}).subs(a_f, a)
    return sp.expand(expr)

E = {kk: to_sym(v) for kk, v in eq.items()}
B = {kk: to_sym(v) for kk, v in bg.items()}
left = set().union(*[v.atoms(sp.Function) for v in list(E.values()) + list(B.values())])
print("functions left after conversion:", left)

# --- alphas (conformal time) ---------------------------------------------------
Xs = p1**2 / (2 * a**2)
F, Fp = G("F", 0, 0), G("F", 1, 0)
aM = Fp * p1 / (H * F)
aB = 2 * p1 * (Xs * G("G3", 0, 1) - Fp / 2) / (H * F)
aK = (2 * Xs * (G("G2", 0, 1) + 2 * Xs * G("G2", 0, 2) - 2 * G("G3", 1, 0) - 2 * Xs * G("G3", 1, 1)) +
      12 * p1 * Xs * H / a**2 * (G("G3", 0, 1) + Xs * G("G3", 0, 2))) / ((H / a)**2 * F)

def ddt(expr):
    """conformal-time derivative of an expression in the instant symbols"""
    out = sp.diff(expr, a) * a * H + sp.diff(expr, H) * Hd + sp.diff(expr, Hd) * Hdd
    out += sp.diff(expr, p1) * p2 + sp.diff(expr, p2) * p3 + sp.diff(expr, p3) * p4
    dX = p1 * p2 / a**2 - p1**2 * H / a**2
    for (base, i, j), s in list(gs.items()):
        out += sp.diff(expr, s) * (G(base, i + 1, j) * p1 + G(base, i, j + 1) * dX)
    return sp.expand(out)

aMd, aBd, aKd = ddt(aM), ddt(aB), ddt(aK)

# unknowns to eliminate, solved from the alpha definitions and the background equations
AK, AB, AM, AKd, ABd, AMd, M2, rho, pm = sp.symbols("aK aB aM aKd aBd aMd M2 rho p")
defs = [sp.Eq(F, M2), sp.Eq(aM, AM), sp.Eq(aB, AB), sp.Eq(aK, AK), sp.Eq(aMd, AMd), sp.Eq(aBd, ABd),
        sp.Eq(B["00"], -rho), sp.Eq(B["xx"], pm), sp.Eq(B["S"], 0)]
unknowns = [F, Fp, G("G3", 0, 1), G("G2", 0, 2), G("F", 2, 0), G("G3", 0, 2), G("G2", 0, 0), G("G3", 1, 0), G("G2", 1, 0)]
sol = sp.solve(defs, unknowns, dict=True)
print("solutions:", len(sol))
sol = sol[0]

def reduce(expr):
    expr = sp.expand(expr.subs(sol))
    # delta phi = -phi' v (conformal time); D -> -p1 v etc.
    v, v1, v2, v3 = sp.symbols("v v1 v2 v3")
    expr = expr.subs({D3: -(p4 * v + 3 * p3 * v1 + 3 * p2 * v2 + p1 * v3), D2: -(p3 * v + 2 * p2 * v1 + p1 * v2),
                      D1: -(p2 * v + p1 * v1), D: -p1 * v})
    return sp.simplify(sp.expand(expr))


# --- 1. scalar equation: remove the remaining G derivatives ----------------------------------------------
rhod, pd, AKd = -3 * H * (rho + pm), sp.Symbol("pd"), sp.Symbol("aKd")
rels = [aKd - AKd, ddt(B["00"]) + rhod, ddt(B["xx"]) - pd, ddt(B["S"])]
rels = [sp.expand(r.subs(sol)) for r in rels]
S = sp.expand(sp.numer(sp.together(reduce(E["S"]))))
sy = {s.name: s for r in rels + [S] for s in r.free_symbols}
gsol = sp.solve(rels, [sy[n] for n in ("G311", "G320", "G330", "G303")], dict=True)[0]
S = sp.expand(sp.numer(sp.together(S.subs(gsol))))
core = [f for f, m in sp.factor_list(S)[1] if len(str(f)) > 100]
assert len(core) == 1, "unexpected factorisation"
core = core[0]
assert not [s for s in core.free_symbols if s.name[0] in "GF" or s.name in ("p1", "p2", "p3")], "G functions left"
print("scalar equation is free of the G functions")

# --- 2. remove Phi_P'' with d/dtau of the 0i equation, Psi' via u = v' + Psi ------------------------------
v, v1, v2 = sp.symbols("v v1 v2")
Q, dpm, Sig, drho, U, U1, R, Rd = sp.symbols("Q dp Sig drho U U1 R Rd")
X0 = sp.expand(reduce(E["0x"]) * a**4 / (sp.I * k)) + a**2 * Q     # = 0 (0i: E^0_x = -i k Q / a^2)
def dtau(e):
    rep = {a: a * H, H: Hd, Hd: Hdd, M2: M2 * AM * H, AB: ABd, v: v1, v1: v2, Psi: Psi1, P1: P2,
           Q: -2 * H * Q + dpm - sp.Rational(2, 3) * Sig + a**2 * (rho + pm) * Psi, rho: rhod, pm: pd}
    return sp.expand(sum(sp.diff(e, s) * d for s, d in rep.items()))
c = sp.expand(core.subs(P2, sp.solve(dtau(X0), P2)[0]))
c = sp.expand(c.subs(v2, U1 - Psi1))
assert c.coeff(Psi1) == 0 and c.coeff(Hdd) == 0
# matter background through R = a^2 (rho+p), R' = a^2 p' + H R  (rho' = -3 H (rho + p))
def matter(e):
    return sp.expand(e.subs(pd, (Rd + H * R) / a**2).subs(pm, R / a**2 - rho))
c, X0 = matter(c), matter(X0)
assert rho not in c.free_symbols and rho not in X0.free_symbols
E00 = matter(sp.expand(reduce(E["00"]) * a**2 + drho))                 # = 0 (E^0_0 = -delta rho)
Etl = sp.expand(reduce(E["tl"]) * a**2 + Sig)                          # = 0 (traceless: -(3/2)(rho+p) sigma)

Psi_sol = sp.solve(Etl, Psi)[0]
P1_sol = sp.solve(E00, P1)[0]
U1_sol = sp.solve(c, U1)[0]
X0_res = X0
print("\nPsi  =", Psi_sol)
print("\nP1   =", sp.factor(P1_sol))
num, den = sp.fraction(sp.together(U1_sol))
print("\nU1 denominator:", sp.factor(den))
print("\n0i residual (test):", sp.collect(X0, [P1, v1, v, Psi, Q]))

# C code. Hd is written with Rx = 2 M2 (H^2 - Hd) - R = (M2 - 1) R + M2 a^2 (rho+p)_DE (GR-sum background):
# all terms of the scalar equation are then proportional to an alpha or to Rx, so nothing cancels
# catastrophically at early times where Omega_DE -> 0.
Rx = sp.Symbol("Rx")
hd = {Hd: H**2 - (R + Rx) / (2 * M2)}
P1c = sp.expand(P1_sol.subs(v1, U - Psi).subs(hd))
num, den = sp.fraction(sp.together(sp.expand(U1_sol.subs(v1, U - Psi).subs(hd))))
num = sp.expand(num)
zero = {AK: 0, AB: 0, AM: 0, AKd: 0, ABd: 0, Rx: 0}
assert sp.expand(num.subs(zero)) == 0, "scalar equation not proportional to the alphas"
vars_ = [U, v, Psi, P, P1, Q, dpm, Sig, drho]
print("\n// C: P1 =", sp.ccode(sp.simplify(P1c)))
print("\n// C: U1 = (")
for var in vars_:
    cf = sp.factor(num.coeff(var))
    if cf != 0:
        print(f"//   + ({sp.ccode(cf)}) * {var}")
rest = sp.expand(num - sum(num.coeff(var) * var for var in vars_))
assert rest == 0
print(f"// ) / ({sp.ccode(sp.factor(den))})")
X0c = sp.expand(X0.subs(v1, U - Psi).subs(hd))
print("\n// C: 0i residual =", sp.ccode(sp.collect(X0c, [P1, U, v, Psi, Q])))
pickle.dump({"Psi": Psi_sol, "P1": P1_sol, "U1": U1_sol, "X0": X0}, open("/tmp/horndeski_alpha.pkl", "wb"))
