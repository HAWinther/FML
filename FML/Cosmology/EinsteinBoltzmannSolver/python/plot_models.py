#!/usr/bin/env python3
"""Compare the models beyond LCDM with LCDM: H(z), C_ell^TT, C_ell^phiphi and P(k), as ratios.

Usage:  python3 python/plot_models.py [BASEDIR]     (default data/models)
BASEDIR contains one output directory per model (lcdm, musigma, ndgp, sdgp, quintessence, jbd, fr,
coupled_quintessence, dark_scattering, cubic_galileon, horndeski, ede, ide; missing ones are
skipped), e.g. from
    for m in "lcdm" "musigma model=musigma mu0=0.3 Sigma0=0.1" ...; do ./cmb lensing=true output_dir=... ; done
Writes BASEDIR/models.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

base = sys.argv[1] if len(sys.argv) > 1 else "data/models"
labels = {"musigma": r"$\mu$-$\Sigma$ ($\mu_0=0.3$, $\Sigma_0=0.1$)",
          "ndgp": r"nDGP, $\Lambda$CDM background ($H_0 r_c/c = 1$)",
          "sdgp": "sDGP (no dark energy)",
          "quintessence": r"quintessence ($\lambda = 1$)",
          "jbd": r"Brans-Dicke ($\omega = 100$)",
          "fr": r"Hu-Sawicky $f(R)$ ($|f_{R0}| = 10^{-5}$, $n = 1$)",
          "coupled_quintessence": r"coupled quintessence ($\lambda = 0.5$, $\beta = 0.05$)",
          "dark_scattering": r"dark scattering ($w = -0.9$, $\xi = 10$ b/GeV)",
          "cubic_galileon": "cubic Galileon (tracker)",
          "horndeski": r"Horndeski $\propto\Omega_{DE}$ ($c_K = 1$, $c_B = 0.6$, $c_M = 0.3$)",
          "ede": r"early dark energy ($f_{EDE} = 0.12$, $\log_{10} z_c = 3.56$)",
          "ide": r"interacting dark energy ($\xi = -0.1$, $w = -0.95$)"}


def load(name):
    d = os.path.join(base, name)
    if not os.path.isfile(os.path.join(d, "cells_lensed.txt")):
        return None
    return {"bg": np.loadtxt(os.path.join(d, "background.txt")), "cl": np.loadtxt(os.path.join(d, "cells.txt")),
            "len": np.loadtxt(os.path.join(d, "cells_lensed.txt")), "pk": np.loadtxt(os.path.join(d, "matter_power.txt"))}


ref = load("lcdm")
if ref is None:
    sys.exit(f"no LCDM run in {base}/lcdm")
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for i, (name, lab) in enumerate(labels.items()):
    m = load(name)
    if m is None:
        continue
    st = {"color": f"C{i % 10}", "ls": "-" if i < 10 else "--"}   # dashed after the 10 default colours
    x = ref["bg"][:, 0]
    sel = (x > -np.log(1 + 1100)) & (x <= 0)
    z = np.exp(-x[sel]) - 1
    ax[0, 0].semilogx(1 + z, m["bg"][sel, 3] / ref["bg"][sel, 3], label=lab, **st)
    n = min(len(m["cl"]), len(ref["cl"]))
    ell = ref["cl"][:n, 0]
    ax[0, 1].semilogx(ell, m["cl"][:n, 1] / ref["cl"][:n, 1], label=lab, **st)
    n = min(len(m["len"]), len(ref["len"]))
    ax[1, 0].semilogx(ref["len"][:n, 0], m["len"][:n, 5] / ref["len"][:n, 5], label=lab, **st)
    # P(k) in h/Mpc units: compare at the same k in 1/Mpc (h is the same in all runs)
    ax[1, 1].semilogx(ref["pk"][:, 0], np.interp(ref["pk"][:, 0], m["pk"][:, 0], m["pk"][:, 1]) / ref["pk"][:, 1],
                      label=lab, **st)
ax[0, 0].set(xlabel="1 + z", ylabel=r"$H(z) / H_{\Lambda CDM}$", title="Expansion")
ax[0, 1].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{TT} / C_\ell^{TT,\Lambda CDM}$", title="CMB temperature (unlensed)")
ax[1, 0].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{\phi\phi} / C_\ell^{\phi\phi,\Lambda CDM}$", title="CMB lensing potential")
ax[1, 1].set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k) / P_{\Lambda CDM}(k)$", title="Matter power spectrum, z = 0")
for a in ax.flat:
    a.axhline(1.0, color="k", lw=0.5)
    a.grid(alpha=0.3)
ax[0, 1].legend(fontsize=8)
fig.tight_layout()
out = os.path.join(base, "models.png")
fig.savefig(out, dpi=130)
print("wrote", out)
