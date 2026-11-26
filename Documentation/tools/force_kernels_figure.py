#!/usr/bin/env python3
"""The PM force kernels of FML::NBODY (NBody/NBody.h, ForceOptions.h) along a grid axis.

Writes figures/force_kernels.pdf (the book) and website/images/force_kernels.png (the website) and prints the values
quoted in Chapter 22 and on the N-body page. Run from Documentation/: python3 tools/force_kernels_figure.py
x = k dx is the wavenumber in units of the cell (x = pi at the grid Nyquist frequency); the per-direction kernels are
in units of 1/dx (gradient) and 1/dx^2 (Laplacian) and are exactly the expressions in NBody.h.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})

# Gradient kernels D(k)/i (the 6th order is the finite-difference force only)
grad = {
    "continuous": lambda x: x,
    "discrete_2pt": lambda x: np.sin(x),
    "discrete_4pt": lambda x: (8 * np.sin(x) - np.sin(2 * x)) / 6,
    "6th order (FD)": lambda x: (45 * np.sin(x) - 9 * np.sin(2 * x) + np.sin(3 * x)) / 30,
}
# Minus the Laplacian -D^2(k) per direction (the Green's function is G = -1 / sum_j lap(k_j))
lap = {
    "continuous": lambda x: x**2,
    "discrete_2pt": lambda x: 2 * (1 - np.cos(x)),
    "discrete_4pt": lambda x: (15 - 16 * np.cos(x) + np.cos(2 * x)) / 6,
    "old discrete_4pt (D4^2)": lambda x: ((8 * np.sin(x) - np.sin(2 * x)) / 6) ** 2,
}


def force_ratio(g, l, x, direction):
    """F_PM / F_exact for a mode with |k| dx = x along the unit vector direction (no window)."""
    n = np.asarray(direction, float)
    n /= np.linalg.norm(n)
    xs = np.outer(x, n)  # components
    L = sum(lap[l](xs[:, j]) for j in range(3))
    Dpar = sum(grad[g](xs[:, j]) * n[j] for j in range(3))  # D . khat
    return Dpar * x / L


if __name__ == "__main__":
    x = np.linspace(1e-4, np.pi, 600)
    xq = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    print("x/pi:", xq / np.pi)
    for name, f in grad.items():
        print(f"gradient {name:16s} D/k    :", np.round(f(xq) / xq, 4))
    for name, f in lap.items():
        print(f"laplace  {name:24s} L/k^2:", np.round(f(xq) / xq**2, 4))
    combos = [
        ("continuous", "continuous"),
        ("discrete_2pt", "continuous"),
        ("discrete_4pt", "continuous"),
        ("discrete_2pt", "discrete_2pt"),
        ("discrete_4pt", "discrete_2pt"),
        ("discrete_4pt", "discrete_4pt"),
        ("continuous", "discrete_2pt"),
    ]
    for d in [(1, 0, 0), (1, 1, 1)]:
        for g, l in combos:
            print(f"F/F_exact dir {d} grad {g:13s} lap {l:13s}:", np.round(force_ratio(g, l, xq[:3], d), 4))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    c = {"continuous": "k", "discrete_2pt": "C0", "discrete_4pt": "C1", "6th order (FD)": "C2",
         "old discrete_4pt (D4^2)": "C3"}
    for name, f in grad.items():
        ax[0].plot(x / np.pi, f(x) / x, color=c[name], label=name)
    ax[0].set_title(r"Gradient: $D(k)/k$")
    for name, f in lap.items():
        ls = ":" if name.startswith("old") else "-"
        ax[1].plot(x / np.pi, f(x) / x**2, color=c[name], ls=ls, label=name)
    ax[1].set_title(r"Laplacian: $-D^2(k)/k^2 = -1/(k^2 G)$")
    for (g, l), col in zip(combos[:6], ["k", "C0", "C1", "C2", "C4", "C5"]):
        ax[2].plot(x / np.pi, force_ratio(g, l, x, (1, 0, 0)), color=col, label=f"$D$ {g[9:] if g.startswith('discrete') else g}, $G$ {l[9:] if l.startswith('discrete') else l}")
    ax[2].set_title(r"Force along an axis: $F_{\rm PM}/F_{\rm exact}$")
    for a in ax:
        a.set_xlabel(r"$k/k_{\rm Ny}$ ($k_{\rm Ny}=\pi/\Delta x$)")
        a.set_xlim(0, 1)
        a.axhline(1, color="0.7", lw=0.6, zorder=0)
        a.legend(fontsize=10, frameon=False, loc="lower left")
    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 1.25)
    ax[2].set_ylim(0, 1.2)
    fig.tight_layout()
    fig.savefig("figures/force_kernels.pdf")
    fig.savefig("website/images/force_kernels.png", dpi=130)
