# Plot cells written by AdaptiveMPIGrid::write_ascii (2D): python3 plot_cells.py cells out.png
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

prefix = sys.argv[1] if len(sys.argv) > 1 else "cells"
out = sys.argv[2] if len(sys.argv) > 2 else "cells.png"
d = np.vstack([np.loadtxt(f, ndmin=2) for f in sorted(glob.glob(prefix + ".*")) if not f.endswith(".png")])
x, y, h, lev, task = d[:, 0], d[:, 1], d[:, 2], d[:, 3].astype(int), d[:, 4].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
for ax, val, cmap, title in [(axes[0], task, "tab10", "MPI task (Hilbert segments)"), (axes[1], lev, "viridis", "Refinement level")]:
    rects = [Rectangle((a, b), s, s) for a, b, s in zip(x, y, h)]
    if cmap == "tab10":
        cmap = ListedColormap(plt.get_cmap("tab20").colors[: val.max() + 1])
    pc = PatchCollection(rects, cmap=cmap, edgecolor="k", linewidth=0.15)
    pc.set_array(val)
    pc.set_clim(val.min() - 0.5, val.max() + 0.5)
    ax.add_collection(pc)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_title(title)
    fig.colorbar(pc, ax=ax, fraction=0.046, ticks=np.unique(val))
fig.tight_layout()
fig.savefig(out, dpi=150)
