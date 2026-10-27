#!/usr/bin/env python3
"""Triangle plot and marginalized constraints of ./mcmc chains (needs getdist: pip install getdist).

Usage:  python3 python/plot_chains.py ROOT [ROOT2 ...] [--params p1,p2,...] [--out file.png]
ROOT is the mcmc_output of a run (chains ROOT_1.txt, ROOT_2.txt, ... and ROOT.paramnames). The first 0% is
discarded (the burn-in is not written). Prints the 68% limits of every parameter and writes the plot
(default ROOT.png).
"""
import sys

import matplotlib
matplotlib.use("Agg")
from getdist import loadMCSamples, plots  # noqa: E402

args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
params, out = None, None
if "--params" in args:
    i = args.index("--params")
    params = args[i + 1].split(",")
    del args[i:i + 2]
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    del args[i:i + 2]
samples = [loadMCSamples(r, settings={"ignore_rows": 0}) for r in args]
for r, s in zip(args, samples):
    print(f"{r}: {s.numrows} distinct samples, effective {s.getEffectiveSamples():.0f}")
    print(s.getTable(limit=1).tableTex())
g = plots.get_subplot_plotter(width_inch=6)
g.triangle_plot(samples, params, filled=True, legend_labels=[r.split("/")[-1] for r in args])
g.export(out or args[0] + ".png")
print("wrote", out or args[0] + ".png")
