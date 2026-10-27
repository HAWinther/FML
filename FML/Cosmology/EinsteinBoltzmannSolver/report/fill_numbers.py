#!/usr/bin/env python3
"""Replace @@key@@ placeholders in report/sections/*.tex with values from report/data/*_numbers.txt
(fractional differences as percent or a x 10^-n, ratios with three decimals). Run after the make_*_figure.py
scripts; the placeholders are replaced in place."""
import glob
import math
import re

num = {}
for f in glob.glob("report/data/*_numbers.txt"):
    for line in open(f):
        p = line.split()
        if len(p) == 2:
            try:
                num[p[0]] = float(p[1])
            except ValueError:
                pass


def fmt(key):
    if key.endswith("_pct"):
        return f"{abs(1 - num[key[:-4]]) * 100:.0f}"
    v = num[key]
    if "zstar" in key:
        return f"{v:.2f}"
    if "rs_star" in key:
        return f"{v:.3f}"
    if "ratio" in key:
        return f"{v:.3f}"
    if v < 1e-3:
        e = math.floor(math.log10(v))
        return f"${v / 10**e:.1f}\\times10^{{{e}}}$"
    return f"{v * 100:.2g}\\%"


for f in glob.glob("report/sections/*.tex") + ["report/paper.tex"]:
    s = open(f).read()
    t = re.sub(r"@@([A-Za-z0-9_.]+)@@", lambda m: fmt(m.group(1)), s)
    if t != s:
        open(f, "w").write(t)
        print("filled", f)
left = [m for f in glob.glob("report/sections/*.tex") + ["report/paper.tex"] for m in re.findall(r"@@[^@]+@@", open(f).read())]
print("unfilled:", left)
