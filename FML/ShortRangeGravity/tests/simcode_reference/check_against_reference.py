#!/usr/bin/env python3
"""Compare the P(k) of a new run from ic_z19.000.gadget with the frozen SimulationCode TreePM reference (treepm_1024).

Usage: python3 check_against_reference.py pofk_z1.000.txt pofk_z0.000.txt [--tol 0.02] [--kmax 1.0]
The files are P(k) of the cb particles in the format of pofk/ here (columns k [h/Mpc], P(k) [(Mpc/h)^3]; '#' comments),
measured as SimulationCode does: 64^3 PCS grid, interlacing, no shot-noise subtraction. Exit status 1 if
|P / P_ref - 1| > tol for any k <= kmax at either redshift. The default tolerance (2%) allows for different time
stepping and tree opening: the reference itself is converged in time to about 0.6% for k <= 1 h/Mpc (summary.json).
The reference does not deconvolve the CIC window of the force (README.md): run the new code the same way (TreePMCode:
deconvolve = false); with the window deconvolved P(k) is about 7% higher at k = 1 h/Mpc.
"""
import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent


def read(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            rows.append((float(f[0]), float(f[1])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pofk_z1")
    parser.add_argument("pofk_z0")
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--kmax", type=float, default=1.0)
    args = parser.parse_args()
    ok = True
    for z, path in [("1.000", args.pofk_z1), ("0.000", args.pofk_z0)]:
        ref, new = read(here / "pofk" / f"treepm_1024_z{z}.txt"), read(path)
        if len(ref) != len(new) or any(abs(a[0] / b[0] - 1) > 1e-4 for a, b in zip(ref, new)):
            sys.exit(f"z = {z}: the k bins differ from the reference (use a 64^3 grid for P(k))")
        worst = max((abs(n[1] / r[1] - 1), r[0]) for r, n in zip(ref, new) if r[0] <= args.kmax)
        status = "PASS" if worst[0] <= args.tol else "FAIL"
        ok &= status == "PASS"
        print(f"[{status}] z = {z}: max |P/P_ref - 1| = {worst[0]:.4f} at k = {worst[1]:.3f} h/Mpc "
              f"(k <= {args.kmax}, tolerance {args.tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
