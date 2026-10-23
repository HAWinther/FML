#!/usr/bin/env python3
#==========================================================================
# Compare the P(k) from the code to a reference P(k) (from MG-PICOLA)
#
# Usage: python3 compare_pofk.py <reference> <ours> [--tol 0.005] [--kmax 0.5] [--verbose]
#
# Both files have k in column 0 and P(k) in column 1. The k-bins are the same
# (the reference has more bins as it goes further in k), but we output the mean k
# of the modes in each bin while the reference has the bin-center so we match the
# rows by index.
#==========================================================================
import argparse
import sys


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip() == "" or line.lstrip().startswith("#"):
                continue
            cols = line.split()
            rows.append((float(cols[0]), float(cols[1])))
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("ours")
ap.add_argument("--tol", type=float, default=0.002, help="max relative difference allowed")
ap.add_argument("--kmin", type=float, default=0.0, help="smallest k (h/Mpc) to compare")
ap.add_argument("--kmax", type=float, default=0.8, help="largest k (h/Mpc) to compare (the Nyquist frequency of the tests)")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

ref, ours = read(args.ref), read(args.ours)
n = min(len(ref), len(ours))
worst, worst_k, ncompared = 0.0, 0.0, 0
for i in range(n):
    k = ref[i][0]
    if k < args.kmin or k > args.kmax:
        continue
    rel = ours[i][1] / ref[i][1] - 1.0
    ncompared += 1
    if args.verbose:
        print(f"k = {k:8.4f}  P_ref = {ref[i][1]:12.4f}  P = {ours[i][1]:12.4f}  diff = {100 * rel:+7.3f}%")
    if abs(rel) > abs(worst):
        worst, worst_k = rel, k
ok = ncompared > 0 and abs(worst) <= args.tol
print(f"{'PASS' if ok else 'FAIL'}: max |P/P_ref - 1| = {100 * abs(worst):.3f}% at k = {worst_k:.3f} h/Mpc "
      f"({ncompared} bins with {args.kmin} <= k <= {args.kmax}, tolerance {100 * args.tol:.2f}%)")
sys.exit(0 if ok else 1)
