#!/usr/bin/env python3
#==========================================================================
# Compare the outputs of two regression runs (made with run.sh)
#
# Usage: python3 compare.py <reference_dir> <new_dir> [--rtol 1e-6] [--atol 1e-12] [--exact]
#   Text files are compared number by number with the given tolerances
#   (--exact: they must be identical). Binary files (Gadget/FML particles)
#   must be byte identical with --exact, otherwise they are skipped.
#   Log files and the parameter files are not compared.
#==========================================================================
import argparse
import math
import os
import re
import sys

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)", re.I)
SKIP = {"log.txt", "parameterfile.lua"}


def is_text(path):
    with open(path, "rb") as f:
        chunk = f.read(4096)
    return b"\0" not in chunk


def numbers(path):
    with open(path, errors="replace") as f:
        return [float(x) for x in NUM.findall(f.read())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    nfail = 0
    ncompared = 0
    worst = {}
    for root, _, files in os.walk(args.ref):
        for fn in sorted(files):
            if fn in SKIP:
                continue
            ref = os.path.join(root, fn)
            rel = os.path.relpath(ref, args.ref)
            new = os.path.join(args.new, rel)
            case = rel.split(os.sep)[0]
            if not os.path.exists(new):
                print(f"MISSING  {rel}")
                nfail += 1
                continue
            ncompared += 1
            if not is_text(ref):
                if args.exact and open(ref, "rb").read() != open(new, "rb").read():
                    print(f"DIFFERS  {rel} (binary)")
                    nfail += 1
                continue
            a, b = numbers(ref), numbers(new)
            if len(a) != len(b):
                print(f"DIFFERS  {rel} (number of values {len(a)} vs {len(b)})")
                nfail += 1
                continue
            maxrel = 0.0
            bad = False
            for x, y in zip(a, b):
                if math.isnan(x) and math.isnan(y):
                    continue
                if args.exact:
                    bad = bad or x != y
                elif abs(x - y) > args.atol + args.rtol * abs(x):
                    bad = True
                if x != y:
                    maxrel = max(maxrel, abs(x - y) / max(abs(x), 1e-300))
            worst[case] = max(worst.get(case, 0.0), maxrel)
            if bad:
                print(f"DIFFERS  {rel} (max relative difference {maxrel:.2e})")
                nfail += 1
    for root, _, files in os.walk(args.new):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), args.new)
            if fn not in SKIP and not os.path.exists(os.path.join(args.ref, rel)):
                print(f"NEW FILE {rel}")
    if not args.quiet:
        for case, w in sorted(worst.items()):
            print(f"{case:28s} max relative difference {w:.2e}")
    print(f"Compared {ncompared} files: {'OK' if nfail == 0 else str(nfail) + ' differ'}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
