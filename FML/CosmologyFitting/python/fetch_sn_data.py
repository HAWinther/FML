#!/usr/bin/env python3
"""Download the large supernova data sets into external/sn/ (not in git; Union3 is included in the repository).

Usage:  python3 python/fetch_sn_data.py [pantheonplus] [desy5] [desdovekie]      (default: all)

Files from the Cobaya SN data repository (https://github.com/CobayaSampler/sn_data), which redistributes:
  PantheonPlus/  Pantheon+SH0ES.dat, Pantheon+SH0ES_STAT+SYS.cov   (Brout et al. 2022, arXiv:2202.04077; 33 MB)
  DESY5/         DES-SN5YR_HD.csv, covsys_000.txt                  (DES Collaboration 2024, arXiv:2401.02929; 54 MB)
  DES-Dovekie/   DES-Dovekie_HD.csv, covtot_inv_000.npz            (DES-SN5YR recalibrated, Popovic et al. 2025,
                                                                    arXiv:2511.07517; 1820 SNe, inverse of the total
                                                                    covariance; converted here to covtot_inv_000.txt,
                                                                    which needs numpy)
Run from the project root. Existing files are kept (delete them to download again).
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/CobayaSampler/sn_data/master/"
SETS = {
    "pantheonplus": ("PantheonPlus", ["Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"]),
    "desy5": ("DESY5", ["DES-SN5YR_HD.csv", "covsys_000.txt"]),
    "desdovekie": ("DES-Dovekie", ["DES-Dovekie_HD.csv", "covtot_inv_000.npz"]),
}


def convert_npz_inverse_covariance(npz, txt):
    """Cobaya's packed inverse covariance (upper triangle, row-major) -> text: n, then the n(n+1)/2 values."""
    import numpy as np

    d = np.load(npz)
    n = int(d[d.files[0]][0])
    v = d[d.files[1]]
    if v.size != n * (n + 1) // 2:
        sys.exit(f"{npz}: expected {n * (n + 1) // 2} values, got {v.size}")
    with open(txt + ".part", "w") as f:
        f.write(f"{n}\n")
        np.savetxt(f, v, fmt="%.17g")
    os.replace(txt + ".part", txt)
    print(f"  -> {txt} ({os.path.getsize(txt) / 1e6:.1f} MB)")


wanted = [a.lower() for a in sys.argv[1:]] or list(SETS)
for name in wanted:
    if name not in SETS:
        sys.exit(f"unknown data set {name} (available: {', '.join(SETS)})")
    folder, files = SETS[name]
    out = os.path.join("external", "sn", folder)
    os.makedirs(out, exist_ok=True)
    for f in files:
        path = os.path.join(out, f)
        if os.path.exists(path):
            print(f"{path} exists")
            continue
        print(f"downloading {BASE + folder + '/' + f} ...", flush=True)
        urllib.request.urlretrieve(BASE + folder + "/" + f, path + ".part")
        os.replace(path + ".part", path)
        print(f"  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    for f in files:
        path = os.path.join(out, f)
        if f.endswith(".npz") and not os.path.exists(path[:-4] + ".txt"):
            convert_npz_inverse_covariance(path, path[:-4] + ".txt")
