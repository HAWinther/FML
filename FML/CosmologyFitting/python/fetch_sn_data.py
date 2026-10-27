#!/usr/bin/env python3
"""Download the large supernova data sets into external/sn/ (not in git; Union3 is included in the repository).

Usage:  python3 python/fetch_sn_data.py [pantheonplus] [desy5]      (default: both)

Files from the Cobaya SN data repository (https://github.com/CobayaSampler/sn_data), which redistributes:
  PantheonPlus/  Pantheon+SH0ES.dat, Pantheon+SH0ES_STAT+SYS.cov   (Brout et al. 2022, arXiv:2202.04077; 33 MB)
  DESY5/         DES-SN5YR_HD.csv, covsys_000.txt                  (DES Collaboration 2024, arXiv:2401.02929; 54 MB)
Run from the project root. Existing files are kept (delete them to download again).
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/CobayaSampler/sn_data/master/"
SETS = {
    "pantheonplus": ("PantheonPlus", ["Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"]),
    "desy5": ("DESY5", ["DES-SN5YR_HD.csv", "covsys_000.txt"]),
}

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
