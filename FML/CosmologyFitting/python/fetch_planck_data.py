#!/usr/bin/env python3
"""Download the Planck likelihood data that is too large for git into external/ (run from CosmologyFitting/).

Usage:  python3 python/fetch_planck_data.py [camspec] [lowT] [lowE]      (default: all)

From the release v1 of the Cobaya native Planck data (https://github.com/CobayaSampler/planck_native_data):
  planck_npipe_camspec/  CamSpec PR4 (NPIPE) 12.6cl, Rosenberg, Gratton & Efstathiou 2022 (arXiv:2205.10869);
                         CamSpec_NPIPE.zip, 85 MB (the covariance unpacks to 510 MB)
  planck2018_lowT/       Planck 2018 low-ell TT (Commander, Gibbs), as Cobaya's planck_2018_lowl.TT; 5.5 MB
  planck2018_lowE/       Planck 2018 low-ell EE (SimAll), as Cobaya's planck_2018_lowl.EE; 0.8 MB
Existing folders are kept (delete them to download again).
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile

BASE = "https://github.com/CobayaSampler/planck_native_data/releases/download/v1/"
SETS = {
    "camspec": ("CamSpec_NPIPE.zip", "planck_npipe_camspec"),
    "lowt": ("planck_2018_lowT.zip", "planck2018_lowT"),
    "lowe": ("planck_2018_lowE.zip", "planck2018_lowE"),
}

wanted = [a.lower() for a in sys.argv[1:]] or list(SETS)
for name in wanted:
    if name not in SETS:
        sys.exit(f"unknown data set {name} (available: camspec, lowT, lowE)")
    asset, folder = SETS[name]
    out = os.path.join("external", folder)
    if os.path.isdir(out):
        print(f"{out} exists")
        continue
    print(f"downloading {BASE + asset} ...", flush=True)
    with urllib.request.urlopen(BASE + asset) as r:
        data = r.read()
    tmp = out + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            # flatten: the zips have the files in one folder (CamSpec_NPIPE/, planck_2018_lowT/, ...)
            target = os.path.join(tmp, os.path.basename(m.filename))
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    os.replace(tmp, out)
    print(f"  -> {out}: {', '.join(sorted(os.listdir(out)))}")
