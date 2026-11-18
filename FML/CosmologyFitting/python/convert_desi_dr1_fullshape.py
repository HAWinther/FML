#!/usr/bin/env python3
"""Convert DESI's DR1 full-shape power-spectrum likelihood files for FML's desi_dr1_fs likelihood.

Usage (from CosmologyFitting/):
  python3 python/convert_desi_dr1_fullshape.py /path/to/data/likelihood
The folder holds the files of https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/data/likelihood/
(e.g. from python download.py in desi-kp-cosmological-likelihoods/dr1/cobaya). Needs numpy and h5py.

For each of BGS (0.1-0.4), LRG (0.4-0.6, 0.6-0.8, 0.8-1.1), ELG (1.1-1.6) and QSO (0.8-2.1) the fiducial likelihood
"spectrum-poles-rotated" with the systematics "rotation-hod-photo" (theta-cut 0.05 deg) of DESI 2024 V/VII: the rotated
P_0, P_2 with the Key-Project scale cuts, the rotated window matrix (to the theory P_0, P_2, P_4 on k = 0.001..0.349
h/Mpc) and the covariance including the systematics. Writes external/desi_dr1_fs/<NAME>_{data,cov,window,ktheory}.txt
and <NAME>_meta.txt (zeff, the mean shot noise of P_0, the number of data points), NAME = BGS_z0, LRG_z0, LRG_z1,
LRG_z2, ELG_z1, QSO_z0 (the names of DESI's Cobaya likelihood).
"""
import os
import sys

import h5py
import numpy as np

SETS = [("BGS_BRIGHT-21.5", "BGS_z0", (0.1, 0.4)), ("LRG", "LRG_z0", (0.4, 0.6)), ("LRG", "LRG_z1", (0.6, 0.8)),
        ("LRG", "LRG_z2", (0.8, 1.1)), ("ELG_LOPnotqso", "ELG_z1", (1.1, 1.6)), ("QSO", "QSO_z0", (0.8, 2.1))]
OUT = os.path.join("external", "desi_dr1_fs")


def ells_of(group):
    return [int(x) for x in group["labels_values"][()].astype(str)]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ddir = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    ktheory = None
    for tracer, name, (z1, z2) in SETS:
        fn = os.path.join(ddir, f"likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_{tracer}_GCcomb_z{z1:.1f}-{z2:.1f}_thetacut0.05.h5")
        with h5py.File(fn, "r") as f:
            obs = f["observable/spectrum"]
            zeff = float(obs.attrs["zeff"])
            ells = ells_of(obs)
            assert ells == [0, 2], ells
            data = np.concatenate([obs[str(ell)]["value"][()] for ell in ells])
            kdata = np.concatenate([obs[str(ell)]["k"][()] for ell in ells])
            ell_of_data = np.concatenate([np.full(obs[str(ell)]["k"].shape, ell) for ell in ells])
            shotnoise = float(np.mean(obs["0"]["num_shotnoise"][()] / obs["0"]["norm"][()]))
            cov = f["covariance/value"][()]
            window = f["window/value"][()]
            th = f["window/theory/spectrum"]
            assert ells_of(th) == [0, 2, 4]
            kth = th["0"]["k"][()]
            for ell in (2, 4):
                assert np.allclose(th[str(ell)]["k"][()], kth)
            if ktheory is None:
                ktheory = kth
            assert np.allclose(kth, ktheory), "the theory k differ between the tracers"
            assert window.shape == (data.size, 3 * kth.size) and cov.shape == (data.size, data.size)
        head = f"DESI DR1 full shape {name} ({tracer}, {z1}-{z2}), zeff = {zeff:.10f}"
        np.savetxt(f"{OUT}/{name}_data.txt", np.column_stack([ell_of_data, kdata, data]), header=head + "\nell k [h/Mpc] P_ell [(Mpc/h)^3] (rotated)")
        np.savetxt(f"{OUT}/{name}_cov.txt", cov, header=head)
        np.savetxt(f"{OUT}/{name}_window.txt", window, header=head + f"\nrows: the data; columns: P_0, P_2, P_4 on the {kth.size} theory k")
        np.savetxt(f"{OUT}/{name}_ktheory.txt", kth, header=head + "\ntheory k [h/Mpc]")
        with open(f"{OUT}/{name}_meta.txt", "w") as fp:
            fp.write(f"# {head}\n# zeff, mean shot noise of P_0 [(Mpc/h)^3], number of data points\n{zeff:.15e} {shotnoise:.15e} {data.size}\n")
        print(f"  {name}: zeff = {zeff:.4f}, {data.size} data, shot noise {shotnoise:.1f}, k = {kdata.min():.3f}-{kdata.max():.3f}")


if __name__ == "__main__":
    main()
