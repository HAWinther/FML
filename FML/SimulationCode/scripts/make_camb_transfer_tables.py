#!/usr/bin/env python3
"""Make CAMB transfer function tables in the format SimulationCode reads (ic_type_of_input = "transferinfofile",
ic_type_of_input_fileformat = "CAMB"), with accurate massive neutrino transfer functions.

The redshifts are uniform in log(1+z) from 0 to zmax (the growth rates are computed from the tables with a 4-point
Lagrange derivative in log(a), so the spacing should be fine enough: the default 60 redshifts up to z = 200 give
the growth rates of CDM and baryons to ~1e-4; the infofile has the exact redshifts: with 3 decimals the growth
rates at z = 0.15-1 were 0.3-0.6% off). NB: the massive neutrino growth rate (used for the initial velocities of
neutrino particles) needs a much finer spacing at high z: with these 60 redshifts it is 8-66% off at z >= 150 and
k >= 0.05 h/Mpc (fine at z <= 49). The massive neutrinos are N_nu = 3 degenerate species, as in SimulationCode. The default CAMB accuracy is not enough for
the massive neutrino transfer function below the free-streaming scale: for 3 x 0.033 eV P_nu(k, z = 0) is 3% / 13% /
16% too low at k = 0.014 / 0.05 / 0.1 h/Mpc (relative to AccuracyBoost = 3, which agrees with CLASS without the ncdm
fluid approximation), so we use --accuracy 3 by default (CDM and baryons are unaffected).

Example (the tables in input/camb_data_lcdm_nu0.4_zmax200):
  python3 make_camb_transfer_tables.py --OmegaMNu 0.0095 --prefix lcdm_nu0.4_zmax200 --outdir ../input

Writes <outdir>/camb_data_<prefix>/<prefix>_transfer_z*.dat and <outdir>/transfer_infofile_<prefix>.txt.
Use cosmology_OmegaCDM = OmegaM - Omegab - OmegaMNu, cosmology_Omegab, cosmology_OmegaMNu, cosmology_h, ... with
the same values in the parameter file. Needs camb (pip install camb).
"""
import argparse
import os

import numpy as np
import camb


def make_tables(outdir, prefix, OmegaM=0.3175, Omegab=0.049, OmegaMNu=0.0, h=0.671, As=2.215e-9, ns=0.966,
                kpivot=0.05, TCMB=2.7255, Neff=3.046, YHe=0.24, tau=0.09, zmax=200.0, nz=60, kmax=100.0,
                k_per_logint=5, accuracy=3):
    OmegaCDM = OmegaM - Omegab - OmegaMNu
    pars = camb.CAMBparams()
    common = dict(H0=100 * h, ombh2=Omegab * h * h, omch2=OmegaCDM * h * h, nnu=Neff, TCMB=TCMB, YHe=YHe, tau=tau)
    if OmegaMNu > 0:
        # Iterate on the mass to get OmegaMNu exactly (CAMB takes the sum of the masses)
        mnu = OmegaMNu * h * h * 93.14
        for _ in range(4):
            pars.set_cosmology(mnu=mnu, num_massive_neutrinos=3, neutrino_hierarchy="degenerate", **common)
            mnu *= OmegaMNu * h * h / pars.omnuh2
        print(f"Sum of neutrino masses {mnu:.5f} eV (3 degenerate), OmegaMNu = {pars.omnuh2 / h**2:.6f}")
    else:
        pars.set_cosmology(mnu=0.0, num_massive_neutrinos=0, **common)
    pars.InitPower.set_params(As=As, ns=ns, pivot_scalar=kpivot)
    zs = np.expm1(np.linspace(0.0, np.log1p(zmax), nz))[::-1]
    pars.set_matter_power(redshifts=list(zs), kmax=kmax, k_per_logint=k_per_logint,
                          accurate_massive_neutrino_transfers=True)
    pars.Transfer.high_precision = True
    pars.set_accuracy(AccuracyBoost=accuracy, lAccuracyBoost=accuracy)
    res = camb.get_results(pars)
    td = res.get_matter_transfer_data()
    zout = np.array(res.transfer_redshifts)

    datadir = os.path.abspath(os.path.join(outdir, f"camb_data_{prefix}"))
    os.makedirs(datadir, exist_ok=True)
    names = ["k/h", "CDM", "baryon", "photon", "nu", "mass_nu", "total", "no_nu", "total_de", "Weyl", "v_CDM", "v_b",
             "v_b-v_c"]
    files = []
    for iz, z in enumerate(zout):
        fn = f"{prefix}_transfer_z{z:.3f}.dat"  # (the file name only; the exact redshift goes in the infofile)
        np.savetxt(os.path.join(datadir, fn), td.transfer_data[:, :, iz].T, fmt="%14.6E",
                   header="".join(f"{n:>15}" for n in names))
        files.append((z, fn))
    files.sort()
    infofile = os.path.join(outdir, f"transfer_infofile_{prefix}.txt")
    with open(infofile, "w") as f:
        f.write(f"{datadir} {len(files)}\n")
        for z, fn in files:
            f.write(f"{fn}  {z:.10g}\n")
    print(f"Wrote {len(files)} tables to {datadir} and {infofile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=".")
    p.add_argument("--prefix", required=True)
    p.add_argument("--OmegaM", type=float, default=0.3175, help="CDM + baryons + massive neutrinos")
    p.add_argument("--Omegab", type=float, default=0.049)
    p.add_argument("--OmegaMNu", type=float, default=0.0)
    p.add_argument("--h", type=float, default=0.671)
    p.add_argument("--As", type=float, default=2.215e-9)
    p.add_argument("--ns", type=float, default=0.966)
    p.add_argument("--kpivot", type=float, default=0.05, help="1/Mpc")
    p.add_argument("--TCMB", type=float, default=2.7255)
    p.add_argument("--Neff", type=float, default=3.046)
    p.add_argument("--zmax", type=float, default=200.0)
    p.add_argument("--nz", type=int, default=60)
    p.add_argument("--kmax", type=float, default=100.0, help="h/Mpc")
    p.add_argument("--accuracy", type=float, default=3, help="CAMB AccuracyBoost and lAccuracyBoost")
    a = p.parse_args()
    make_tables(a.outdir, a.prefix, OmegaM=a.OmegaM, Omegab=a.Omegab, OmegaMNu=a.OmegaMNu, h=a.h, As=a.As, ns=a.ns,
                kpivot=a.kpivot, TCMB=a.TCMB, Neff=a.Neff, zmax=a.zmax, nz=a.nz, kmax=a.kmax, accuracy=a.accuracy)
