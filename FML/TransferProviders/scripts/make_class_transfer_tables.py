#!/usr/bin/env python3
"""Make CLASS transfer function tables in CLASS's native format for the "class" source of make_transfer_provider
(FML/TransferProviders), with CLASS's python wrapper (pip install classy).

The same cosmology arguments and redshift sampling as SimulationCode/scripts/make_camb_transfer_tables.py (the CAMB
tables for the "camb" source), so the two codes can be run on the same input:
  * redshifts uniform in log(1+z) from 0 to zmax (the growth rates come from the a-interpolation of the tables, so
    the spacing must be fine enough; --zlist gives the redshifts explicitly)
  * massive neutrinos as N degenerate species (default 3), N_ur set so that N_eff is the given value
  * the primordial spectrum P_R(k) = A_s (k / k_pivot)^(n_s - 1)

--gauge synchronous (CLASS's default) gives the densities in the CDM-synchronous gauge directly. --gauge newtonian
gives Newtonian-gauge densities and velocities; the provider converts them to the CDM-synchronous gauge with the
Newtonian CDM velocity (input_gauge = Newtonian). For neutrino particles the neutrino fluid approximation is switched
off (ncdm_fluid_approximation = 3, as CLASS's default approximation is inaccurate below the free-streaming scale);
--neutrino-fluid-approximation keeps CLASS's default.

Writes <outdir>/class_data_<prefix>/<prefix>_z<i>_tk.dat (one file per redshift, the header says the redshift) and
<outdir>/class_data_<prefix>/<prefix>_background.dat, and prints the settings for make_transfer_provider:
  class_root = <outdir>/class_data_<prefix>/<prefix>_

Example:
  python3 make_class_transfer_tables.py --OmegaMNu 0.0024 --prefix lcdm_nu0.1 --outdir ../input
"""
import argparse
import os

import numpy as np
import classy
from classy import Class


def write_class_table(filename, header_lines, columns):
    """A table in CLASS's output format: comment lines, the numbered column names, then the data"""
    names = list(columns.keys())
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    with open(filename, "w") as f:
        for line in header_lines:
            f.write("# " + line + "\n")
        f.write("#" + "".join(f" {i + 1:4d}:{n:<20s}" for i, n in enumerate(names)).rstrip() + "\n")
        np.savetxt(f, data, fmt="%25.12e")


def make_tables(outdir, prefix, OmegaM=0.3175, Omegab=0.049, OmegaMNu=0.0, OmegaCDM=None, h=0.671, As=2.215e-9,
                ns=0.966, kpivot=0.05, TCMB=2.7255, Neff=3.046, YHe=0.24, tau=0.09, zmax=200.0, nz=60, zlist=None,
                kmax=100.0, gauge="synchronous", nmassive=3, mnu=None, neutrino_fluid_approximation=False,
                background_points=3000):
    if gauge not in ("synchronous", "newtonian"):
        raise ValueError("gauge must be synchronous or newtonian")
    params = {
        "h": h,
        "Omega_b": Omegab,
        "T_cmb": TCMB,
        "A_s": As,
        "n_s": ns,
        "k_pivot": kpivot,
        "tau_reio": tau,
        "output": "mTk,vTk",
        "gauge": gauge,
        "P_k_max_h/Mpc": kmax,
        "background_Nloga": background_points,
    }
    if YHe != "BBN":
        params["YHe"] = float(YHe)

    # Massive neutrinos: nmassive degenerate species; N_ur so that N_eff = Neff (CLASS's default T_ncdm = 0.71611
    # gives each massive species 1.0132 to N_eff)
    has_nu = (mnu is not None and mnu > 0) or OmegaMNu > 0
    if has_nu:
        params["N_ncdm"] = 1
        params["deg_ncdm"] = nmassive
        if mnu is not None and mnu > 0:
            params["m_ncdm"] = mnu / nmassive
        else:
            params["Omega_ncdm"] = OmegaMNu
        params["N_ur"] = Neff - nmassive * (0.71611 / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4
        if not neutrino_fluid_approximation:
            params["ncdm_fluid_approximation"] = 3
    else:
        params["N_ncdm"] = 0
        params["N_ur"] = Neff
    if OmegaCDM is None:
        if mnu is not None and mnu > 0:
            raise ValueError("with --mnu give --OmegaCDM (OmegaMNu follows from the mass)")
        OmegaCDM = OmegaM - Omegab - OmegaMNu
    params["Omega_cdm"] = OmegaCDM

    if zlist is not None:
        zs = np.array(sorted(set(zlist), reverse=True), dtype=float)
    else:
        zs = np.expm1(np.linspace(0.0, np.log1p(zmax), nz))[::-1]
    params["z_max_pk"] = float(zs.max())
    params["z_pk"] = ", ".join(f"{z:.10g}" for z in zs)

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    datadir = os.path.abspath(os.path.join(outdir, f"class_data_{prefix}"))
    os.makedirs(datadir, exist_ok=True)
    root = os.path.join(datadir, f"{prefix}_")

    # One transfer file per redshift. The provider reads the redshift from "at redshift z=" in the header and the
    # columns by name, as in CLASS's own files
    for i, z in enumerate(zs):
        tk = cosmo.get_transfer(float(z), output_format="class")
        header = [
            f"Transfer functions T_i(k) for adiabatic (AD) mode (normalized to initial curvature=1) at redshift z={z:.10g}",
            f"for k={tk['k (h/Mpc)'][0]:g} to {tk['k (h/Mpc)'][-1]:g} h/Mpc,",
            f"number of wavenumbers equal to {len(tk['k (h/Mpc)'])}",
            f"gauge = {gauge} (written by make_class_transfer_tables.py with classy {getattr(classy, '__version__', '?')})",
            "d_i   stands for (delta rho_i/rho_i)(k,z) with above normalization",
            "t_i   stands for theta_i(k,z) with above normalization",
        ]
        write_class_table(f"{root}z{i + 1}_tk.dat", header, tk)

    # The background (H(a) and the densities and pressures), from early to late times as in CLASS's file
    bg = cosmo.get_background()
    order = np.argsort(-np.asarray(bg["z"]))
    write_class_table(f"{root}background.dat",
                      ["Table of selected background quantities",
                       "All densities are multiplied by (8piG/3) (below, shortcut notation (.) for this factor)",
                       "Densities are in units [Mpc^-2] while all distances are in [Mpc]."],
                      {name: np.asarray(values)[order] for name, values in bg.items()})

    Omega_nu = cosmo.Omega_nu if has_nu else 0.0
    print(f"Wrote {len(zs)} transfer tables (z = {zs.max():g} ... {zs.min():g}, {gauge} gauge) and the background "
          f"to {datadir}")
    print("Cosmology: h = %.6f, Omegab = %.6f, OmegaCDM = %.6f, OmegaMNu = %.6f, A_s = %.6e, n_s = %.6f, "
          "k_pivot = %g/Mpc" % (h, Omegab, OmegaCDM, Omega_nu, As, ns, kpivot))
    print("make_transfer_provider settings:")
    print(f'  source = "class"; class_root = "{root}"; input_gauge = Gauge::'
          f'{"SynchronousCDM" if gauge == "synchronous" else "Newtonian"};')
    print(f"  A_s = {As:.6e}; n_s = {ns:.6f}; k_pivot_mpc = {kpivot:g};")
    cosmo.struct_cleanup()
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=".")
    p.add_argument("--prefix", required=True)
    p.add_argument("--gauge", default="synchronous", choices=["synchronous", "newtonian"])
    p.add_argument("--OmegaM", type=float, default=0.3175, help="CDM + baryons + massive neutrinos")
    p.add_argument("--Omegab", type=float, default=0.049)
    p.add_argument("--OmegaMNu", type=float, default=0.0)
    p.add_argument("--OmegaCDM", type=float, default=None, help="instead of OmegaM (needed with --mnu)")
    p.add_argument("--mnu", type=float, default=None, help="sum of the neutrino masses in eV (instead of OmegaMNu)")
    p.add_argument("--nmassive", type=int, default=3, help="number of degenerate massive neutrinos")
    p.add_argument("--neutrino-fluid-approximation", action="store_true",
                   help="keep CLASS's default ncdm fluid approximation (inaccurate below the free-streaming scale)")
    p.add_argument("--h", type=float, default=0.671)
    p.add_argument("--As", type=float, default=2.215e-9)
    p.add_argument("--ns", type=float, default=0.966)
    p.add_argument("--kpivot", type=float, default=0.05, help="1/Mpc")
    p.add_argument("--TCMB", type=float, default=2.7255)
    p.add_argument("--Neff", type=float, default=3.046)
    p.add_argument("--YHe", default="0.24", help="helium fraction, or BBN (CLASS's default)")
    p.add_argument("--tau", type=float, default=0.09)
    p.add_argument("--zmax", type=float, default=200.0)
    p.add_argument("--nz", type=int, default=60)
    p.add_argument("--zlist", type=float, nargs="+", default=None, help="explicit redshifts (instead of zmax, nz)")
    p.add_argument("--kmax", type=float, default=100.0, help="h/Mpc")
    p.add_argument("--background-points", type=int, default=3000, help="CLASS background_Nloga")
    a = p.parse_args()
    make_tables(a.outdir, a.prefix, a.OmegaM, a.Omegab, a.OmegaMNu, a.OmegaCDM, a.h, a.As, a.ns, a.kpivot, a.TCMB,
                a.Neff, a.YHe, a.tau, a.zmax, a.nz, a.zlist, a.kmax, a.gauge, a.nmassive, a.mnu,
                a.neutrino_fluid_approximation, a.background_points)
