#!/usr/bin/env python3
#==========================================================================
# Make the input for running the COLASolver in the N-body gauge
# (Fidler et al. 2015, 2016) with CLASS (needs the python wrapper: pip install classy)
#
# In the N-body gauge the particles (CDM + baryons) follow Newtonian equations of
# motion exactly to linear order in GR, but the force is sourced by more than the
# particles: photons, massless and massive neutrinos and a metric term. We compute
# this extra source as a linear density field delta_extra(k,z), defined such that
#
#   delta_cb'' + calH delta_cb' = 4 pi G a^2 rho_cb (delta_cb + delta_extra)
#
# reproduces the N-body gauge delta_cb(k,z) of CLASS (' = d/dtau, conformal time).
# With delta_cb' = -theta_cb (continuity holds exactly in the N-body gauge) this is
#
#   delta_extra = -(theta_cb' + calH theta_cb) / (4 pi G a^2 rho_cb) - delta_cb
#
# with theta_cb in the N-body gauge computed from the synchronous gauge output of CLASS:
# theta_c = H_T' + (h' + 6 eta')/2 and theta_b = theta_c + theta_b(synchronous).
#
# Output (in --outdir):
#   nbgauge_transfer_z*.dat     N-body gauge transfer functions in the CAMB format
#                               (for the initial conditions; the photon/nu/massive nu
#                               columns are also in the N-body gauge)
#   nbgauge_transferinfo.txt    The transfer info file listing them (ic_input_filename)
#   nbgauge_extra.txt           T_extra(k,z) = -delta_extra/k^2 in the same units
#                               (force_nbody_gauge_filename)
# and prints the lines to put in the parameterfile.
#
# The cosmology is read from a COLASolver parameterfile (simple top-level assignments
# like cosmology_h = 0.67) and can be overridden on the command line.
#
# Usage:
#   python3 make_nbody_gauge_transfer.py parameterfile.lua --outdir nbgauge_input
#   python3 make_nbody_gauge_transfer.py --h 0.67 --Omegab 0.049 ... --zini 49 --outdir out
#==========================================================================
import argparse
import os
import re
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

PARAMS = {
    # name in the parameterfile       : (argument, default)
    "cosmology_h": ("h", 0.671),
    "cosmology_Omegab": ("Omegab", 0.049),
    "cosmology_OmegaCDM": ("OmegaCDM", 0.2685),
    "cosmology_OmegaMNu": ("OmegaMNu", 0.0),
    "cosmology_OmegaK": ("OmegaK", 0.0),
    "cosmology_Neffective": ("Neff", 3.046),
    "cosmology_TCMB_kelvin": ("TCMB", 2.7255),
    "cosmology_As": ("As", 2.215e-9),
    "cosmology_ns": ("ns", 0.966),
    "cosmology_kpivot_mpc": ("kpivot", 0.05),
    "ic_initial_redshift": ("zini", 49.0),
    "simulation_boxsize": ("boxsize", 1000.0),
    "force_nmesh": ("nmesh", 512),
}


def read_parameterfile(filename):
    """Top-level assignments 'name = value' (numbers and strings) in a Lua parameterfile"""
    values = {}
    with open(filename) as f:
        for line in f:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^-\n][^\n]*?)\s*(--.*)?$", line)
            if not m or m.group(1) in values:
                continue
            value = m.group(2).strip()
            if value.startswith('"') and value.endswith('"'):
                values[m.group(1)] = value[1:-1]
            else:
                try:
                    values[m.group(1)] = float(value)
                except ValueError:
                    pass
    return values


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parameterfile", nargs="?", help="COLASolver parameterfile to take the cosmology from")
    parser.add_argument("--outdir", required=True, help="Folder for the output")
    for name, (arg, default) in PARAMS.items():
        parser.add_argument("--" + arg, type=float, default=None, help=f"{name} (default {default})")
    parser.add_argument("--zmax", type=float, default=None,
                        help="Largest redshift in the tables (default: a bit above the initial redshift)")
    parser.add_argument("--nz", type=int, default=250, help="Number of redshifts in the tables")
    parser.add_argument("--kmax", type=float, default=None,
                        help="Largest k in h/Mpc (default: 2 x the Nyquist frequency of the force grid, at least 1)")
    args = parser.parse_args()

    from_file = read_parameterfile(args.parameterfile) if args.parameterfile else {}
    if args.parameterfile:
        for name, required in [("cosmology_model", "LCDM"), ("gravity_model", "GR")]:
            if name in from_file and from_file[name] != required:
                sys.exit(f"The N-body gauge input is only implemented for {name} = {required} "
                         f"(the parameterfile has {from_file[name]})")
    p = {}
    for name, (arg, default) in PARAMS.items():
        value = getattr(args, arg)
        p[arg] = value if value is not None else from_file.get(name, default)
    p["nz"], p["outdir"] = args.nz, args.outdir
    p["zmax"] = args.zmax if args.zmax is not None else 1.25 * (1 + p["zini"]) - 1
    knyq = np.pi * p["nmesh"] / p["boxsize"]
    p["kmax"] = args.kmax if args.kmax is not None else max(1.0, 2 * knyq)
    return p


def run_class(p, nbody_gauge):
    from classy import Class

    settings = {
        "h": p["h"],
        "Omega_b": p["Omegab"],
        "Omega_cdm": p["OmegaCDM"],
        "Omega_k": p["OmegaK"],
        "T_cmb": p["TCMB"],
        "A_s": p["As"],
        "n_s": p["ns"],
        "k_pivot": p["kpivot"],
        "output": "mTk,vTk",
        "extra_metric_transfer_functions": "yes",
        "gauge": "synchronous",
        "z_max_pk": p["zmax"] + 1.0,
        "P_k_max_h/Mpc": p["kmax"],
        "k_per_decade_for_pk": 20,
    }
    if p["OmegaMNu"] > 0.0:
        # As in the COLASolver: three neutrinos of equal mass with temperature set by Neff
        settings.update({"N_ur": 0.0, "N_ncdm": 1, "deg_ncdm": 3, "Omega_ncdm": p["OmegaMNu"],
                         "T_ncdm": (p["Neff"] / 3.0) ** 0.25 * (4.0 / 11.0) ** (1.0 / 3.0)})
    else:
        settings["N_ur"] = p["Neff"]
    if nbody_gauge:
        settings["nbody_gauge_transfer_functions"] = "yes"
    cosmo = Class()
    cosmo.set(settings)
    cosmo.compute()
    return cosmo


def main():
    p = parse_arguments()
    os.makedirs(p["outdir"], exist_ok=True)
    h = p["h"]

    # Redshifts: uniform in log(a), including z = 0 and the initial redshift
    lna = np.linspace(np.log(1.0 / (1.0 + p["zmax"])), 0.0, p["nz"])
    lna = np.unique(np.concatenate([lna, [np.log(1.0 / (1.0 + p["zini"]))]]))
    a = np.exp(lna)
    zs = np.maximum(1.0 / a - 1.0, 0.0)

    # delta_extra needs a time derivative which we compute on a finer grid
    lna_fine = np.linspace(lna[0], 0.0, max(1200, 4 * p["nz"]))
    a_fine = np.exp(lna_fine)
    zs_fine = np.maximum(1.0 / a_fine - 1.0, 0.0)

    print("Running CLASS...")
    sync = run_class(p, nbody_gauge=False)
    nbg = run_class(p, nbody_gauge=True)
    T_nbg = [nbg.get_transfer(z) for z in zs]
    T_sync = [sync.get_transfer(z) for z in zs_fine]
    k_hmpc = T_nbg[0]["k (h/Mpc)"]
    k = k_hmpc * h  # 1/Mpc

    def sync_col(name):
        return np.array([t[name] for t in T_sync])

    def nbg_col(name):
        return np.array([t[name] for t in T_nbg])

    # Background (CLASS units: rho in 1/Mpc^2 with 8 pi G/3 = 1, so 4 pi G a^2 rho = 1.5 a^2 rho)
    H0 = sync.Hubble(0.0)
    E = np.array([sync.Hubble(z) for z in zs]) / H0
    E_fine = np.array([sync.Hubble(z) for z in zs_fine]) / H0
    calH, calH_fine = a * E * H0, a_fine * E_fine * H0
    rho_cb = (p["Omegab"] + p["OmegaCDM"]) * H0**2 / a**3
    rho_cb_fine = (p["Omegab"] + p["OmegaCDM"]) * H0**2 / a_fine**3
    fb = p["Omegab"] / (p["Omegab"] + p["OmegaCDM"])

    # N-body gauge delta_cb and theta_cb (on the fine grid, from the synchronous gauge)
    shift = 3 * calH_fine[:, None] * sync_col("t_tot") / k**2
    delta_cb_fine = (1 - fb) * (sync_col("d_cdm") + shift) + fb * (sync_col("d_b") + shift)
    theta_c = sync_col("H_T_Nb_prime") + 0.5 * (sync_col("h_prime") + 6 * sync_col("eta_prime"))
    theta_cb_fine = theta_c + fb * sync_col("t_b")

    # The extra source (and interpolate to the output redshifts). We need theta_cb' + calH theta_cb. To avoid
    # numerical derivatives of large quantities we use the Einstein equation for (h' + 6 eta')' (eq. 21d in
    # Ma & Bertschinger 1995; 12 pi G a^2 (rho+p) sigma = k2gamma_Nb + calH H_T' in CLASS) which gives
    #   theta_c' + calH theta_c = H_T'' - calH (h' + 6 eta')/2 + k^2 eta - k2gamma_Nb
    # and for the baryons we add (theta_b' + calH theta_b) in the synchronous gauge
    def ddtau(f):
        return calH_fine[:, None] * np.gradient(f, lna_fine, axis=0, edge_order=2)

    theta_b_sync = sync_col("t_b")
    rhs_c = (ddtau(sync_col("H_T_Nb_prime")) - 0.5 * calH_fine[:, None] * (sync_col("h_prime") + 6 * sync_col("eta_prime"))
             + k**2 * sync_col("eta") - sync_col("k2gamma_Nb"))
    rhs_b = rhs_c + ddtau(theta_b_sync) + calH_fine[:, None] * theta_b_sync
    rhs_cb = (1 - fb) * rhs_c + fb * rhs_b
    delta_extra_fine = -rhs_cb / (1.5 * a_fine**2 * rho_cb_fine)[:, None] - delta_cb_fine

    # Check: the same with a numerical derivative of theta_cb (less accurate at low redshift)
    rhs_numerical = ddtau(theta_cb_fine) + calH_fine[:, None] * theta_cb_fine
    sel = zs_fine > 1.0
    err = np.max(np.abs(rhs_numerical[sel] / rhs_cb[sel] - 1))
    print(f"Check: theta_cb' + calH theta_cb from the Einstein equations vs numerical derivative (z > 1): "
          f"max rel. difference {err:.1e}")
    delta_extra = CubicSpline(lna_fine, delta_extra_fine, axis=0)(lna)
    theta_cb = CubicSpline(lna_fine, theta_cb_fine, axis=0)(lna)
    delta_cb = (1 - fb) * nbg_col("d_cdm") + fb * nbg_col("d_b")

    # Check: CLASS's N-body gauge delta_cb agrees with our synchronous -> N-body gauge transformation
    err = np.max(np.abs(CubicSpline(lna_fine, delta_cb_fine, axis=0)(lna) / delta_cb - 1))
    print(f"Check: N-body gauge delta_cb from CLASS vs our transformation: max rel. difference {err:.1e}")

    # Check: continuity delta_cb' = -theta_cb
    ddelta_dtau = calH_fine[:, None] * np.gradient(delta_cb_fine, lna_fine, axis=0, edge_order=2)
    err = np.max(np.abs(ddelta_dtau[sel] / (-theta_cb_fine[sel]) - 1))
    print(f"Check: continuity delta_cb' = -theta_cb in the N-body gauge (z > 1): max rel. difference {err:.1e}")

    #======================================================================
    # Write the transfer files (CAMB format: T = -delta/k^2 in Mpc^2, k in h/Mpc)
    #======================================================================
    def T(delta):
        return -delta / k**2

    header = ("#          k/h    CDM            baryon         photon         nu             mass_nu        total"
              "          no_nu          total_de       Weyl           v_CDM          v_b            v_b-v_c\n")
    has_ncdm = "d_ncdm[0]" in T_nbg[0]
    files = []
    for i, z in enumerate(zs):
        t = T_nbg[i]
        d_ncdm = t["d_ncdm[0]"] if has_ncdm else t["d_cdm"]
        Om_cb = p["Omegab"] + p["OmegaCDM"]
        d_m = (p["Omegab"] * t["d_b"] + p["OmegaCDM"] * t["d_cdm"] + p["OmegaMNu"] * d_ncdm) / (Om_cb + p["OmegaMNu"])
        d_nonu = (p["Omegab"] * t["d_b"] + p["OmegaCDM"] * t["d_cdm"]) / Om_cb
        zero = np.zeros_like(k)
        cols = [k_hmpc, T(t["d_cdm"]), T(t["d_b"]), T(t["d_g"]), T(t["d_ur"]) if "d_ur" in t else zero,
                T(d_ncdm), T(d_m), T(d_nonu), T(d_m), zero, zero, zero, zero]
        name = f"nbgauge_transfer_z{z:.6f}.dat"
        np.savetxt(os.path.join(p["outdir"], name), np.array(cols).T, fmt="%15.6e", header=header[1:-1], comments="#")
        files.append((name, z))

    # The info file: folder, number of files, then the files from low to high redshift
    with open(os.path.join(p["outdir"], "nbgauge_transferinfo.txt"), "w") as f:
        f.write(f"{os.path.abspath(p['outdir'])} {len(files)}\n")
        for name, z in sorted(files, key=lambda x: x[1]):
            f.write(f"{name} {z:.6f}\n")

    # The extra source: T_extra(k, z) in the same units
    with open(os.path.join(p["outdir"], "nbgauge_extra.txt"), "w") as f:
        f.write("# N-body gauge extra linear source for the COLASolver (made by make_nbody_gauge_transfer.py)\n")
        f.write("# T_extra(k,z) = -delta_extra(k,z)/k^2 in Mpc^2 (same normalization as the CAMB transfer functions)\n")
        f.write("# Format: line 1 (after the comments): nk nz\n")
        f.write("#         line 2: the nz redshifts, line 3: H(z)/H0 at these redshifts\n")
        f.write("#         then nk lines: k (h/Mpc) and T_extra at the nz redshifts\n")
        f.write(f"# cosmology h={p['h']!r} Omegab={p['Omegab']!r} OmegaCDM={p['OmegaCDM']!r} "
                f"OmegaMNu={p['OmegaMNu']!r} OmegaK={p['OmegaK']!r} Neff={p['Neff']!r} TCMB={p['TCMB']!r} "
                f"As={p['As']!r} ns={p['ns']!r} kpivot={p['kpivot']!r}\n")
        order = np.argsort(zs)  # increasing redshift
        f.write(f"{len(k)} {len(zs)}\n")
        f.write(" ".join(f"{z:.10e}" for z in zs[order]) + "\n")
        f.write(" ".join(f"{e:.10e}" for e in E[order]) + "\n")
        Textra = T(delta_extra)[order]
        for j in range(len(k)):
            f.write(f"{k_hmpc[j]:.10e} " + " ".join(f"{x:.10e}" for x in Textra[:, j]) + "\n")

    #======================================================================
    # Diagnostics: evolve the linear equation the simulation solves from the initial redshift
    # with and without the extra source and compare to CLASS at z = 0
    #======================================================================
    i0 = int(np.argmin(np.abs(zs - p["zini"])))
    x = lna[i0:]
    # The background on the fine grid, the source as in the table (interpolated from the output redshifts)
    Om_cb = CubicSpline(lna_fine, 1.5 * a_fine**2 * rho_cb_fine / calH_fine**2)
    dlnH = CubicSpline(lna_fine, np.gradient(np.log(calH_fine), lna_fine, edge_order=2))
    print(f"\nLinear growth from z = {p['zini']:g} to 0 compared to CLASS (N-body gauge):")
    print("    k (h/Mpc)    without delta_extra    with delta_extra")
    kbox = 2 * np.pi / p["boxsize"]
    for kk in sorted(set([kbox, 1e-3, 3e-3, 1e-2, 0.1])):
        j = int(np.argmin(np.abs(k_hmpc - kk)))
        dE = CubicSpline(x, delta_extra[i0:, j])
        res = []
        for with_extra in [False, True]:
            def rhs(xx, y):
                return [y[1], -(1 + dlnH(xx)) * y[1] + Om_cb(xx) * (y[0] + (dE(xx) if with_extra else 0.0))]
            y0 = [delta_cb[i0, j], -theta_cb[i0, j] / calH[i0]]
            sol = solve_ivp(rhs, [x[0], x[-1]], y0, rtol=1e-10, atol=1e-30)
            res.append(sol.y[0, -1] / delta_cb[-1, j] - 1)
        print(f"    {k_hmpc[j]:.2e}     {100 * res[0]:+.3f}%               {100 * res[1]:+.5f}%")
    print("  (the error in delta; the error in P(k) is twice this)")

    out = os.path.abspath(p["outdir"])
    print(f"""
Wrote {len(files)} transfer files and the extra source to {out}
Put this in the parameterfile:

ic_type_of_input = "transferinfofile"
ic_type_of_input_fileformat = "CAMB"
ic_input_filename = "{out}/nbgauge_transferinfo.txt"
ic_input_redshift = {p['zini']!r}
ic_initial_redshift = {p['zini']!r}
force_linear_massive_neutrinos = false
force_nbody_gauge = true
force_nbody_gauge_filename = "{out}/nbgauge_extra.txt"
""")


if __name__ == "__main__":
    main()
