#!/usr/bin/env python3
#==========================================================================
# Fading Dark Sector vs LCDM N-body simulations: the nonlinear boost of the FDS (arXiv:2611.17422) measured
# with SimulationCode and compared with the Einstein-Boltzmann solver's nonlinear = react (the FDS) and
# nonlinear = hmcode2020 (LCDM).
#
# The two cosmologies have the same A_s, n_s, h, Omega_b, T_CMB, N_eff and the same matter-radiation equality: the
# LCDM OmegaCDM is the FDS's CDM density at equality scaled as a^-3 (the coupled CDM density falls by ~2% after it
# at the best fit), computed from the solver's FDS background. Same random phases (fixed amplitude, paired phases),
# so the ratio P_FDS / P_LCDM has almost no cosmic variance. Full N-body (Quinn, no COLA); the CDM + baryons are one
# set of particles (in the FDS: G_eff/G = 1 + 2 c'^2 f_c^2 and the drag f_c (-c' dphi/dlna), as the reaction).
#
#   setup <dir>    parameter files for every box, model and phase, the solver's predictions and run_all.sh
#   analyse <dir>  the ratios of the simulations against the predictions: two plots and a table
#
#   python3 fds_vs_lcdm_simulations.py setup runs/fds [--preset production|test] [--c -0.85] [--cprime 0.053]
#       [--short-range] [--nbody ../nbody] [--cmb ../../Cosmology/EinsteinBoltzmannSolver/cmb]
#   cd runs/fds && MPIRUN="mpirun -np 64" ./run_all.sh     (or the boxes one by one: ./run_all.sh L1000)
#   python3 fds_vs_lcdm_simulations.py analyse runs/fds
#
# Presets (two boxes each; the large box for k < ~0.3 h/Mpc, the small for the nonlinear scales):
#   production: L = 1000 and 250 Mpc/h, 512^3 particles, force grid 1024^3 / 1536^3, 160 / 240 steps. Per run
#               about 40 / 70 GB of memory; 8 runs.
#   test:       L = 500 and 125 Mpc/h, 64^3 particles (the pipeline in minutes on a laptop)
# --short-range adds the P3M short-range force in the small box (SHORT_RANGE.md; not yet calibrated for
# production P(k): compare with the PM runs).
#==========================================================================
import argparse
import glob
import os
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIMCODE = os.path.abspath(os.path.join(HERE, ".."))
EBS = os.path.abspath(os.path.join(SIMCODE, "..", "Cosmology", "EinsteinBoltzmannSolver"))
sys.path.insert(0, os.path.join(SIMCODE, "tests", "regression"))
import cases  # noqa: E402  (the complete list of parameters and lua_value)

REDSHIFTS = [2.0, 1.0, 0.5, 0.0]

# The cosmology (the solver's defaults, as the plots of the reaction; massless neutrinos)
COSMO = dict(h=0.67, Omegab=0.05, OmegaCDM=0.267, As=2.1e-9, ns=0.965, kpivot_mpc=0.05, TCMB=2.7255, Neff=3.046)

PRESETS = {
    "production": [
        dict(name="L1000", box=1000.0, npart=512, nmesh=1024, pofk_nmesh=1024, steps=[40, 40, 30, 50]),
        dict(name="L250", box=250.0, npart=512, nmesh=1536, pofk_nmesh=1024, steps=[60, 60, 45, 75]),
    ],
    "test": [
        dict(name="L500", box=500.0, npart=64, nmesh=128, pofk_nmesh=128, steps=[20, 10, 10, 10]),
        dict(name="L125", box=125.0, npart=64, nmesh=192, pofk_nmesh=128, steps=[30, 15, 15, 20]),
    ],
}


def solver_args(model, c, cprime, Oc):
    a = [f"h={COSMO['h']}", f"OmegaB={COSMO['Omegab']}", f"OmegaCDM={Oc!r}", f"A_s={COSMO['As']}",
         f"n_s={COSMO['ns']}", f"k_pivot={COSMO['kpivot_mpc']}", f"TCMB={COSMO['TCMB']}", f"Neff={COSMO['Neff']}",
         "m_nu=0"]
    if model == "fds":
        a += ["model=fading_dark_sector", f"fds_c={c!r}", f"fds_cprime={cprime!r}"]
    return a


def lcdm_omega_cdm(cmb, c, cprime, workdir):
    """The LCDM OmegaCDM with the matter-radiation equality of the FDS (from the solver's background.txt)"""
    out = os.path.join(workdir, "fds_background")
    os.makedirs(out, exist_ok=True)
    cmd = [cmb] + solver_args("fds", c, cprime, COSMO["OmegaCDM"]) + [
        "compute_cells=false", "pert_k_max=0.01", "write_background=true", "precision=fast", f"output_dir={out}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    d = np.loadtxt(os.path.join(out, "background.txt"))
    x, H, Ob, Oc, Og, On = d[:, 0], d[:, 3], d[:, 7], d[:, 8], d[:, 9], d[:, 10]
    f = np.log((Ob + Oc) / (Og + On))
    i = np.where(np.diff(np.sign(f)) > 0)[0][0]
    xe = x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])
    lnrho = np.log(Oc * H**2 * np.exp(3 * x))  # rho_c a^3 in units of (100 km/s/Mpc)^2 3/(8 pi G)
    return float(np.exp(np.interp(xe, x, lnrho)) / COSMO["h"]**2), float(np.exp(-xe) - 1)


def sim_params(box, model, phase, c, cprime, Oc, short_range):
    p = dict(cases.BASE)
    for k in [k for k in p if k.startswith("bispectrum") or k.startswith("pofk_multipole") or k.startswith("fof")]:
        del p[k]
    p.update(
        all_parameters_must_be_in_file=False,
        simulation_name=f"{model}_{phase}",
        simulation_boxsize=box["box"],
        simulation_use_cola=False,
        cosmology_model="FadingDarkSector" if model == "fds" else "LCDM",
        gravity_model="FadingDarkSector" if model == "fds" else "GR",
        cosmology_OmegaCDM=Oc,
        cosmology_Omegab=COSMO["Omegab"],
        cosmology_h=COSMO["h"],
        cosmology_As=COSMO["As"],
        cosmology_ns=COSMO["ns"],
        cosmology_kpivot_mpc=COSMO["kpivot_mpc"],
        cosmology_TCMB_kelvin=COSMO["TCMB"],
        cosmology_Neffective=COSMO["Neff"],
        particle_Npart_1D=box["npart"],
        particle_allocation_factor=1.6,
        output_redshifts=REDSHIFTS,
        output_particles=False,
        output_folder="output",
        timestep_nsteps=box["steps"],
        timestep_method="Quinn",
        timestep_algorithm="KDK",
        timestep_scalefactor_spacing="linear",
        ic_random_seed=1234,
        ic_fix_amplitude=True,
        ic_reverse_phases=(phase == "b"),
        ic_nmesh=box["npart"],
        ic_LPT_order=2,
        ic_type_of_input="internal",
        ic_input_filename="",
        ic_input_redshift=49.0,
        ic_initial_redshift=49.0,
        ic_use_gravity_model_GR=False,
        ic_axion_fraction=0.0,
        ic_axion_log10_m=-22.0,
        ic_axion_switch_mH=0.0,
        force_nmesh=box["nmesh"],
        pofk=True,
        pofk_nmesh=box["pofk_nmesh"],
        pofk_interlacing=True,
        pofk_subtract_shotnoise=True,
        pofk_density_assignment_method="PCS",
        pofk_every_step=False,
    )
    if model == "fds":
        p.update(cosmology_fds_c=c, cosmology_fds_cprime=cprime)
    if short_range:
        cell = box["box"] / box["nmesh"]
        p.update(force_greens_function_kernel="continuous", force_gradient_kernel="continuous",
                 force_density_method="assignment", force_interpolation_method="assignment",
                 force_early_method="none", force_short_range=True, force_short_range_backend="cell",
                 force_short_range_split_scale=1.25 * cell,
                 force_short_range_softening_support=2.8 * box["box"] / box["npart"] / 30.0,
                 force_short_range_cutoff_tolerance=1e-6, force_short_range_table_points=16384)
    return p


def setup(a):
    cmb, nbody = os.path.abspath(a.cmb), os.path.abspath(a.nbody)
    root = os.path.abspath(a.dir)
    os.makedirs(root, exist_ok=True)
    Oc_lcdm, zeq = lcdm_omega_cdm(cmb, a.c, a.cprime, root)
    print(f"FDS c = {a.c}, c' = {a.cprime}: z_eq = {zeq:.2f}; LCDM with the same z_eq: OmegaCDM = {Oc_lcdm:.6f} "
          f"(FDS today {COSMO['OmegaCDM']})")
    models = {"fds": COSMO["OmegaCDM"], "lcdm": Oc_lcdm}
    boxes = PRESETS[a.preset]
    runs = []
    for box in boxes:
        for model, Oc in models.items():
            for phase in "ab":
                d = os.path.join(root, box["name"], f"{model}_{phase}")
                os.makedirs(d, exist_ok=True)
                p = sim_params(box, model, phase, a.c, a.cprime, Oc, a.short_range and box is boxes[-1])
                with open(os.path.join(d, "parameterfile.lua"), "w") as f:
                    f.write(f"-- FDS vs LCDM: {box['name']} {model} phase {phase} (fds_vs_lcdm_simulations.py)\n")
                    for k, v in p.items():
                        f.write(f"{k} = {cases.lua_value(v)}\n")
                runs.append((box["name"], f"{model}_{phase}"))

    # The solver's predictions at the output redshifts: linear P_cb and nonlinear P (react / hmcode2020)
    pred = []
    for model, Oc in models.items():
        out = os.path.join(root, "predictions", model)
        nl = "react" if model == "fds" else "hmcode2020"
        args = solver_args(model, a.c, a.cprime, Oc) + [
            "compute_cells=false", "pert_k_max=10", f"nonlinear={nl}", "write_matter_power=true",
            "matter_power_z=" + ",".join(f"{z:g}" for z in REDSHIFTS), f"output_dir={out}"]
        pred.append((model, " ".join([shlex.quote(cmb)] + [shlex.quote(x) for x in args])))

    with open(os.path.join(root, "run_all.sh"), "w") as f:
        f.write("#!/bin/bash\n# FDS vs LCDM simulations (fds_vs_lcdm_simulations.py). Usage: MPIRUN=\"mpirun -np N\" "
                "./run_all.sh [box ...]\n")
        f.write("# Each run starts with the internal Einstein-Boltzmann solver on task 0 (~1-3 min)\nset -e\n")
        f.write('cd "$(dirname "$0")"\nMPIRUN=${MPIRUN:-}\n')
        f.write(f"NBODY={shlex.quote(nbody)}\n")
        f.write("if [ ! -f predictions/fds/matter_power.txt ] || [ ! -f predictions/lcdm/matter_power.txt ]; then\n")
        for model, c in pred:
            f.write(f"  echo 'predictions: {model}'; {c} > predictions_{model}.log 2>&1\n")
        f.write("fi\n")
        f.write(f'BOXES=${{@:-{" ".join(b["name"] for b in boxes)}}}\n')
        f.write('for b in $BOXES; do for r in fds_a fds_b lcdm_a lcdm_b; do\n')
        f.write('  if ls $b/$r/output/snapshot_*_z0.000/pofk_z0.000.txt > /dev/null 2>&1; then echo "$b/$r done"; continue; fi\n')
        f.write('  echo "$b/$r"; (cd $b/$r && $MPIRUN $NBODY parameterfile.lua > log.txt 2>&1)\n')
        f.write("done; done\n")
    os.chmod(os.path.join(root, "run_all.sh"), 0o755)
    print(f"{len(runs)} runs in {root}; run them with: cd {root} && MPIRUN=\"mpirun -np N\" ./run_all.sh")


def read_pofk(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def find_pofk(run, z):
    fs = sorted(glob.glob(os.path.join(run, "output", f"snapshot_*_z{z:.3f}", f"pofk_z{z:.3f}.txt")))
    if not fs:
        raise FileNotFoundError(f"no P(k) at z = {z} in {run}/output")
    return fs[0]


def analyse(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.abspath(a.dir)
    pred = {}
    for model in ("fds", "lcdm"):
        pred[model] = {}
        for z in REDSHIFTS:
            name = "matter_power.txt" if z == 0 else f"matter_power_z{z:g}.txt"
            f = os.path.join(root, "predictions", model, name)
            if not os.path.exists(f):
                f = glob.glob(os.path.join(root, "predictions", model, f"matter_power_z{z}*.txt"))[0]
            d = np.loadtxt(f)
            pred[model][z] = (d[:, 0], d[:, 2], d[:, 3])  # k [h/Mpc], linear P_cb, nonlinear P
    boxes = sorted(b for b in os.listdir(root) if os.path.isdir(os.path.join(root, b, "fds_a")))
    color = {0.0: "#2a78d6", 0.5: "#eb6834", 1.0: "#1baf7a", 2.0: "#eda100"}
    zs = sorted(REDSHIFTS)
    kmin, kmax = 1e9, 0.0
    markers = ["o", "s", "^"]
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.9), dpi=200)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.9), dpi=200)
    rows = []
    curves = []
    for z in zs:
        c = color[z]
        kf, linf, nlf = pred["fds"][z]
        kl, linl, nll = pred["lcdm"][z]
        lin = np.interp(kf, kl, linl)
        curves.append((c, kf, nlf / np.interp(kf, kl, nll), linf / lin, (nlf / linf) / (np.interp(kf, kl, nll) / lin)))
        for ib, b in enumerate(boxes):
            P = {}
            for r in ("fds_a", "fds_b", "lcdm_a", "lcdm_b"):
                P[r] = read_pofk(find_pofk(os.path.join(root, b, r), z))
            k = P["fds_a"][0]
            pf = 0.5 * (P["fds_a"][1] + P["fds_b"][1])
            pl = 0.5 * (P["lcdm_a"][1] + P["lcdm_b"][1])
            box = float([l.split("=")[1] for l in open(os.path.join(root, b, "fds_a", "parameterfile.lua"))
                         if l.startswith("simulation_boxsize")][0])
            npart = int([l.split("=")[1] for l in open(os.path.join(root, b, "fds_a", "parameterfile.lua"))
                         if l.startswith("particle_Npart_1D")][0])
            # Up to half the particle Nyquist frequency, and where the power is well above the shot noise
            m = (k > 0) & (k < 0.5 * np.pi * npart / box) & (pl > 10 * (box / npart)**3)
            if not m.any():
                continue
            kmin, kmax = min(kmin, k[m][0]), max(kmax, k[m][-1])
            r = pf[m] / pl[m]
            linr = np.interp(k[m], kf, linf) / np.interp(k[m], kl, linl)
            ax1.plot(k[m], r, ls="none", marker=markers[ib % 3], ms=2.5, color=c, alpha=0.7)
            ax2.plot(k[m], r / linr, ls="none", marker=markers[ib % 3], ms=2.5, color=c, alpha=0.7)
            pr = np.interp(k[m], kf, nlf) / np.interp(k[m], kl, nll)
            for kk in (0.1, 0.3, 1.0, 2.0):
                if k[m][0] <= kk <= k[m][-1]:
                    j = np.argmin(abs(k[m] - kk))
                    rows.append((b, z, k[m][j], r[j], pr[j], linr[j]))
    for c, kf, r, lr, br in curves:
        w = (kf > 0.7 * kmin) & (kf < 1.4 * kmax)
        ax1.plot(kf[w], r[w], color=c, lw=1.8)
        ax1.plot(kf[w], lr[w], color=c, lw=1.2, ls="--")
        ax2.plot(kf[w], br[w], color=c, lw=1.8)
    for ax, yl, title, fn in (
            (ax1, r"$P_{\rm FDS}/P_{\Lambda{\rm CDM}}$", "Power spectrum relative to LCDM", "fds_sims_ratio.png"),
            (ax2, r"$(P_{\rm NL}/P_{\rm lin})_{\rm FDS}/(P_{\rm NL}/P_{\rm lin})_{\Lambda{\rm CDM}}$",
             "Nonlinear boost relative to LCDM", "fds_sims_boost_ratio.png")):
        for z in zs:
            ax.plot([], [], color=color[z], lw=2, label=f"z = {z:g}")
        ax.plot([], [], color="#777", lw=1.8, label="ReACT / HMcode-2020")
        if ax is ax1:
            ax.plot([], [], color="#777", lw=1.2, ls="--", label="linear")
        for ib, b in enumerate(boxes):
            ax.plot([], [], color="#777", ls="none", marker=markers[ib % 3], label=f"simulations {b}")
        ax.axhline(1, color="#999", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.7 * kmin, 1.4 * kmax)
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(root, fn))
    print(f"{'box':>6} {'z':>4} {'k':>7} {'sim':>8} {'react':>8} {'linear':>8}")
    for b, z, k, r, pr, lr in rows:
        print(f"{b:>6} {z:4.1f} {k:7.3f} {r:8.4f} {pr:8.4f} {lr:8.4f}")
    print(f"plots: {root}/fds_sims_ratio.png, {root}/fds_sims_boost_ratio.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("dir")
    s.add_argument("--preset", choices=PRESETS, default="production")
    s.add_argument("--c", type=float, default=-0.85)
    s.add_argument("--cprime", type=float, default=0.053)
    s.add_argument("--short-range", action="store_true")
    s.add_argument("--nbody", default=os.path.join(SIMCODE, "nbody"))
    s.add_argument("--cmb", default=os.path.join(EBS, "cmb"))
    s = sub.add_parser("analyse")
    s.add_argument("dir")
    a = ap.parse_args()
    setup(a) if a.cmd == "setup" else analyse(a)
