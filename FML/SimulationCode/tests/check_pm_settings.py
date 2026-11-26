#!/usr/bin/env python3
#==========================================================================
# Accuracy of the PM settings (backlog item 59): quick 64^3 regression cases
#
# 64^3 particles in 200 Mpc/h (k_Ny = 1.0 h/Mpc), 2LPT at z = 19, fixed amplitude, to z = 0. A converged run of the
# code (3x force grid, PCS + interlacing 2, no COLA, 100 logarithmic steps) and the settings of the item 59 study:
#   base           COLA, CIC, 1x grid, 20 steps (the baseline of the study; its error is the 1x force grid)
#   grid_2x        COLA, CIC, 2x grid, 20 steps (the accurate recommendation)
#   tsc_il2_dec    COLA, TSC + interlacing 2 + window deconvolution, 1x grid, 20 steps (the cheap recommendation)
#   nufft          COLA, the NUFFT force, 1x grid, 20 steps
#   fastpm_2x      FastPM stepping without COLA, CIC, 2x grid, 40 steps
#   lap_4pt        COLA, 1x grid, the discrete_4pt Laplacian (the 5-point stencil; was broken before)
#   early_calib    COLA, 1x grid, the lattice-calibrated Green's function until z = 10
#
# Checks, with P(k) at z = 0 (PCS, interlaced, on a 128^3 grid):
#   1. the converged run against its stored P(k) (pm_settings_reference.json) to 0.5% for k <= k_Ny/2
#   2. P_case / P_converged - 1 at k_Ny/8, k_Ny/4, k_Ny/2 against the stored values to TOL (default 0.005)
#   3. the physics of the study: grid_2x, tsc_il2_dec, nufft and fastpm_2x within 5% of the converged run at
#      k_Ny/2 and the baseline more than 10% low there (the 1x force grid); lap_4pt within 3% of the baseline
# The stored values are from an MPI run with 2 ranks and 1 thread; other layouts change P(k) only by round-off.
#
# Usage: python3 check_pm_settings.py <path/to/nbody> <outdir> [--write-reference]
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#   TOL    : tolerance of check 2 (default 0.005)
# About 1 minute (most of it the converged run). Needs numpy.
#==========================================================================
import json
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
write_reference = "--write-reference" in sys.argv
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
TOL = float(os.environ.get("TOL", "0.005"))
REFFILE = os.path.join(here, "pm_settings_reference.json")

N, BOX = 64, 200.0
KNY = np.pi * N / BOX
KPTS = {"kNy/8": KNY / 8, "kNy/4": KNY / 4, "kNy/2": KNY / 2}
common = dict(cases.BASE)
common.update(all_parameters_must_be_in_file=False, particle_Npart_1D=N, ic_nmesh=N, force_nmesh=N,
              simulation_boxsize=BOX, ic_initial_redshift=19.0, ic_fix_amplitude=True, ic_random_seed=4321,
              output_redshifts=[0.0], output_particles=False, timestep_nsteps=[20], timestep_method="Quinn",
              pofk=True, pofk_nmesh=2 * N, pofk_interlacing=True, pofk_subtract_shotnoise=True,
              pofk_density_assignment_method="PCS", particle_allocation_factor=2.0)
NOCOLA = dict(simulation_use_cola=False)
runs = {
    "converged": dict(NOCOLA, force_nmesh=3 * N, force_density_assignment_method="PCS", force_interlacing_order=2,
                      timestep_nsteps=[100], timestep_scalefactor_spacing="logarithmic"),
    "base": {},
    "grid_2x": dict(force_nmesh=2 * N),
    "tsc_il2_dec": dict(force_density_assignment_method="TSC", force_interlacing_order=2,
                        force_deconvolve_window=True),
    "nufft": dict(force_density_method="nufft"),
    "fastpm_2x": dict(NOCOLA, timestep_method="FastPM", force_nmesh=2 * N, timestep_nsteps=[40]),
    "lap_4pt": dict(force_greens_function_kernel="discrete_4pt"),
    "early_calib": dict(force_lattice_greens_function=True, force_lattice_greens_until_redshift=10.0,
                        ic_lattice_offset="half_force_cell"),
}

env = dict(os.environ, OMP_NUM_THREADS="1")
pk = {}
for name, overrides in runs.items():
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    params = dict(common)
    params.update(overrides)
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    with open(os.path.join(d, "log.txt"), "w") as log:
        if subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=log, env=env).returncode:
            sys.exit(f"Run '{name}' failed, see {d}/log.txt")
    data = np.loadtxt(os.path.join(d, "output", "snapshot_Reg_z0.000", "pofk_z0.000.txt"))
    pk[name] = (data[:, 0], data[:, 1])

kc, Pc = pk["converged"]
ratios = {name: {kn: float(np.interp(np.log(kt), np.log(pk[name][0]), pk[name][1] / Pc) - 1) for kn, kt in KPTS.items()}
          for name in runs if name != "converged"}
m = (kc > 0) & (kc <= KNY / 2)

if write_reference:
    with open(REFFILE, "w") as f:
        json.dump(dict(k=kc[m].tolist(), P_converged=Pc[m].tolist(), ratios=ratios), f, indent=1)
    print("Wrote", REFFILE)
    for name, r in ratios.items():
        print(f"  {name:12s}", "  ".join(f"{kn}: {100 * v:+6.2f}%" for kn, v in r.items()))
    sys.exit(0)

ref = json.load(open(REFFILE))
fails = []
# 1. the converged run against its stored P(k)
dev = np.max(np.abs(np.interp(np.log(ref["k"]), np.log(kc[m]), Pc[m]) / np.array(ref["P_converged"]) - 1))
print(f"converged run vs stored P(k), k <= k_Ny/2: max |dev| = {100 * dev:.3f}% (tolerance 0.5%)")
if dev > 0.005:
    fails.append("converged P(k)")
# 2. each setting against its stored error
for name, r in ratios.items():
    line = []
    for kn, v in r.items():
        e = ref["ratios"][name][kn]
        line.append(f"{kn}: {100 * v:+6.2f}% (stored {100 * e:+6.2f}%)")
        if abs(v - e) > TOL:
            fails.append(f"{name} at {kn}")
    print(f"  {name:12s}", "  ".join(line))
# 3. the physics
for name in ("grid_2x", "tsc_il2_dec", "nufft", "fastpm_2x"):
    if abs(ratios[name]["kNy/2"]) > 0.05:
        fails.append(f"{name}: more than 5% from the converged run at k_Ny/2")
if ratios["base"]["kNy/2"] > -0.10:
    fails.append("base: the 1x grid should lose more than 10% at k_Ny/2")
if abs(ratios["lap_4pt"]["kNy/2"] - ratios["base"]["kNy/2"]) > 0.03:
    fails.append("lap_4pt: more than 3% from the baseline at k_Ny/2")
if fails:
    sys.exit("FAILED: " + "; ".join(fails))
print("All PM settings checks passed")
