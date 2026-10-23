#!/usr/bin/env python3
#==========================================================================
# Convergence of the time-stepping methods
#
# Runs the same simulation (64^3 particles, Box = 256 Mpc/h, z = 19 -> 0,
# no COLA unless stated) with the different time-stepping methods and a
# few number of steps, and compares P(k) at z = 0 to a reference run with
# many steps. Prints the max relative difference on large scales
# (k < 0.1 h/Mpc) and up to k = 0.5 h/Mpc.
#
# Checks (exit code 1 if one fails):
#  - FastPM and BullFrog (which follow linear growth exactly) get P(k) for
#    k < 0.1 h/Mpc right to < 1% with 10 steps
#  - All methods converge to the reference: < 1.5% for k < 0.5 h/Mpc with 40 steps
#
# The force grid is the same as the particle grid (64^3) by default. With FORCE_NMESH=128
# the errors are smaller still (e.g. BullFrog with 5 steps: 0.08% for k < 0.1 h/Mpc)
#
# Usage: python3 check_timestepping.py <path/to/nbody> <outdir> [--nref 200]
#   MPIRUN      : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#   FORCE_NMESH : the force grid (default 64)
# Needs numpy
#==========================================================================
import argparse
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("nbody")
parser.add_argument("outdir")
parser.add_argument("--nref", type=int, default=200, help="Number of steps in the reference run")
args = parser.parse_args()
exe, outdir = os.path.abspath(args.nbody), os.path.abspath(args.outdir)
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()

common = dict(
    particle_Npart_1D=64,
    ic_nmesh=64,
    force_nmesh=int(os.environ.get("FORCE_NMESH", "64")),
    pofk_nmesh=128,
    simulation_boxsize=256.0,
    simulation_use_cola=False,
    ic_initial_redshift=19.0,
    output_redshifts=[0.0],
    output_particles=False,
    ic_fix_amplitude=True,
    ic_random_seed=4321,
)

NSTEPS = [5, 10, 20, 40]
runs = {"reference": dict(timestep_method="Quinn", timestep_nsteps=[args.nref])}
for method in ["Quinn", "FastPM", "BullFrog"]:
    for n in NSTEPS:
        runs[f"{method}_{n}"] = dict(timestep_method=method, timestep_nsteps=[n])
for n in NSTEPS:
    runs[f"COLA-Tassev_{n}"] = dict(timestep_method="Tassev", timestep_nsteps=[n], simulation_use_cola=True)
    runs[f"COLA-FastPM_{n}"] = dict(timestep_method="FastPM", timestep_nsteps=[n], simulation_use_cola=True)

env = dict(os.environ, OMP_NUM_THREADS="1")
for name, overrides in runs.items():
    d = os.path.join(outdir, name)
    pofk_file = os.path.join(d, "output", "snapshot_Reg_z0.000", "pofk_z0.000.txt")
    if os.path.exists(pofk_file):
        continue
    params = dict(cases.BASE)
    params.update(common)
    params.update(overrides)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    with open(os.path.join(d, "log.txt"), "w") as log:
        ok = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=log, env=env).returncode
    if ok != 0:
        sys.exit(f"Run '{name}' failed, see {d}/log.txt")


def pofk(name):
    return np.loadtxt(os.path.join(outdir, name, "output", "snapshot_Reg_z0.000", "pofk_z0.000.txt"))


ref = pofk("reference")
k = ref[:, 0]
large, mid = k < 0.1, k < 0.5


def maxdiff(name, mask):
    return np.max(np.abs(pofk(name)[mask, 1] / ref[mask, 1] - 1))


print(f"Max |P(k)/P_ref(k) - 1| at z = 0 (reference: Quinn with {args.nref} steps)\n")
print(f"{'method':<14}" + "".join(f"{n:>8} steps" for n in NSTEPS))
for label, kmask in [("k < 0.1 h/Mpc", large), ("k < 0.5 h/Mpc", mid)]:
    print(f"\n{label}")
    for method in ["Quinn", "FastPM", "BullFrog", "COLA-Tassev", "COLA-FastPM"]:
        print(f"{method:<14}" + "".join(f"{100 * maxdiff(f'{method}_{n}', kmask):>12.2f}%" for n in NSTEPS))
print()

results = []


def check(ok, what):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + what)


for method in ["FastPM", "BullFrog"]:
    d = maxdiff(f"{method}_10", large)
    check(d < 0.01, f"{method} with 10 steps gets P(k) for k < 0.1 h/Mpc to {100 * d:.2f}% (< 1%)")
for method in ["Quinn", "FastPM", "BullFrog", "COLA-Tassev", "COLA-FastPM"]:
    d = maxdiff(f"{method}_40", mid)
    check(d < 0.015, f"{method} with 40 steps gets P(k) for k < 0.5 h/Mpc to {100 * d:.2f}% (< 1.5%)")

sys.exit(0 if all(results) else 1)
