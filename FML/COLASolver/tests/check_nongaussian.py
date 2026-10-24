#!/usr/bin/env python3
#==========================================================================
# Physics check of the non-gaussian (local fNL) initial conditions
#
# Runs four small simulations (64^3, Box = 2000 Mpc/h) with the same seed and
# fixed amplitude: the gaussian IC and non-gaussian IC with fNL = 0, +300, -300.
# Then checks that:
#  1. fNL = 0 gives the same P(k) as the gaussian IC (same random numbers)
#  2. The change in the bispectrum at the initial redshift is linear in fNL:
#     B(+fNL) - B(0) = -(B(-fNL) - B(0)) to first order
#  3. Positive fNL gives a larger reduced bispectrum Q on large scales
#
# Usage: python3 check_nongaussian.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
# Needs numpy
#==========================================================================
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
FNL = 300.0

common = dict(
    particle_Npart_1D=64,
    ic_nmesh=64,
    force_nmesh=64,
    pofk_nmesh=64,
    simulation_boxsize=2000.0,
    bispectrum=True,
    bispectrum_nmesh=64,
    bispectrum_nbins=8,
    output_redshifts=[20.0, 0.0],
    timestep_nsteps=[0, 5],
    output_particles=False,
    ic_fix_amplitude=True,
    ic_random_seed=777,
)
nongaussian = dict(ic_random_field_type="nongaussian", ic_fnl_type="local", ic_fnl_redshift=20.0)
runs = {
    "gaussian": {},
    "fnl0": dict(nongaussian, ic_fnl=0.0),
    "fnl_plus": dict(nongaussian, ic_fnl=+FNL),
    "fnl_minus": dict(nongaussian, ic_fnl=-FNL),
}

env = dict(os.environ, OMP_NUM_THREADS="1")
for name, overrides in runs.items():
    params = dict(cases.BASE)
    params.update(common)
    params.update(overrides)
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    with open(os.path.join(d, "log.txt"), "w") as log:
        ok = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=log, env=env).returncode
    if ok != 0:
        sys.exit(f"Run '{name}' failed, see {d}/log.txt")


def pofk(name, z):
    return np.loadtxt(os.path.join(outdir, name, "output", f"pofk_Reg_cb_z{z}.txt"))


def bispectrum(name, z="20.000"):
    return np.loadtxt(os.path.join(outdir, name, "output", f"snapshot_Reg_z{z}", f"bispectrum_z{z}.txt"))


results = []


def check(ok, what):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + what)


for z in ["20.000", "0.000"]:
    g, n0 = pofk("gaussian", z), pofk("fnl0", z)
    diff = np.max(np.abs(n0[:, 1] / g[:, 1] - 1))
    check(diff < 1e-3, f"P(k) for fNL = 0 equals the gaussian IC at z = {z} (max rel. difference {diff:.1e})")

b0, bp, bm = bispectrum("fnl0"), bispectrum("fnl_plus"), bispectrum("fnl_minus")
mask = (b0[:, 8] > 0) & (b0[:, 3] > 0)
dp, dm = bp[mask, 6] - b0[mask, 6], bm[mask, 6] - b0[mask, 6]
opposite = np.mean(np.sign(dp) == -np.sign(dm))
nonlinear = np.median(np.abs(dp + dm) / np.abs(dp - dm))
check(opposite > 0.95 and nonlinear < 0.05,
      f"Bispectrum response is linear in fNL (opposite sign in {100 * opposite:.0f}% of bins, "
      f"median non-linearity {nonlinear:.3f})")
large = b0[mask, 3] < 0.02
qm, q0, qp = bm[mask, 7][large].mean(), b0[mask, 7][large].mean(), bp[mask, 7][large].mean()
check(qm < q0 < qp, f"Reduced bispectrum on large scales increases with fNL (Q = {qm:.1f}, {q0:.1f}, {qp:.1f} "
      f"for fNL = {-FNL:.0f}, 0, {FNL:.0f})")

sys.exit(0 if all(results) else 1)
