#!/usr/bin/env python3
#==========================================================================
# Check of the linear sources (linear massive neutrinos, N-body gauge) with the early-time force
# (force_early_method)
#
# The linear sources are delta_cb(k,zini) times a transfer function ratio, added to the density field of each step,
# so delta_cb(k,zini) must have the same window as that field. It was stored from the first force evaluation, which
# uses the early-time density assignment (NUFFT: no window; interlaced: another assignment), and then combined with
# the CIC field of all the later steps.
#
# With simulation_enforce_LPT_trajectories the particles do not depend on the force, so after the switch to the
# late-time force (z < force_early_until_redshift) the density field, and so P_total(k) of every step, must be the
# same as without the early-time force. Before the fix P_total differed by up to 7e-4 (nufft) and 1.7e-3 (interlaced
# with PCS) at z = 0 for Mnu = 0.2 eV. The N-body gauge with the early-time force is also run (it must work and give
# the same P_total after the switch).
#
# Usage: python3 check_early_force.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
# Needs numpy
#==========================================================================
import glob
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
env = dict(os.environ, OMP_NUM_THREADS="1")
ZSWITCH = 5.0

fails = 0


def check(ok, what):
    global fails
    print(("[PASS] " if ok else "[FAIL] ") + what)
    if not ok:
        fails += 1


def run(name, case, early):
    # The regression case (with its transfer info file) in outdir/early/case, with the settings below added
    cases.write_case(os.path.join(outdir, early), case)
    d = os.path.join(outdir, early, case)
    p = dict(simulation_enforce_LPT_trajectories=True, pofk_every_step=True, output_particles=False,
             force_early_method=early, force_early_until_redshift=ZSWITCH, force_early_interlacing_order=2,
             force_early_density_assignment_method="PCS", all_parameters_must_be_in_file=False)
    with open(os.path.join(d, "parameterfile.lua"), "a") as f:
        for k, v in p.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    r = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:])
    check(r.returncode == 0, f"{name} runs")
    return d


def pofk_total(d):
    """P_total(k) of the steps after the switch to the late-time force, by redshift"""
    res = {}
    for f in glob.glob(os.path.join(d, "output", "pofk_Reg_total_z*.txt")):
        z = float(os.path.basename(f)[len("pofk_Reg_total_z"):-len(".txt")])
        if z < ZSWITCH:
            res[z] = np.loadtxt(f)[:, 1]
    return res


for case in ["scaledep_cola_neutrinos", "nbody_gauge"]:
    ref = pofk_total(run(case + "_none", case, "none"))
    for early in ["nufft", "interlaced"]:
        new = pofk_total(run(case + "_" + early, case, early))
        ok = len(ref) > 0 and sorted(ref) == sorted(new)
        maxdiff = max((np.abs(new[z] / ref[z] - 1.0).max() for z in ref), default=np.inf) if ok else np.inf
        check(ok and maxdiff < 1e-6,
              f"{case}, force_early_method = {early}: P_total at z < {ZSWITCH} (LPT trajectories) / without the "
              f"early-time force - 1 = {maxdiff:.1e} (z = {sorted(ref)})")

print("All early-time force checks passed" if fails == 0 else f"{fails} early-time force check(s) FAILED")
sys.exit(1 if fails else 0)
