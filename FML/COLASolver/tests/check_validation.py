#!/usr/bin/env python3
#==========================================================================
# Check that invalid analysis settings are rejected when the parameters are read ("Invalid parameters"), before
# the simulation runs, instead of crashing at the first output:
#  - pofk = true without pofk_nmesh (was optional with default 0: SIGFPE at the first output)
#  - a grid size or bin count of 0
#  - an unknown fof_so_definitions entry and bad fof_massfunction_* settings (this validation was compiled out
#    with USE_HDF5=true)
#
# Usage: python3 check_validation.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#==========================================================================
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()

common = dict(particle_Npart_1D=16, force_nmesh=16, ic_nmesh=16, output_redshifts=[0.0], timestep_nsteps=[2])


def run(name, overrides, remove=()):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    p = dict(cases.BASE, **dict(common, **overrides))
    for k in remove:
        p.pop(k, None)
    p["all_parameters_must_be_in_file"] = False
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in p.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


fails = 0


def rejected(name, overrides, message, remove=()):
    global fails
    rc, out = run(name, overrides, remove)
    # The run must stop with the message, before the initial conditions are made
    ok = rc != 0 and message in out and "# Create initial conditions" not in out
    print(("[PASS] " if ok else "[FAIL] ") + f"{name}: rejected when reading the parameters ({message})")
    if not ok:
        fails += 1
        print(out[-2000:])


rejected("pofk_without_nmesh", dict(pofk=True), "pofk_nmesh", remove=["pofk_nmesh"])
rejected("pofk_nmesh_zero", dict(pofk=True, pofk_nmesh=0), "pofk_nmesh must be > 0")
rejected("bispectrum_nbins_zero", dict(bispectrum=True, bispectrum_nmesh=16, bispectrum_nbins=0,
                                       bispectrum_kmin=0.01, bispectrum_kmax=0.1),
         "bispectrum_nmesh and bispectrum_nbins must be > 0")
rejected("fof_so_unknown", dict(fof=True, fof_so_definitions=["200x"]), "unknown definition")
rejected("fof_massfunction_bins", dict(fof=True, fof_massfunction_nbins=0, fof_massfunction_Mmin=1e16,
                                       fof_massfunction_Mmax=1e10), "fof_massfunction_nbins must be > 0")

print("All validation checks passed" if fails == 0 else f"{fails} validation check(s) FAILED")
sys.exit(1 if fails else 0)
