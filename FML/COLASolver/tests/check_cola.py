#!/usr/bin/env python3
#==========================================================================
# Checks of COLA and the LPT growth factors (about 10 seconds, needs numpy)
#
#  1. simulation_use_scaledependent_cola = true only gives scale-dependent COLA when the growth is scale-dependent:
#     in GR it is turned off (with a message, same result as false) and with massive neutrinos it is used. Reading
#     particles with exact LPT reconstruction without COLA works with the default (true)
#
# Usage: python3 check_cola.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#==========================================================================
import os
import shlex
import struct
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = shlex.split(os.environ.get("MPIRUN", "mpirun -np 2"))
env = dict(os.environ, OMP_NUM_THREADS="1")
os.makedirs(outdir, exist_ok=True)
failures = []


def check(ok, what):
    print(("[PASS] " if ok else "[FAIL] ") + what)
    if not ok:
        failures.append(what)


def run(name, overrides):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    p = dict(cases.BASE)
    p.update(overrides)
    # A transfer info file starts with the folder of the transfer files (as in cases.write_case)
    if p["ic_type_of_input"] == "transferinfofile":
        with open(p["ic_input_filename"]) as f:
            lines = f.read().split("\n")
        folder = os.path.basename(lines[0].split()[0].rstrip("/"))
        lines[0] = os.path.join(cases.INPUT, folder) + " " + " ".join(lines[0].split()[1:])
        p["ic_input_filename"] = os.path.join(d, "transfer_infofile.txt")
        with open(p["ic_input_filename"], "w") as f:
            f.write("\n".join(lines))
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in p.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    with open(os.path.join(d, "log.txt"), "w") as log:
        res = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, env=env, stdout=log, stderr=log)
    text = open(os.path.join(d, "log.txt")).read()
    if res.returncode != 0:
        print(text[-2000:])
        raise RuntimeError(f"{name} failed, see {d}/log.txt")
    return d, text


def read_gadget(prefix):
    """All particle types of a (multi-file) GADGET snapshot, sorted by ID"""
    folder, base = os.path.split(prefix)
    rows = []
    for fn in sorted(os.listdir(folder)):
        if not fn.startswith(base + "."):
            continue
        with open(os.path.join(folder, fn), "rb") as f:
            def block():
                size = struct.unpack("<I", f.read(4))[0]
                data = f.read(size)
                assert struct.unpack("<I", f.read(4))[0] == size
                return data
            header = block()
            count = sum(struct.unpack("<6I", header[:24]))
            pos = np.frombuffer(block(), dtype="<f4").reshape(count, 3)
            vel = np.frombuffer(block(), dtype="<f4").reshape(count, 3)
            ids = block()
            ids = np.frombuffer(ids, dtype="<i8" if len(ids) == 8 * count else "<i4")
            rows.append((ids, pos, vel))
    ids, pos, vel = (np.concatenate([r[j] for r in rows]) for j in range(3))
    order = np.argsort(ids)
    return ids[order], pos[order].astype(float), vel[order].astype(float)


def table(d, name):
    return np.loadtxt(os.path.join(d, "output", name))


def local_fit(x, y, x0, npts=15, degree=4):
    """Value and derivative at x0 of a polynomial fit to the 2 npts + 1 points around x0 (the tables of the code have
    6 digits: a spline through them gives the derivative only to ~1e-3)"""
    i0 = np.argmin(np.abs(x - x0))
    sel = slice(max(i0 - npts, 0), i0 + npts + 1)
    poly = np.polyfit(x[sel] - x0, y[sel], degree)
    return np.polyval(poly, 0.0), np.polyval(np.polyder(poly), 0.0)


ZINI = 20.0
snap = "output/snapshot_Reg_z{0:.3f}/gadget_z{0:.3f}"

#==========================================================================
# 1. Scale-dependent COLA only with scale-dependent growth
#==========================================================================
short = dict(output_redshifts=[0.0], timestep_nsteps=[2], output_particles=False)
d_on, text_on = run("gr_scaledep_true", dict(short, simulation_use_scaledependent_cola=True))
d_off, text_off = run("gr_scaledep_false", dict(short, simulation_use_scaledependent_cola=False))
check("the growth is scale-independent: using standard COLA" in text_on and "Scaledependent COLA" not in text_on,
      "GR: simulation_use_scaledependent_cola = true uses standard COLA (with a message)")
pk_on, pk_off = (np.loadtxt(os.path.join(d, "output", "pofk_Reg_cb_z0.000.txt")) for d in [d_on, d_off])
check(np.array_equal(pk_on, pk_off), "GR: simulation_use_scaledependent_cola = true gives the same P(k) as false")
nu04 = dict(cosmology_OmegaMNu=0.0095,
            cosmology_OmegaCDM=0.259,
            ic_type_of_input="transferinfofile",
            ic_input_filename=cases.INPUT + "/transfer_infofile_lcdm_nu0.4_zmax200.txt",
            ic_input_redshift=ZINI)
_, text = run("nu_scaledep_true", dict(short, simulation_use_scaledependent_cola=True, **nu04))
check("Scaledependent COLA" in text and "using standard COLA" not in text,
      "massive neutrinos: simulation_use_scaledependent_cola = true uses scale-dependent COLA")
w, _ = run("write_zini", dict(output_redshifts=[ZINI], timestep_nsteps=[0], pofk=False))
try:
    _, text = run("read_exact_nocola", dict(simulation_use_cola=False,
                                            simulation_use_scaledependent_cola=True,
                                            ic_random_field_type="read_particles",
                                            ic_reconstruct_gadgetfilepath=os.path.join(w, snap.format(ZINI)),
                                            ic_reconstruct_exact=True,
                                            ic_reconstruct_assigment_method="CIC",
                                            ic_reconstruct_interlacing=False,
                                            ic_reconstruct_smoothing_filter="sharpk",
                                            ic_reconstruct_dimless_smoothing_scale=0.01,
                                            output_redshifts=[0.0],
                                            timestep_nsteps=[2],
                                            output_particles=False))
except RuntimeError:
    text = ""
check("Doing exact reconstruction" in text, "exact LPT reconstruction without COLA (simulation_use_scaledependent_cola "
      "= true)")

if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("All checks passed")
