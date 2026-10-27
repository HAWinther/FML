#!/usr/bin/env python3
#==========================================================================
# Checks of the initial conditions (about 1 minute, needs numpy)
#
#  1. COLA frame velocities: with 0 steps the particles written at the initial redshift must have the
#     same velocities with and without COLA (the COLA frame velocity is added back at output):
#     - CDM + baryon particles, linear 0.4 eV neutrinos (scale-dependent growth), scale-independent COLA
#     - one species with the same cosmology (control)
#     - the N-body gauge (scale-dependent growth rate of the transfer functions) with COLA, 4000 Mpc/h box
#  2. Massive neutrinos without ic_type_of_input = transferinfofile give a clear error (not a segfault)
#  3. Reading particles (ic_random_field_type = read_particles):
#     - exact LPT reconstruction from an offset lattice (COLASolver's own fiducial offset with a 2x force grid,
#       and half_force_cell with a force grid equal to the particle grid): the run continued from the file must
#       give the P(k) of the original run
#     - LPT reconstruction from the density field (PCS) with COLA: P(k) close to the original run
#     - a file with another BoxSize or h is an error
#  4. A warning when the input P(k) does not reach the largest k of the IC grid
#
# Usage: python3 check_initial_conditions.py <path/to/nbody> <outdir>
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


def run(name, overrides, expected_error=None):
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
    if expected_error is not None:
        check(res.returncode != 0 and expected_error in text and "Segmentation" not in text,
              f"{name}: fails with '{expected_error}'")
    elif res.returncode != 0:
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


def pofk(d, z):
    return np.loadtxt(os.path.join(d, "output", f"pofk_Reg_cb_z{z}.txt"))


ZINI = 20.0
at_zini = dict(output_redshifts=[ZINI], timestep_nsteps=[0], pofk=False)
snap_zini = "output/snapshot_Reg_z20.000/gadget_z20.000"

#==========================================================================
# 1. Velocities with and without COLA at the initial redshift
#==========================================================================
nu04 = dict(cosmology_OmegaMNu=0.0095,
            cosmology_OmegaCDM=0.259,
            ic_type_of_input="transferinfofile",
            ic_input_filename=cases.INPUT + "/transfer_infofile_lcdm_nu0.4_zmax200.txt",
            ic_input_redshift=ZINI,
            **at_zini)
for label, extra in [("two_species", dict(particle_baryons=True)), ("one_species", dict())]:
    v = {}
    for cola in [True, False]:
        d, _ = run(f"vel_{label}_cola{int(cola)}", dict(nu04, simulation_use_cola=cola, **extra))
        ids, pos, v[cola] = read_gadget(os.path.join(d, snap_zini))
    err = np.sqrt(np.mean((v[True] - v[False])**2) / np.mean(v[False]**2))
    check(err < 1e-4, f"{label}, 0.4 eV linear neutrinos: velocities with COLA = without COLA (rms rel. diff {err:.2e})")

# N-body gauge: the difference f_nbg(k) - f(0) matters on the largest scales, so compare the velocities averaged
# over 4^3 cells of the 4000 Mpc/h box
BOX = 4000.0
nbg = dict(simulation_boxsize=BOX,
           ic_fix_amplitude=True,
           ic_type_of_input="transferinfofile",
           ic_input_filename=cases.INPUT + "/nbody_gauge_lcdm/nbgauge_transferinfo.txt",
           ic_input_redshift=ZINI,
           force_nbody_gauge=True,
           force_nbody_gauge_filename=cases.INPUT + "/nbody_gauge_lcdm/nbgauge_extra.txt",
           **at_zini)
vmean = {}
for cola in [True, False]:
    d, _ = run(f"vel_nbody_gauge_cola{int(cola)}", dict(nbg, simulation_use_cola=cola))
    ids, pos, vel = read_gadget(os.path.join(d, snap_zini))
    cell = np.floor(pos / BOX * 4).astype(int) % 4
    index = (cell[:, 0] * 4 + cell[:, 1]) * 4 + cell[:, 2]
    counts = np.bincount(index, minlength=64)
    vmean[cola] = np.array([np.bincount(index, weights=vel[:, i], minlength=64) / counts for i in range(3)])
err = np.sqrt(np.mean((vmean[True] - vmean[False])**2) / np.mean(vmean[False]**2))
check(err < 1e-3, f"N-body gauge: large-scale velocities with COLA = without COLA (rms rel. diff {err:.2e})")

#==========================================================================
# 2. Massive neutrinos need the transfer functions
#==========================================================================
run("neutrinos_powerspectrum", dict(cosmology_OmegaMNu=0.0048, cosmology_OmegaCDM=0.2637, **at_zini),
    expected_error="cosmology_OmegaMNu > 0 needs ic_type_of_input = transferinfofile")

#==========================================================================
# 3. Reading particles
#==========================================================================
continued = dict(output_redshifts=[1.0, 0.0], timestep_nsteps=[3, 3])
read = dict(ic_random_field_type="read_particles",
            ic_reconstruct_assigment_method="PCS",
            ic_reconstruct_interlacing=True,
            ic_reconstruct_smoothing_filter="sharpk",
            ic_reconstruct_dimless_smoothing_scale=0.01,
            **continued)
lattices = {"fiducial_offset_force64": dict(force_nmesh=64),
            "half_force_cell": dict(ic_lattice_offset="half_force_cell"),
            "no_offset": dict()}
for label, extra in lattices.items():
    w, _ = run(f"write_{label}", dict(output_redshifts=[ZINI, 1.0, 0.0], timestep_nsteps=[0, 3, 3], **extra))
    filename = os.path.join(w, snap_zini)
    r, _ = run(f"read_exact_{label}", dict(read, ic_reconstruct_gadgetfilepath=filename, ic_reconstruct_exact=True,
                                         **{k: v for k, v in extra.items() if k != "ic_lattice_offset"}))
    for z in ["1.000", "0.000"]:
        err = np.max(np.abs(pofk(r, z)[:, 1] / pofk(w, z)[:, 1] - 1))
        check(err < 1e-4, f"exact reconstruction, lattice {label}: P(k, z = {z}) as the original run (max rel. "
              f"diff {err:.1e})")
    if label == "no_offset":
        r, _ = run("read_grid", dict(read, ic_reconstruct_gadgetfilepath=filename, ic_reconstruct_exact=False))
        ratio = pofk(r, "0.000")[:, 1] / pofk(w, "0.000")[:, 1] - 1
        check(np.max(np.abs(ratio)) < 0.025, f"reconstruction from the density field (PCS), COLA: P(k, z = 0) as the "
              f"original run to {np.max(np.abs(ratio)):.3f}")
        run("read_wrong_boxsize", dict(read, ic_reconstruct_gadgetfilepath=filename, ic_reconstruct_exact=True,
                                       simulation_boxsize=300.0), expected_error="BoxSize")
        run("read_wrong_h", dict(read, ic_reconstruct_gadgetfilepath=filename, ic_reconstruct_exact=True,
                                 cosmology_h=0.7), expected_error="HubbleParam")

# Exact reconstruction divides the velocities by f(a): with scale-dependent growth and COLA this is an error
w, _ = run("write_nu04", dict(nu04))
run("read_exact_nu04", dict(nu04, **read, ic_reconstruct_gadgetfilepath=os.path.join(w, snap_zini),
                            ic_reconstruct_exact=True), expected_error="needs scale-independent growth")

#==========================================================================
# 4. Input P(k) that does not reach the largest k of the IC grid
#==========================================================================
table = np.loadtxt(cases.INPUT + "/example_power_spectrum_cb_z0.000.txt")
short = os.path.join(outdir, "pofk_to_k0.5.txt")
np.savetxt(short, table[table[:, 0] < 0.5])
_, text = run("short_pofk", dict(ic_input_filename=short, **at_zini))
check("WARNING: the input P(k) / T(k) only goes to k" in text, "warning for an input P(k) shorter than the IC grid")
_, text = run("full_pofk", dict(at_zini))
check("WARNING: the input P(k)" not in text, "no warning for an input P(k) that covers the IC grid")

if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("All checks passed")
