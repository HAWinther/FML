#!/usr/bin/env python3
#==========================================================================
# Check of the FoF/SO halo catalogue and the mass functions
#
# Runs a small COLA simulation (32^3, Box = 128 Mpc/h) with FoF and SO (200m, 200c) and checks that:
#  1. The SO radius is the OUTERMOST radius where the mean enclosed density is above Delta: N_Delta of the
#     catalogue agrees with a brute-force count from the particles around the SO center (the sphere used to stop
#     at the first particle below the threshold, often 1-2 particles from the center even for large halos).
#     Halos whose density is nowhere above Delta (e.g. 200c at PM resolution) still get 1-2 particles
#  2. dn/dlnM in massfunc*.txt is the number of halos in [Mmin e^(i dlogM), Mmin e^((i+1) dlogM)) (halos just
#     below Mmin are not in the first bin) and n(>M) the number of halos with mass >= the bin center written in
#     the first column; Mmin is between two halo masses so that halos lie just below it
#  3. A negative pofk_multipole_ellmax is rejected
#
# Usage: python3 check_halos.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
# Needs numpy
#==========================================================================
import os
import struct
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()

box, npart = 128.0, 32
OmegaCB = cases.BASE["cosmology_OmegaCDM"] + cases.BASE["cosmology_Omegab"]
mpart = 2.7754e11 * OmegaCB * box**3 / npart**3
nbins, dlogM = 8, 0.5
Mmin = 12.5 * mpart
Mmax = float(Mmin * np.exp(nbins * dlogM))

common = dict(
    particle_Npart_1D=npart,
    simulation_boxsize=box,
    output_redshifts=[0.0],
    timestep_nsteps=[6],
    fof=True,
    fof_nmin_per_halo=10,
    fof_so_definitions=["200m", "200c"],
    fof_buffer_length_mpch=6.0,
    fof_massfunction_nbins=nbins,
    fof_massfunction_Mmin=Mmin,
    fof_massfunction_Mmax=Mmax,
)


def run(name, overrides):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    p = dict(cases.BASE, **dict(common, **overrides))
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in p.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(os.path.join(d, "log.txt"), "w") as log:
        rc = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=subprocess.STDOUT,
                            env=env).returncode
    return d, rc


fails = 0


def check(ok, what):
    global fails
    print(("[PASS] " if ok else "[FAIL] ") + what)
    if not ok:
        fails += 1


d, rc = run("halos", {})
if rc != 0:
    sys.exit(f"The run failed, see {d}/log.txt")
snap = os.path.join(d, "output", "snapshot_Reg_z0.000")
halos = np.loadtxt(os.path.join(snap, "halos_z0.000.txt"), ndmin=2)
nfof, mfof = halos[:, 0], halos[:, 1]
# Columns: np, mass, pos, vel, vel_rms, SO pos, vel, vel_rms, truncated, overlap, then M, R, N per definition
so = {"200m": 22, "200c": 25}

# 1. SO: brute-force outermost crossing around the SO centers (the catalogue is written with 6 digits and the
# particles in single precision, so allow a difference of one particle or 2%)
def read_gadget_pos(prefix):
    pos = []
    for i in range(16):
        filename = f"{prefix}.{i}"
        if not os.path.exists(filename):
            break
        with open(filename, "rb") as f:
            def block():
                size = struct.unpack("<I", f.read(4))[0]
                data = f.read(size)
                assert struct.unpack("<I", f.read(4))[0] == size
                return data
            count = struct.unpack("<6I", block()[:24])[1]
            pos.append(np.frombuffer(block(), dtype="<f4").reshape(count, 3).astype(float))
    return np.concatenate(pos)


pos = read_gadget_pos(os.path.join(snap, "gadget_z0.000"))
check(len(pos) == npart**3, "read all particles")
nmean = len(pos) / box**3
Delta = {"200m": 200.0, "200c": 200.0 / OmegaCB}
big = nfof >= 50
check(big.sum() >= 5, f"enough halos with >= 50 FoF particles ({big.sum()})")
for name, c in so.items():
    nbrute = np.zeros(len(halos), dtype=int)
    for i, h in enumerate(halos):
        d = pos - h[11:14]
        d -= box * np.round(d / box)
        r = np.sort(np.sqrt((d * d).sum(axis=1)))
        r = r[r <= common["fof_buffer_length_mpch"]]
        above = np.nonzero(np.arange(1, len(r) + 1) / (4 * np.pi / 3 * r**3) >= Delta[name] * nmean)[0]
        nbrute[i] = above[-1] + 1 if len(above) else 0
    ncat = halos[:, c + 2]
    bad = np.abs(ncat - nbrute) > np.maximum(1, 0.02 * nbrute)
    worst = np.argmax(np.abs(ncat - nbrute))
    check(not bad.any(), f"N_{name} is the outermost crossing ({bad.sum()} of {len(halos)} halos differ; worst: "
          f"N_FoF = {nfof[worst]:.0f}, catalogue {ncat[worst]:.0f}, brute force {nbrute[worst]})")
    check(np.allclose(halos[:, c], halos[:, c + 2] * mpart, rtol=1e-3),
          f"M_{name} = N_{name} times the particle mass")
    print(f"   N_{name} / N_FoF of the halos with >= 50 FoF particles: {np.round(ncat[big] / nfof[big], 2)}")

# 2. Mass functions
V = box**3
centers = Mmin * np.exp(dlogM * (np.arange(nbins) + 0.5))
edges = Mmin * np.exp(dlogM * np.arange(nbins + 1))
check(((mfof < Mmin) & (mfof > Mmin * np.exp(-dlogM))).any(), "some halos just below Mmin")
for fname, masses in [("massfunc_z0.000.txt", mfof), ("massfunc_so200m_z0.000.txt", halos[:, so["200m"]])]:
    mf = np.loadtxt(os.path.join(snap, fname), ndmin=2)
    dn = np.histogram(masses, bins=edges)[0] / (V * dlogM)
    ngt = np.array([(masses >= m).sum() for m in centers]) / V
    check(np.allclose(mf[:, 0], centers, rtol=1e-5), f"{fname}: masses are the bin centers")
    check(np.allclose(mf[:, 2], dn, rtol=1e-4, atol=0), f"{fname}: dn/dlnM counts the halos in the bin")
    check(np.allclose(mf[:, 1], ngt, rtol=1e-4, atol=0), f"{fname}: n(>M) counts the halos above the bin center")

# 3. A negative pofk_multipole_ellmax is an error
d, rc = run("ellmax", dict(pofk_multipole=True, pofk_multipole_ellmax=-2))
log = open(os.path.join(d, "log.txt")).read()
check(rc != 0 and "pofk_multipole_ellmax must be >= 0" in log, "pofk_multipole_ellmax < 0 is rejected")

print("All halo checks passed" if fails == 0 else f"{fails} halo check(s) FAILED")
sys.exit(1 if fails else 0)
