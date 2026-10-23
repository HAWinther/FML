#!/usr/bin/env python3
#==========================================================================
# Check of the forces when the force grid is finer than the particle grid
#
# With a regular particle lattice (the initial conditions) and a force grid finer
# than it, the lattice puts power just below the Nyquist frequency of the grid. The
# continuous gradient kernel (D = ik) is large there and gives spurious forces that
# change the growth of individual large-scale modes by tens of percent. The fiducial
# kernel is therefore discrete_2pt (sin(k dx)/dx) when force_nmesh > particle_Npart_1D.
#
# Runs a linear (tiny amplitude) simulation, 64^3 particles in 1000 Mpc/h from z = 19 to 0,
# with a 128^3 force grid with the fiducial and with the continuous kernel, and compares
# the growth of the displacement field mode by mode with linear theory.
#
# With GRID_FACTOR=3 the fiducial kernel is still far better than the continuous one (6.9% vs 30%
# scatter at k = 0.05 h/Mpc) but CIC assignment then leaves some lattice effects (+2% bias at
# k = 0.05 h/Mpc, so the check fails); TSC assignment removes them (-0.1% bias, 1.9% scatter).
#
# Usage: python3 check_force_grid.py <path/to/nbody> <outdir>
#   MPIRUN      : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#   GRID_FACTOR : force_nmesh / particle_Npart_1D (default 2)
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

N, BOX, ZINI = 64, 1000.0, 19.0
FACTOR = int(os.environ.get("GRID_FACTOR", "2"))
common = dict(cases.BASE)
common.update(particle_Npart_1D=N, ic_nmesh=N, force_nmesh=FACTOR * N, pofk=False, simulation_boxsize=BOX,
              simulation_use_cola=False, ic_initial_redshift=ZINI, output_redshifts=[ZINI, 0.0],
              timestep_nsteps=[0, 50], timestep_method="FastPM", output_particles=True, output_fileformat="GADGET",
              ic_fix_amplitude=True, ic_random_seed=99, ic_sigma8_normalization=True, ic_sigma8=0.02)
runs = {"fiducial": dict(force_gradient_kernel="fiducial"), "continuous": dict(force_gradient_kernel="continuous")}

env = dict(os.environ, OMP_NUM_THREADS="1")
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


def read_positions(d, z):
    """Positions (in [0,1)) sorted by ID from the gadget files"""
    pos, ids = [], []
    for f in sorted(glob.glob(os.path.join(d, "output", f"snapshot_Reg_z{z:.3f}", f"gadget_z{z:.3f}.*"))):
        b = open(f, "rb").read()
        n = int(np.frombuffer(b[4:28], np.uint32)[1])
        box = np.frombuffer(b[4 + 4 * 6 + 8 * 6 + 16 + 8 + 24 + 8:4 + 4 * 6 + 8 * 6 + 16 + 8 + 24 + 16], np.float64)[0]
        o = 4 + 256 + 8
        pos.append(np.frombuffer(b[o:o + 12 * n], np.float32).reshape(n, 3) / box)
        o += 12 * n + 8 + 12 * n + 8
        idsize = (len(b) - o - 4) // n
        ids.append(np.frombuffer(b[o:o + idsize * n], np.uint32 if idsize == 4 else np.uint64))
    return np.concatenate(pos)[np.argsort(np.concatenate(ids))]


def displacement_fourier(x_ini, x):
    q = (np.round(x_ini * N) % N).astype(int)
    dx = x - q / N
    dx -= np.round(dx)
    grid = np.zeros((3, N, N, N))
    for c in range(3):
        grid[c][q[:, 0], q[:, 1], q[:, 2]] = dx[:, c]
    return np.fft.rfftn(grid, axes=(1, 2, 3))


kx = np.fft.fftfreq(N) * N
kz = np.fft.rfftfreq(N) * N
K = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
edges = [0.5, 1.5, 2.5, 4.5, 8.5]

print(f"Growth of the displacement (z = 19 -> 0) relative to linear theory, force grid = {FACTOR} x particle grid:")
print("    k (h/Mpc)       " + "".join(f"{2 * np.pi / BOX * b:>16.3f}" for b in edges[1:]))
result = {}
for name in runs:
    d = os.path.join(outdir, name)
    x_ini, x0 = read_positions(d, ZINI), read_positions(d, 0.0)
    psi_ini, psi0 = displacement_fourier(x_ini, x_ini), displacement_fourier(x_ini, x0)
    g = np.loadtxt(os.path.join(d, "output", "gravitymodel_Reg_k0.001.txt"))
    growth = np.interp(1.0, g[:, 0], g[:, 2]) / np.interp(1 / (1 + ZINI), g[:, 0], g[:, 2])
    mean, scatter = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (K > lo) & (K <= hi)
        per_mode = np.sum(psi0[:, s] * np.conj(psi_ini[:, s]), axis=0).real / np.sum(np.abs(psi_ini[:, s]) ** 2, axis=0)
        mean.append(np.mean(per_mode) / growth - 1)
        scatter.append(np.std(per_mode) / growth)
    result[name] = (np.array(mean), np.array(scatter))
    print(f"    {name:12} mean  " + "".join(f"{100 * m:+15.2f}%" for m in mean))
    print(f"    {'':12} scatter" + "".join(f"{100 * s:15.2f}%" for s in scatter))

ok = np.max(np.abs(result["fiducial"][0])) < 0.01 and np.max(result["fiducial"][1]) < 0.02
print(("PASS" if ok else "FAIL") + f" With the fiducial kernel the large-scale growth is right to "
      f"{100 * np.max(np.abs(result['fiducial'][0])):.2f}% (scatter between modes < "
      f"{100 * np.max(result['fiducial'][1]):.2f}%); with the continuous kernel the scatter is up to "
      f"{100 * np.max(result['continuous'][1]):.0f}%")
sys.exit(0 if ok else 1)
