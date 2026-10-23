#!/usr/bin/env python3
#==========================================================================
# Physics check of the N-body gauge source (force_nbody_gauge)
#
# Makes the N-body gauge input with CLASS (scripts/make_nbody_gauge_transfer.py) and
# runs a large box (8000 Mpc/h, 128^3, z = 49 -> 0) with a tiny amplitude so that
# everything is linear. Then compares the growth of the largest modes,
# P(k,z=0)/P(k,zini), to the N-body gauge growth of CLASS. We run twice with the same
# N-body gauge initial conditions: with the source and with the source set to zero.
#  - The change from adding the source must be what linear theory predicts (this is
#    independent of the resolution of the PM force, which also slows the growth a bit)
#  - With the source the growth on the largest scales agrees with CLASS as well as a standard
#    (Newtonian) run agrees with its own linear theory. The PM force on a grid the size of the
#    particle grid makes the growth a bit too slow (0.5% at k = 1e-3 h/Mpc, more at higher k),
#    so we compare to a third run with the standard setup
#
# Usage: python3 check_nbody_gauge.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
# Needs numpy, scipy and classy (pip install classy)
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
os.makedirs(outdir, exist_ok=True)

ZINI = 49.0
BOX = 8000.0
N = 128
AS = 2.215e-9 * 1e-4  # tiny amplitude so that the evolution is linear
NSTEPS = 100

common = dict(cases.BASE)
common.update(
    simulation_name="NbGauge",
    simulation_boxsize=BOX,
    simulation_use_cola=False,
    particle_Npart_1D=N,
    ic_nmesh=N,
    force_nmesh=N,
    pofk_nmesh=N,
    cosmology_As=AS,
    ic_initial_redshift=ZINI,
    ic_input_redshift=ZINI,
    ic_fix_amplitude=True,
    ic_random_seed=2024,
    output_redshifts=[0.0],
    output_particles=False,
    pofk=False,
    timestep_nsteps=[NSTEPS],
    timestep_method="FastPM",
    timestep_scalefactor_spacing="logarithmic",
    force_linear_massive_neutrinos=False,
)

# Make the N-body gauge input with CLASS from the parameters we run with
inputdir = os.path.join(outdir, "nbgauge_input")
paramfile = os.path.join(outdir, "cosmology.lua")
with open(paramfile, "w") as f:
    for k, v in common.items():
        f.write(f"{k} = {cases.lua_value(v)}\n")
script = os.path.join(here, "..", "scripts", "make_nbody_gauge_transfer.py")
if not os.path.exists(os.path.join(inputdir, "nbgauge_extra.txt")):
    subprocess.run([sys.executable, script, paramfile, "--outdir", inputdir, "--nmesh", str(N)], check=True)

# A copy of the source table with T_extra = 0 (so that the IC are exactly the same)
zero_source = os.path.join(inputdir, "nbgauge_extra_zero.txt")
with open(os.path.join(inputdir, "nbgauge_extra.txt")) as fin, open(zero_source, "w") as fout:
    nline = 0
    for line in fin:
        if line.startswith("#"):
            fout.write(line)
            continue
        nline += 1
        if nline <= 3:
            fout.write(line)
        else:
            cols = line.split()
            fout.write(cols[0] + " " + " ".join("0.0" for _ in cols[1:]) + "\n")

nbgauge_ic = dict(ic_type_of_input="transferinfofile",
                  ic_type_of_input_fileformat="CAMB",
                  ic_input_filename=inputdir + "/nbgauge_transferinfo.txt")
runs = {
    "with_source": dict(nbgauge_ic, force_nbody_gauge=True, force_nbody_gauge_filename=inputdir + "/nbgauge_extra.txt"),
    "without_source": dict(nbgauge_ic, force_nbody_gauge=True, force_nbody_gauge_filename=zero_source),
    # Standard setup: P(k) at z = 0 scaled back with the growth factor of the code (tiny sigma8)
    "standard": dict(force_nbody_gauge=False, ic_type_of_input="powerspectrum", ic_input_redshift=0.0,
                     ic_sigma8_normalization=True, ic_sigma8=0.83e-2),
}
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
        ok = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=log, env=env).returncode
    if ok != 0:
        sys.exit(f"Run '{name}' failed, see {d}/log.txt")


def growth(name):
    """P(k,z=0)/P(k,zini) of the particles (the P(k) computed every step on the force grid)"""
    out = os.path.join(outdir, name, "output")
    p0 = np.loadtxt(os.path.join(out, "pofk_NbGauge_cb_z0.000.txt"))
    pini = np.loadtxt(os.path.join(out, f"pofk_NbGauge_cb_z{ZINI:.3f}.txt"))
    return p0[:, 0], p0[:, 1] / pini[:, 1]


# CLASS: (T_cb(k,0) / T_cb(k,zini))^2 from the N-body gauge transfer files
info = open(os.path.join(inputdir, "nbgauge_transferinfo.txt")).read().split()
files = dict(zip([float(z) for z in info[3::2]], info[2::2]))


def Tcb(z):
    d = np.loadtxt(os.path.join(inputdir, files[min(files, key=lambda x: abs(x - z))]))
    fb = common["cosmology_Omegab"] / (common["cosmology_Omegab"] + common["cosmology_OmegaCDM"])
    return d[:, 0], (1 - fb) * d[:, 1] + fb * d[:, 2]


k_class, T0 = Tcb(0.0)
_, Tini = Tcb(ZINI)

# Linear theory prediction of the change from the source: evolve
#   delta'' + calH delta' = 1.5 calH^2 Omega_cb(a) (delta + s delta_extra)
# with s = 0, 1 (in log(a)) using the background and source in the tables
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

lines = [x for x in open(os.path.join(inputdir, "nbgauge_extra.txt")) if not x.startswith("#")]
nk_tab, nz_tab = map(int, lines[0].split())
z_tab = np.array(lines[1].split(), float)
E_tab = np.array(lines[2].split(), float)
tab = np.array([x.split() for x in lines[3:3 + nk_tab]], float)
k_tab, Textra = tab[:, 0], tab[:, 1:]
Om_cb0 = common["cosmology_Omegab"] + common["cosmology_OmegaCDM"]


def predicted_change(kk):
    # delta_extra(k,z) / delta_cb(k,zini) and delta_cb(k,z) from the tables
    j = int(np.argmin(np.abs(k_tab - kk)))
    lna = np.log(1 / (1 + z_tab))[::-1]
    sel = lna >= np.log(1 / (1 + ZINI)) - 1e-9
    x = lna[sel]
    E = E_tab[::-1][sel]
    a = np.exp(x)
    Om = CubicSpline(x, 1.5 * Om_cb0 / a**3 / E**2)
    dlnH = CubicSpline(x, np.gradient(np.log(a * E), x, edge_order=2))
    kc, T0c = Tcb(0.0)
    jc = int(np.argmin(np.abs(kc - kk)))
    Tini_k = Tini[jc]
    R = CubicSpline(x, Textra[j][::-1][sel] / Tini_k)
    # initial growth rate from the transfer files
    _, Tm = Tcb(z_tab[np.argmin(np.abs(z_tab - ZINI)) + 1])
    zm = z_tab[np.argmin(np.abs(z_tab - ZINI)) + 1]
    f_ini = (np.log(Tm[jc]) - np.log(Tini_k)) / (np.log(1 / (1 + zm)) - np.log(1 / (1 + ZINI)))
    res = []
    for s in [0.0, 1.0]:
        sol = solve_ivp(lambda xx, y: [y[1], -(1 + dlnH(xx)) * y[1] + Om(xx) * (y[0] + s * R(xx))],
                        [x[0], x[-1]], [1.0, f_ini], rtol=1e-10, atol=1e-14)
        res.append(sol.y[0, -1])
    return (res[1] / res[0]) ** 2


k, g_with = growth("with_source")
_, g_without = growth("without_source")
g_class = np.interp(np.log(k), np.log(k_class), (T0 / Tini) ** 2)

print(f"\nGrowth of P(k) from z = {ZINI:g} to 0, box = {BOX:g} Mpc/h, {N}^3 particles:")
print("   k (h/Mpc)   with/CLASS   without/CLASS   with/without   predicted with/without")
large = k < 0.004
# The P(k) bins (width 2pi/Box centered on the multiples of 2pi/Box) average over the modes in them so
# we average the prediction over the modes (weighted by their power)
kf = 2 * np.pi / BOX
n = np.arange(-6, 7)
nmag = np.sqrt(n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).flatten()
P_class = lambda kk: np.interp(np.log(kk), np.log(k_class), T0**2 * k_class ** (common["cosmology_ns"] + 3))  # noqa
pred = np.ones_like(k)
g_class_modes = np.interp(np.log(k), np.log(k_class), (T0 / Tini) ** 2)
for i in range(len(k)):
    modes = nmag[(nmag >= i + 0.5) & (nmag < i + 1.5)] * kf
    if len(modes) == 0 or i >= 5:
        continue
    kk = np.unique(modes)
    weight = np.array([np.sum(modes == x) for x in kk]) * P_class(kk)
    pred[i] = np.sum(weight * np.array([predicted_change(x) for x in kk])) / np.sum(weight)
    g_class_modes[i] = np.sum(weight * np.interp(np.log(kk), np.log(k_class), (T0 / Tini) ** 2)) / np.sum(weight)
g_class = g_class_modes
for i in np.where(large)[0]:
    print(f"   {k[i]:.2e}    {100 * (g_with[i] / g_class[i] - 1):+.3f}%      {100 * (g_without[i] / g_class[i] - 1):+.3f}%"
          f"        {100 * (g_with[i] / g_without[i] - 1):+.3f}%        {100 * (pred[i] - 1):+.3f}%")

results = []


def check(ok, what):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + what)


change = g_with[large] / g_without[large]
err = np.max(np.abs(change / pred[large] - 1))
effect = np.max(np.abs(pred[large] - 1))
check(err < 0.05 * effect,
      f"The source changes P(k) as predicted by linear theory (max difference {100 * err:.4f}% for an effect of "
      f"up to {100 * effect:.3f}%)")
# The standard run compared to the linear growth factor of the code
_, g_standard = growth("standard")
grav = np.loadtxt(os.path.join(outdir, "standard", "output", "gravitymodel_NbGauge_k0.001.txt"))
D = lambda a: np.interp(a, grav[:, 0], grav[:, 2])  # noqa
g_standard = g_standard / (D(1.0) / D(1 / (1 + ZINI))) ** 2
print("\n   k (h/Mpc)   standard run / linear theory (the error from the PM force resolution)")
for i in np.where(large)[0]:
    print(f"   {k[i]:.2e}    {100 * (g_standard[i] - 1):+.3f}%")
sel = slice(0, 3)
err_with = np.max(np.abs(g_with[sel] / g_class[sel] - g_standard[sel]))
err_without = np.max(np.abs(g_without[sel] / g_class[sel] - g_standard[sel]))
check(err_with < err_without / 3,
      f"Compared to the standard run the N-body gauge run follows CLASS to {100 * err_with:.3f}% on the largest "
      f"scales ({100 * err_without:.3f}% without the source)")
sys.exit(0 if all(results) else 1)
