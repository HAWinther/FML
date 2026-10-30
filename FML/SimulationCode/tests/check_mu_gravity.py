#!/usr/bin/env python3
#==========================================================================
# Physics check of gravity_model = mu (phenomenological mu(a,k), no screening)
#
# Linear-amplitude simulations (the input P(k) scaled down by 1e4; 64^3 particles, Box = 256 Mpc/h, 2x force grid,
# fixed amplitude, the same IC from GR growth at z = 19) for GR, mu = 1 + mu0 Omega_DE(a)/Omega_DE (mu0 = 0.5) and the
# Bertschinger-Zukin mu(a,k) (beta1 = 4/3, lambda1 = 10 Mpc/h, s = 4), without COLA (Quinn, 200 steps) and with
# scale-dependent COLA (30 steps). Checks P_mu / P_GR at z = 0 against (D_mu / D_GR)^2 from an independent solution of
# D'' + (2 + dlnH/dlna) D' = 1.5 Omega_M(a) mu(a,k) D (with H(a) and Omega_M(a) from the code's cosmology output):
# within 0.1% for k <= 0.1 h/Mpc (the force resolution of the PM grid gives up to ~1% at k = 0.6 h/Mpc)
#
# Usage: python3 check_mu_gravity.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
# Needs numpy and scipy
#==========================================================================
import os
import subprocess
import sys

import numpy as np
from scipy.integrate import solve_ivp

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
os.makedirs(outdir, exist_ok=True)

# Linear amplitude: the example P(k) / 1e4
pk = np.loadtxt(os.path.join(here, "..", "input", "example_power_spectrum_cb_z0.000.txt"))
pkfile = os.path.join(outdir, "pk_linear_amplitude.txt")
np.savetxt(pkfile, np.column_stack([pk[:, 0], pk[:, 1] * 1e-4]), header="k (h/Mpc)  P(k) (Mpc/h)^3 / 1e4")

common = dict(particle_Npart_1D=64, ic_nmesh=64, force_nmesh=128, pofk_nmesh=128, simulation_boxsize=256.0,
              output_redshifts=[0.0], output_particles=False, ic_fix_amplitude=True, ic_random_seed=4321,
              ic_initial_redshift=19.0, ic_use_gravity_model_GR=True, ic_input_filename=pkfile)
quinn = dict(simulation_use_cola=False, timestep_method="Quinn", timestep_nsteps=[200],
             timestep_scalefactor_spacing="logarithmic")
cola = dict(simulation_use_cola=True, simulation_use_scaledependent_cola=True, timestep_method="Quinn",
            timestep_nsteps=[30], timestep_scalefactor_spacing="linear")
models = {
    "GR": dict(gravity_model="GR"),
    "DE": dict(gravity_model="mu", gravity_model_mu_parametrization="DE", gravity_model_mu_mu0=0.5),
    "BZ": dict(gravity_model="mu", gravity_model_mu_parametrization="BZ", gravity_model_mu_BZ_params=[4 / 3, 10.0, 4.0]),
}
runs = {}
for m, mx in models.items():
    runs[m] = dict(quinn, **mx)
    runs[m + "_cola"] = dict(cola, **mx)

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


def pofk(name):
    return np.loadtxt(os.path.join(outdir, name, "output", "snapshot_Reg_z0.000", "pofk_z0.000.txt"))


# Independent linear growth with the background of the code
c = np.loadtxt(os.path.join(outdir, "GR", "output", "cosmology_Reg.txt"))
la = np.log(c[:, 0])
dlogH = lambda x: np.interp(x, la, c[:, 2])
OmegaM = lambda x: np.interp(x, la, c[:, 3])
OmegaDE = lambda x: np.interp(x, la, c[:, 8])
mus = {
    "GR": lambda a, k: 1.0,
    "DE": lambda a, k: 1.0 + 0.5 * OmegaDE(np.log(a)) / OmegaDE(0.0),
    "BZ": lambda a, k: (1 + 4 / 3 * 100 * k * k * a**4) / (1 + 100 * k * k * a**4),
}


def growth(mu, k, zini=19.0):
    rhs = lambda x, y, m: [y[1], -(2 + dlogH(x)) * y[1] + 1.5 * OmegaM(x) * m(np.exp(x), k) * y[0]]
    s = solve_ivp(lambda x, y: rhs(x, y, mus["GR"]), [np.log(1e-3), -np.log(1 + zini)], [1e-3, 1e-3], rtol=1e-10, atol=1e-14)
    y0 = s.y[:, -1]
    s = solve_ivp(lambda x, y: rhs(x, y, mu), [-np.log(1 + zini), 0.0], y0, rtol=1e-10, atol=1e-14)
    return s.y[0, -1] / y0[0]


k = pofk("GR")[:, 0]
ks = [0.03, 0.06, 0.1, 0.2, 0.4, 0.6]
idx = [int(np.argmin(np.abs(k - x))) for x in ks]
failed = 0
print("P_mu / P_GR at z = 0 relative to (D_mu / D_GR)^2 - 1")
print(f"{'':10}" + "".join(f"{k[i]:>9.3f}" for i in idx))
for m in ["DE", "BZ"]:
    pred = np.array([(growth(mus[m], k[i]) / growth(mus["GR"], k[i])) ** 2 for i in idx])
    for suffix in ["", "_cola"]:
        r = pofk(m + suffix)[:, 1] / pofk("GR" + suffix)[:, 1]
        dev = np.array([r[i] for i in idx]) / pred - 1
        print(f"{m + suffix:10}" + "".join(f"{100 * d:+8.3f}%" for d in dev))
        large = np.array(ks) <= 0.1
        if np.max(np.abs(dev[large])) > 1e-3:
            print(f"FAIL: {m + suffix} differs by more than 0.1% from linear theory for k <= 0.1 h/Mpc")
            failed += 1
print("All mu checks passed" if failed == 0 else f"{failed} mu check(s) FAILED")
sys.exit(1 if failed else 0)
