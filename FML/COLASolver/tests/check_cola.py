#!/usr/bin/env python3
#==========================================================================
# Checks of COLA and the LPT growth factors (about 10 seconds, needs numpy and scipy)
#
#  1. simulation_use_scaledependent_cola = true only gives scale-dependent COLA when the growth is scale-dependent:
#     in GR it is turned off (with a message, same result as false) and with massive neutrinos it is used. Reading
#     particles with exact LPT reconstruction without COLA works with the default (true)
#  2. simulation_enforce_LPT_trajectories = true: the particles follow 2LPT. The Lagrangian position and the 1LPT and
#     2LPT displacements of each particle are solved for from its positions at z = 20, 1 and 0, and its velocities at
#     z = 20, 1 and 0 must be the 2LPT velocities a^2 H (dD1/dlna Psi1 + dD2/dlna Psi2) of the growth factors (the COLA
#     kick of the LPT force was done without the PM kick: the velocities stayed close to the initial ones)
#  3. DGP 2LPT growth factor: D2/D1^2 at z = 0 (H0 rc = 0.5, 1, 2, with screening; and H0 rc = 1 without) against an
#     independent solution of the growth equations
#        D1'' + (2 + dlogH/dloga) D1' = 1.5 OmegaM(a) GeffOverG D1
#        D2'' + (2 + dlogH/dloga) D2' = 1.5 OmegaM(a) GeffOverG (D2 - (1 - eps) D1^2)
#     with eps = 2 (rc H0)^2 OmegaM / (9 a^3 beta^3 GeffOverG) the nonlinear (Vainshtein) source, only with screening
#     (the code multiplied D2 by 1 - eps as well, and also without screening: D2 was 0.5-0.7% too small with
#     screening, 2% without)
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
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

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

#==========================================================================
# 2. Enforcing the LPT trajectories with COLA
#==========================================================================
d, _ = run("enforce_LPT", dict(simulation_enforce_LPT_trajectories=True,
                               output_redshifts=[ZINI, 1.0, 0.0],
                               timestep_nsteps=[0, 3, 3],
                               pofk=False))
box = cases.BASE["simulation_boxsize"]
growth = table(d, "gravitymodel_Reg_k0.0001.txt")
cosmo = table(d, "cosmology_Reg.txt")
loga = np.log(growth[:, 0])
D1 = lambda lna, n=0: local_fit(loga, growth[:, 2], lna)[n]  # noqa: E731
D2 = lambda lna, n=0: local_fit(loga, growth[:, 4], lna)[n]  # noqa: E731
E = lambda lna: local_fit(np.log(cosmo[:, 0]), cosmo[:, 1], lna)[0]  # noqa: E731
snaps = {z: read_gadget(os.path.join(d, snap.format(z))) for z in [ZINI, 1.0, 0.0]}
ids, x0, _ = snaps[ZINI]
# Positions relative to z = 20 (unwrapped): x(z) - x(zini) = (D1(a) - D1(ai)) Psi1 + (D2(a) - D2(ai)) Psi2
dx = {z: (snaps[z][1] - x0 + box / 2) % box - box / 2 for z in [1.0, 0.0]}
a = {z: 1.0 / (1.0 + z) for z in [ZINI, 1.0, 0.0]}
M = np.array([[D1(np.log(a[z])) - D1(np.log(a[ZINI])), D2(np.log(a[z])) - D2(np.log(a[ZINI]))] for z in [1.0, 0.0]])
Minv = np.linalg.inv(M)
psi1 = Minv[0, 0] * dx[1.0] + Minv[0, 1] * dx[0.0]
psi2 = Minv[1, 0] * dx[1.0] + Minv[1, 1] * dx[0.0]
for z in [ZINI, 1.0, 0.0]:
    la = np.log(a[z])
    # GADGET velocities: v = 100 sqrt(a) E(a) dx/dlna (x in Mpc/h)
    v_lpt = 100.0 * np.sqrt(a[z]) * E(la) * (D1(la, 1) * psi1 + D2(la, 1) * psi2)
    v = snaps[z][2]
    err = np.sqrt(np.mean((v - v_lpt)**2) / np.mean(v_lpt**2))
    check(err < 1e-4, f"simulation_enforce_LPT_trajectories: the velocities at z = {z:g} are the 2LPT velocities (rms "
          f"rel. diff {err:.1e}, rms v = {np.sqrt(np.mean(v**2)):.1f} km/s)")

#==========================================================================
# 3. DGP: D2 against an independent solution of the growth equations
#==========================================================================
dgp = {k: v for k, v in cases.CASES["lightcone_dgp"].items() if not k.startswith("plc_") and k != "lightcone"}
dgp.update(output_redshifts=[ZINI], timestep_nsteps=[0], pofk=False, output_particles=False)
for rc, screening in [(0.5, True), (1.0, True), (2.0, True), (1.0, False)]:
    d, _ = run(f"dgp_rc{rc}_screening{int(screening)}",
               dict(dgp, gravity_model_dgp_rcH0overc=rc, gravity_model_screening=screening))
    growth = table(d, "gravitymodel_Reg_k0.0001.txt")
    cosmo = table(d, "cosmology_Reg.txt")
    lna_c = np.log(cosmo[:, 0])
    E, dlogH, OmegaM_of_a = (CubicSpline(lna_c, cosmo[:, i]) for i in [1, 2, 3])
    OmegaM = cases.BASE["cosmology_OmegaCDM"] + cases.BASE["cosmology_Omegab"]

    def rhs(lna, y):
        a = np.exp(lna)
        beta = 1.0 + 2.0 * rc * E(lna) * (1.0 + dlogH(lna) / 3.0)
        geff = 1.0 + 1.0 / (3.0 * beta)
        eps = 2.0 * rc**2 * OmegaM / (9.0 * a**3 * beta**3 * geff) if screening else 0.0
        s = 1.5 * OmegaM_of_a(lna) * geff
        d1, dd1, d2, dd2 = y
        return [dd1, s * d1 - (2.0 + dlogH(lna)) * dd1, dd2, s * (d2 - (1.0 - eps) * d1 * d1) - (2.0 + dlogH(lna)) * dd2]

    # Start from the growth factors of the code at a = 0.02 (the DGP terms are negligible before)
    loga = np.log(growth[:, 0])
    lna0 = np.log(0.02)
    y0 = [*local_fit(loga, growth[:, 2], lna0), *local_fit(loga, growth[:, 4], lna0)]
    sol = solve_ivp(rhs, [lna0, 0.0], y0, rtol=1e-10, atol=1e-14)
    d1, d2 = sol.y[0, -1], sol.y[2, -1]
    D1c, D2c = local_fit(loga, growth[:, 2], 0.0)[0], local_fit(loga, growth[:, 4], 0.0)[0]
    err1 = D1c / d1 - 1
    err2 = (D2c / D1c**2) / (d2 / d1**2) - 1
    check(abs(err1) < 1e-4 and abs(err2) < 2e-4,
          f"DGP H0 rc = {rc}, screening = {screening}: D2/D1^2 at z = 0 = {D2c / D1c**2:.5f} (independent solution "
          f"{d2 / d1**2:.5f}, rel. diff {err2:.1e}; D1 {err1:.1e})")

if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("All checks passed")
