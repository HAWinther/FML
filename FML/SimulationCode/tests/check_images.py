#!/usr/bin/env python3
#==========================================================================
# Checks of the images (images = true, src/OutputImages.h; about 1 minute, needs numpy)
#
#  1. The files: image_<field>_z<z>.png (valid PNG: chunk CRCs, zlib stream, the size) and .npy (images_npy)
#  2. velocity_divergence = f delta on large scales (the slope of theta against delta below 3 k_f, whole box)
#  3. baryons_minus_cdm (particle_baryons) is small compared to delta
#  4. neutrinos: linear (force_linear_massive_neutrinos) delta_nu / delta_cb in (0, 1] and falling with k (free
#     streaming); neutrino particles correlated with delta_cb on large scales
#  5. fifth_force: f(R) linear, with approximate screening and with the exact multigrid solver: |F_5| / |F_N| in
#     [0, 0.5) and, relative to the linear run, smaller in the dense pixels than in the empty ones (screening)
#  6. The images do not change the simulation: P(k) with and without them the same (the exact solver keeps its
#     starting point for the next step)
#  7. Clear errors: fifth_force in GR, baryons_minus_cdm without particle_baryons, an unknown field
#
# Usage: python3 check_images.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#==========================================================================
import os
import shlex
import struct
import subprocess
import sys
import zlib

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
    return d


def png_size(filename):
    """Width and height of a PNG, checking the chunk CRCs and that the image data is a valid zlib stream"""
    data = open(filename, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, size = 8, b"", None
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert zlib.crc32(kind + body) == struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]
        if kind == b"IHDR":
            size = struct.unpack(">II", body[:8])
        if kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    assert len(raw) == size[1] * (3 * size[0] + 1)
    return size


def field(d, name, z):
    return np.load(os.path.join(d, "output", f"snapshot_Reg_z{z}", f"image_{name}_z{z}.npy"))


def cross_ratio(a, b, kmax):
    """sum Re(a b*) / sum |b|^2 over the 2D modes 0 < |k| < kmax (units of k_f)"""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    k = np.fft.fftfreq(a.shape[0]) * a.shape[0]
    K = np.hypot(k[:, None], k[None, :])
    m = (K > 0) & (K < kmax)
    return np.real(np.sum(fa[m] * np.conj(fb[m]))) / np.sum(np.abs(fb[m])**2)


images = dict(all_parameters_must_be_in_file=False, images=True, images_nmesh=32, images_npy=True,
              images_slab_thickness=1.0)
OmegaM = cases.BASE["cosmology_OmegaCDM"] + cases.BASE["cosmology_Omegab"]

#==========================================================================
# 1-3. LCDM with CDM and baryon particles: density, velocity divergence, baryons - CDM
#==========================================================================
tables = dict(ic_type_of_input="tables", ic_input_filename=cases.INPUT + "/transfer_tables/lcdm_nu0.0.txt",
              ic_input_redshift=20.0)
lcdm = run("lcdm", dict(**images, **tables, particle_baryons=True,
                        images_fields=["density", "velocity_divergence", "baryons_minus_cdm"], images_npix=48))
for z in ["1.000", "0.000"]:
    for name in ["density", "velocity_divergence", "baryons_minus_cdm"]:
        png = os.path.join(lcdm, "output", f"snapshot_Reg_z{z}", f"image_{name}_z{z}.png")
        try:
            ok = png_size(png) == (48, 48) and field(lcdm, name, z).shape == (48, 48)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(e)
        check(ok, f"z = {z}: image_{name} written (valid 48x48 PNG and .npy)")
    a = 1.0 / (1.0 + float(z))
    Om = OmegaM / a**3 / (OmegaM / a**3 + 1.0 - OmegaM)
    f = Om**0.55
    delta = field(lcdm, "density", z) - 1.0
    slope = cross_ratio(field(lcdm, "velocity_divergence", z), delta, 3)
    check(abs(slope / f - 1.0) < 0.08, f"z = {z}: velocity divergence = f delta on large scales "
                                       f"(slope {slope:.3f}, f = {f:.3f})")
    diff = field(lcdm, "baryons_minus_cdm", z)
    check(np.std(diff) < 0.5 * np.std(delta),
          f"z = {z}: delta_b - delta_c small (rms {np.std(diff):.3g}, delta {np.std(delta):.3g})")

#==========================================================================
# 4. Neutrinos: linear and particles
#==========================================================================
nu = dict(cosmology_OmegaMNu=0.0095, cosmology_OmegaCDM=0.259, ic_type_of_input="tables",
          ic_input_filename=cases.INPUT + "/transfer_tables/lcdm_nu0.4.txt", ic_input_redshift=20.0)
lin = run("neutrinos_linear", dict(**images, **nu, images_fields=["density", "neutrinos"]))
delta = field(lin, "density", "0.000") - 1.0
nu_field = field(lin, "neutrinos", "0.000")
r_low, r_high = cross_ratio(nu_field, delta, 2), cross_ratio(nu_field, delta, 12)
check(0.0 < r_high < r_low <= 1.0,
      f"linear neutrinos: delta_nu / delta_cb in (0, 1] and smaller at higher k ({r_low:.3f}, {r_high:.3f})")
part = run("neutrinos_particles", dict(**images, **nu, particle_neutrinos=True, simulation_use_cola=False,
                                       force_linear_massive_neutrinos=False, images_fields=["density", "neutrinos"]))
delta = field(part, "density", "0.000") - 1.0
nu_field = field(part, "neutrinos", "0.000")
r = cross_ratio(nu_field, delta, 3)
check(0.1 < r < 1.5 and np.all(np.isfinite(nu_field)),
      f"neutrino particles: delta_nu follows delta_cb on large scales (ratio {r:.3f})")

#==========================================================================
# 5-6. The fifth force of f(R): approximate screening and the exact solver; the images change nothing
#==========================================================================
fofr = dict(gravity_model="f(R)", gravity_model_fofr_fofr0=1e-5, gravity_model_fofr_nfofr=1.0,
            gravity_model_fofr_exact_solution=False, gravity_model_screening=True,
            gravity_model_screening_efficiency=1.0, gravity_model_screening_enforce_largescale_linear=False,
            gravity_model_screening_linear_scale_hmpc=0.1, multigrid_solver_residual_convergence=1e-7,
            multigrid_nsweeps=5, multigrid_nsweeps_first_step=10, multigrid_coarse_factor=1)
ratio = {}
for label, extra in [("linear", dict(gravity_model_screening=False)), ("screening", dict()),
                     ("exact", dict(gravity_model_fofr_exact_solution=True))]:
    d = run(f"fofr_{label}", {**images, **fofr, **extra, "images_fields": ["density", "fifth_force"],
                              "images_slab_thickness": 0.25})
    ratio[label] = field(d, "fifth_force", "0.000")
    rho = field(d, "density", "0.000")
    check(ratio[label].min() >= 0.0 and ratio[label].max() < 0.5,
          f"f(R) {label}: |F_5| / |F_N| in [0, 0.5) ({ratio[label].min():.3f} - {ratio[label].max():.3f})")
    if label != "linear":
        # Screening: the fifth force relative to the linear one is smaller in the dense pixels than in the empty
        s = ratio[label] / ratio["linear"]
        dense, empty = s[rho > np.percentile(rho, 95)].mean(), s[rho < np.percentile(rho, 50)].mean()
        check(dense < 0.9 * empty, f"f(R) {label}: screened in the dense pixels (|F_5| relative to linear "
                                   f"{dense:.3f} in the densest 5%, {empty:.3f} in the empty half)")
    if label == "exact":
        without = run("fofr_exact_no_images", {**fofr, **extra})
        same = all(open(os.path.join(d, "output", f"pofk_Reg_cb_z{z}.txt")).read() ==
                   open(os.path.join(without, "output", f"pofk_Reg_cb_z{z}.txt")).read() for z in ["1.000", "0.000"])
        check(same, "f(R) exact solver: P(k) the same with and without the images (the solver state is kept)")

#==========================================================================
# 7. Errors
#==========================================================================
run("error_gr_fifth_force", dict(**images, images_fields=["fifth_force"]),
    expected_error="fifth_force needs a gravity model with a fifth force")
run("error_no_baryons", dict(**images, images_fields=["baryons_minus_cdm"]), expected_error="is not available here")
run("error_unknown", dict(**images, images_fields=["potentail"]), expected_error="is not available here")

if failures:
    print(f"{len(failures)} image checks FAILED")
    sys.exit(1)
print("All image checks passed")
