#!/usr/bin/env python3
#==========================================================================
# Checks of the lightcone (32^3 particles in 256 Mpc/h)
#
#  cola     : the lightcone positions with COLA (GR) and scale-dependent COLA (a k-dependent mu(a,k)) with 10 steps
#             compared to a 200 step run of the same model (median distance to the nearest particle). The lightcone drifts the particles from the start
#             of the step with their true velocity, so the COLA drift (the LPT displacement over the step) must
#             come after it (it was counted twice: median 0.52 Mpc/h with 10 steps, 0.09 when fixed)
#  batches  : an octant with replicas where a step has more crossings than particles per task. Without
#             plc_output_in_batches all crossings must be written (particles were lost when the full buffer was
#             written to the same file, and with 4 tasks it crashed), with batches the same particles must be
#             written and every batch must have one file per task (NumFiles in the headers). The first step is
#             small (this used to switch off the batches for the rest of the run)
#  zfinish  : plc_z_finish inside a (long) step: the number of lightcone particles just above r(z_finish)
#             compared with a 200 step run (no particles were kept between r(z_finish) and the chord
#             approximation of r(a), here 89 - 113 Mpc/h)
#  coverage : without replicas an observer at (0.5, 0.5, 0.5) and R(z_init) > 0.5 boxsize gives an incomplete
#             octant: must be an error
#  healpix  : (if compiled with USE_HEALPIX, needs healpy) the onion density maps of an octant (no replicas)
#             compared with the full sky (replicas): delta = 0 outside the octant (pixel centers), zero mean and
#             the same delta as the full sky map (up to the mean) inside, and the C_ell are those of the full sky
#             map restricted to the octant divided by fsky (the unobserved pixels were delta = -1, the observed
#             shifted by +0.875 and the C_ell dominated by the mask). Without USE_HEALPIX:
#             plc_make_onion_density_maps = true must fail when reading the parameters
#
# Usage: python3 check_lightcone.py <path/to/nbody> <outdir> [test ...]
#   MPIRUN   : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#   MPIRUN4  : how to launch with 4 tasks for the batches test (default "mpirun --oversubscribe -np 4";
#              MPIRUN4="" to skip)
# Needs numpy and scipy (and healpy for the healpix test)
#==========================================================================
import glob
import os
import re
import subprocess
import sys
import warnings

import numpy as np
from scipy.spatial import cKDTree

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
tests = sys.argv[3:] or ["cola", "batches", "zfinish", "coverage", "healpix"]
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
mpirun4 = os.environ.get("MPIRUN4", "mpirun --oversubscribe -np 4").split()
env = dict(os.environ, OMP_NUM_THREADS="1")

BOX, N = 256.0, 32
NBAR = N**3 / BOX**3
LIGHTCONE = dict(lightcone=True, plc_pos_observer=[0.0, 0.0, 0.0], plc_boundary_mpch=20.0, plc_z_init=0.08,
                 plc_z_finish=0.0, plc_use_replicas=False, plc_ndim_rep=0, plc_output_gadgetfile=False,
                 plc_output_asciifile=True, plc_output_in_batches=False, plc_make_onion_density_maps=False,
                 pofk=False, output_particles=False, output_redshifts=[0.0], timestep_nsteps=[10])


def run(name, overrides, launcher=None, must_succeed=True):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    params = dict(cases.BASE)
    params.update(LIGHTCONE)
    params.update(overrides)
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {cases.lua_value(v)}\n")
    launcher = mpirun if launcher is None else launcher
    with open(os.path.join(d, "log.txt"), "w") as log:
        try:
            rc = subprocess.run(launcher + [exe, "parameterfile.lua"], cwd=d, stdout=log, stderr=log, env=env,
                                timeout=600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    if rc and must_succeed:
        sys.exit(f"Run '{name}' failed, see {d}/log.txt")
    return d, rc


def lightcone_ascii(d):
    """The positions (Mpc/h) of all the lightcone particles in the ascii files"""
    pos = [np.zeros((0, 3))]
    for f in glob.glob(os.path.join(d, "output", "lightcone_Reg", "lightcone_cb_ascii_step*")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # empty files
            a = np.loadtxt(f, comments="#", ndmin=2)
        if a.size:
            pos.append(a[:, :3])
    return np.concatenate(pos)


def gadget_header(fn):
    """npart (type 1), NumFiles and the positions of a (lightcone) GADGET file"""
    b = open(fn, "rb").read()
    n = int(np.frombuffer(b[4:28], np.int32)[1])
    numfiles = int(np.frombuffer(b[4 + 124:4 + 128], np.int32)[0])
    pos = np.frombuffer(b[4 + 256 + 8:4 + 256 + 8 + 12 * n], np.float32).reshape(n, 3)
    return n, numfiles, pos


results = []


def report(ok, msg):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + msg, flush=True)


if "cola" in tests:
    # Scale-dependent COLA is only used with scale-dependent growth: a k-dependent mu(a,k), with its own reference
    bz = dict(gravity_model="mu", gravity_model_mu_parametrization="BZ",
              gravity_model_mu_BZ_params=[4.0 / 3.0, 10.0, 4.0])
    trees = {}
    for label, model in [("", {}), ("_bz", bz)]:
        ref, _ = run("cola_ref" + label, dict(model, timestep_nsteps=[200]))
        trees[label] = cKDTree(lightcone_ascii(ref))
    for name, over in [("cola", {}),
                       ("cola_scaledependent", dict(bz, simulation_use_scaledependent_cola=True)),
                       ("nocola", dict(simulation_use_cola=False))]:
        d, _ = run(name, over)
        if name == "cola_scaledependent" and "Scaledependent COLA" not in open(os.path.join(d, "log.txt")).read():
            report(False, f"{name}: scale-dependent COLA was not used")
        tree = trees["_bz" if name == "cola_scaledependent" else ""]
        x = lightcone_ascii(d)
        dist = np.median(tree.query(x)[0])
        if name == "nocola":
            print(f"     (no COLA, 10 steps: median distance to the 200 step lightcone {dist:.3f} Mpc/h)")
            continue
        report(dist < 0.2, f"{name} 10 steps: median distance of the lightcone particles to those of the 200 step "
               f"run {dist:.3f} Mpc/h (< 0.2; {len(x)} vs {tree.n} particles)")

if "batches" in tests:
    # An octant with replicas (more particles on the tasks with small x). Steps of 0.095 in a: in 'small_first'
    # the lightcone starts at a = 0.9035, just before the step boundary at 0.9048, so the first step has few
    # crossings and the second ~0.9 box volumes (~0.9 x 32^3, more than the particles of a task). In 'big' it
    # starts at a = 0.85 (~3.6 box volumes in two steps)
    batch = dict(plc_use_replicas=True, plc_ndim_rep=0, plc_output_gadgetfile=True, plc_output_asciifile=False)
    zinit = {"small_first": 1 / 0.9035 - 1, "big": 1 / 0.85 - 1}
    counts = {}
    runs = [("nobatch", "small_first", False, mpirun), ("batch", "small_first", True, mpirun)]
    if mpirun4:
        runs += [("nobatch_np4", "small_first", False, mpirun4), ("batch_np4", "small_first", True, mpirun4),
                 ("big_batch_np4", "big", True, mpirun4)]
    for name, config, in_batches, launcher in runs:
        d, rc = run("batches_" + name, dict(batch, plc_z_init=zinit[config], plc_output_in_batches=in_batches),
                    launcher, must_succeed=False)
        if rc:
            report(False, f"batches {name}: the run failed (see {d}/log.txt)")
            continue
        # The files of each step (and batch): <prefix>.<task>
        groups = {}
        for f in glob.glob(os.path.join(d, "output", "lightcone_Reg", "lightcone_cb_gadget_step*")):
            prefix, task = f.rsplit(".", 1)
            groups.setdefault(prefix, []).append(int(task))
        total, ok_files, ntasks = 0, True, set()
        for prefix, tasks in sorted(groups.items()):
            headers = [gadget_header(f"{prefix}.{t}") for t in tasks]
            total += sum(h[0] for h in headers)
            numfiles = {h[1] for h in headers}
            ntasks.add(len(tasks))
            if numfiles != {len(tasks)} or sorted(tasks) != list(range(len(tasks))):
                ok_files = False
                print(f"     {os.path.basename(prefix)}: files from tasks {sorted(tasks)}, NumFiles {numfiles}")
        nbatches = sum(1 for p in groups if "_batch" in p)
        report(ok_files and len(ntasks) == 1,
               f"batches {name}: {len(groups)} sets of files ({nbatches} batches), each with one file per task "
               f"({sorted(ntasks)}) and NumFiles = the number of files")
        # All the particles within r(z_init) cross the lightcone: compare with the mean density
        log = open(os.path.join(d, "log.txt")).read()
        rmax = max(float(m) for m in re.findall(r"Size of the slice: [-\d.e+]+ -> ([-\d.e+]+) Mpc/h", log))
        expected = 1 / 8 * 4 / 3 * np.pi * rmax**3 * NBAR
        report(abs(total / expected - 1) < 0.03,
               f"batches {name}: {total} particles within r(z_init) = {rmax:.1f} Mpc/h, expected {expected:.0f} "
               f"from the mean density (to 3%)")
        counts[name] = total
    for name in ["batch", "nobatch_np4", "batch_np4"]:
        if name in counts and "nobatch" in counts:
            rel = abs(counts[name] / counts["nobatch"] - 1)
            report(rel < 1e-3, f"batches {name}: {counts[name]} particles, as without batches with 2 tasks "
                   f"({counts['nobatch']}) to {rel:.1e} (< 1e-3)")

if "zfinish" in tests:
    # 3 steps (the last from a = 0.68 to 1), full sky from z = 0.08 to z_finish = 0.03 (inside the last step),
    # compared with 200 steps
    zfin = dict(plc_z_init=0.08, plc_z_finish=0.03, plc_use_replicas=True, plc_ndim_rep=3)
    r = {}
    for name, nsteps in [("zfinish", 3), ("zfinish_ref", 200)]:
        d, _ = run(name, dict(zfin, timestep_nsteps=[nsteps]))
        r[name] = np.sqrt((lightcone_ascii(d)**2).sum(1))
    head = open(glob.glob(os.path.join(outdir, "zfinish", "output", "lightcone_Reg", "lightcone_cb_ascii_step*"))[0]
                ).read(2000)
    rfin, rini = (float(v) for v in re.search(r"distances is in \[([-\d.e+]+) -> ([-\d.e+]+)\]", head).groups())
    for lo, hi in [(rfin, rfin + 10), (rfin + 10, rfin + 30), (rfin, rini)]:
        n, nref = (np.sum((x >= lo) & (x < hi)) for x in (r["zfinish"], r["zfinish_ref"]))
        report(abs(n / nref - 1) < 0.05, f"zfinish: {n} particles with {lo:.1f} <= r < {hi:.1f} Mpc/h, {nref} with "
               f"200 steps (to 5%; r(z_finish) = {rfin:.1f} Mpc/h)")

if "coverage" in tests:
    d, rc = run("coverage", dict(plc_pos_observer=[0.5, 0.5, 0.5]), must_succeed=False)
    log = open(os.path.join(d, "log.txt")).read()
    report(rc != 0 and "will not be complete" in log and "Drifting particles" not in log,
           "coverage: no replicas, observer at (0.5, 0.5, 0.5) and R(z_init) = 0.91 boxsize > 0.5 is an error "
           f"before the first lightcone step (exit code {rc})")

if "healpix" in tests:
    features = subprocess.run(mpirun + [exe, "--features"], capture_output=True, text=True, env=env).stdout
    maps = dict(plc_make_onion_density_maps=True, plc_nside=16, plc_da_maps=0.03, plc_use_chunkpix=False)
    if "healpix" not in features:
        d, rc = run("healpix_missing", maps, must_succeed=False)
        log = open(os.path.join(d, "log.txt")).read()
        report(rc != 0 and "Invalid parameters: plc_make_onion_density_maps = true needs Healpix" in log and
               "Initializing lightcone" not in log,
               "healpix: without USE_HEALPIX plc_make_onion_density_maps = true fails when reading the parameters")
    else:
        import healpy as hp
        octant, _ = run("healpix_octant", maps)
        full, _ = run("healpix_fullsky", dict(maps, plc_use_replicas=True, plc_ndim_rep=3))
        lc_o, lc_f = (os.path.join(d, "output", "lightcone_Reg") for d in (octant, full))
        for f in sorted(glob.glob(os.path.join(lc_o, "OnionMap_delta_cb_imap*"))):
            imap = f.rsplit("imap", 1)[1]
            m, mfull = hp.read_map(f), hp.read_map(os.path.join(lc_f, f"OnionMap_delta_cb_imap{imap}"))
            nside = hp.npix2nside(len(m))
            n = np.array(hp.pix2vec(nside, np.arange(len(m)))).T
            inside = np.all(n > 1e-10, axis=1)  # the pixels with centers inside the octant
            # The full sky map restricted to the octant (the same particles there): the C_ell of the octant run
            # must be the pseudo-C_ell of this divided by fsky
            masked = np.where(inside, mfull - np.mean(mfull[inside]), 0.0)
            cl_masked = hp.anafast(masked, lmax=2 * nside - 1, iter=1) / np.mean(inside)
            cfile = os.path.join(lc_o, f"Cell_delta_cb_imap{imap}.txt")
            cl = np.loadtxt(cfile)[:, 1]
            fsky = float(re.search(r"fsky = ([\d.e+-]+)", open(cfile).readline()).group(1))
            diff = (m - mfull)[inside]
            ratio = cl[1:] / cl_masked[1:]
            report(np.all(m[~inside] == 0) and abs(np.mean(m[inside])) < 1e-5 and np.std(diff) < 1e-5 and
                   abs(fsky / np.mean(inside) - 1) < 1e-4 and np.all(np.abs(ratio - 1) < 0.01),
                   f"healpix imap {imap}: delta outside the octant {np.abs(m[~inside]).max():.2g} (0), mean inside "
                   f"{np.mean(m[inside]):+.1e} (0), delta - delta(full sky) inside {np.mean(diff):+.4f} +- "
                   f"{np.std(diff):.1e} (constant), fsky = {fsky:.4f} ({np.mean(inside):.4f}), C_ell / C_ell(full sky "
                   f"map in the octant) = {ratio.min():.4f} - {ratio.max():.4f} (1 to 1%)")

print("PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
