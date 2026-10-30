#!/usr/bin/env python3
#==========================================================================
# Check that invalid settings stop the run with a clear message (before the simulation runs where possible)
# instead of crashing later, hanging or silently doing something else:
#  - pofk = true without pofk_nmesh (was optional with default 0: SIGFPE at the first output); a grid size or bin
#    count of 0; an unknown fof_so_definitions entry and bad fof_massfunction_* settings (this validation was
#    compiled out with USE_HDF5=true)
#  - simulation_cola_LPT_order = 3 (not implemented: aborted after the IC); simulation_enforce_LPT_trajectories
#    with COLA turned off by cola_LPT_order = 0 (the particles never moved); Tassev with timestep_cola_nLPT = 0
#    (0/0); an unknown timestep_algorithm (was ignored); an output with 0 steps after the first (wrote the
#    previous snapshot at another redshift), also when splitting the total number of steps
#  - gravity_model_fofr_fofr0 < 0 (ODE failure), gravity_model_mbeta_params without 4 numbers (read out of
#    bounds), cosmology_ede_n = 1 (ODE failure), a Tabulated H/H0 table that does not reach z = 0 (H(a=1) != H0)
#  - exact LPT reconstruction with 2LPT from GADGET files with a number of particles that is not N^3 (went on as if
#    the particles were on the lattice of round(cbrt(N)))
#  - lightcone Healpix maps with plc_nside or plc_nside_chunks > 8192 (int Healpix_Base; npix overflowed when
#    the first map was allocated)
#  - command line: no parameterfile, an unknown option and --parameters to a file that cannot be written give
#    a non-zero exit code
# and that these now work:
#  - a GeffG(a) file with a header line (the reader looped forever)
#  - the JBD cosmology leaves the format of std::cout alone (As was printed as 0.00000000)
#
# Usage: python3 check_validation.py <path/to/nbody> <outdir>
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#==========================================================================
import os
import struct
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

exe, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
mpirun = os.environ.get("MPIRUN", "mpirun -np 2").split()
os.makedirs(outdir, exist_ok=True)

common = dict(particle_Npart_1D=16, force_nmesh=16, ic_nmesh=16, output_redshifts=[0.0], timestep_nsteps=[2])
env = dict(os.environ, OMP_NUM_THREADS="1")


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
    try:
        r = subprocess.run(mpirun + [exe, "parameterfile.lua"], cwd=d, capture_output=True, text=True, env=env,
                           timeout=300)
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT (the run did not finish in 300 s)"
    return r.returncode, r.stdout + r.stderr


fails = 0


def check(ok, what, out=""):
    global fails
    print(("[PASS] " if ok else "[FAIL] ") + what)
    if not ok:
        fails += 1
        print(out[-2000:])


def rejected(name, overrides, message, remove=(), before_ic=True):
    rc, out = run(name, overrides, remove)
    # The run must stop with the message (before the initial conditions are made if it is a parameter check)
    ok = rc > 0 and message in out and (not before_ic or "# Create initial conditions" not in out)
    check(ok, f"{name}: stops with [{message}]", out)


# Analysis settings
rejected("pofk_without_nmesh", dict(pofk=True), "pofk_nmesh", remove=["pofk_nmesh"])
rejected("pofk_nmesh_zero", dict(pofk=True, pofk_nmesh=0), "pofk_nmesh must be > 0")
rejected("bispectrum_nbins_zero", dict(bispectrum=True, bispectrum_nmesh=16, bispectrum_nbins=0,
                                       bispectrum_kmin=0.01, bispectrum_kmax=0.1),
         "bispectrum_nmesh and bispectrum_nbins must be > 0")
rejected("fof_so_unknown", dict(fof=True, fof_so_definitions=["200x"]), "unknown definition")
rejected("fof_massfunction_bins", dict(fof=True, fof_massfunction_nbins=0, fof_massfunction_Mmin=1e16,
                                       fof_massfunction_Mmax=1e10), "fof_massfunction_nbins must be > 0")

# COLA and time stepping
rejected("cola_3LPT", dict(simulation_use_cola=True, simulation_cola_LPT_order=3, ic_LPT_order=3),
         "simulation_cola_LPT_order = 3 is not implemented")
rejected("enforce_LPT_without_cola", dict(simulation_use_cola=True, simulation_cola_LPT_order=0,
                                          simulation_enforce_LPT_trajectories=True), "needs COLA")
rejected("tassev_nLPT_zero", dict(timestep_method="Tassev", timestep_cola_nLPT=0.0),
         "timestep_cola_nLPT cannot be 0")
rejected("timestep_algorithm", dict(timestep_algorithm="DKD"), "timestep_algorithm")
rejected("zero_steps_output", dict(output_redshifts=[1.0, 0.95, 0.0], timestep_nsteps=[3, 0, 3]), "has 0 steps")
rejected("zero_steps_split", dict(output_redshifts=[1.0, 0.95, 0.0], timestep_nsteps=[10]), "gives 0 steps",
         before_ic=False)

# Cosmologies and gravity models
rejected("fofr0_negative", dict(gravity_model="f(R)", gravity_model_fofr_fofr0=-1e-5), "must be > 0")
rejected("mbeta_params", dict(gravity_model="mbeta", gravity_model_mbeta_params=[0.4, 0.0, 1.0]),
         "must have 4 numbers")
rejected("ede_n1", dict(cosmology_model="EDE", cosmology_ede_fEDE=0.1, cosmology_ede_zc=3500.0,
                        cosmology_ede_n=1.0), "cosmology_ede_n > 1")
htable = os.path.join(outdir, "H_from_z0.5.txt")
with open(htable, "w") as f:
    Om = cases.BASE["cosmology_OmegaCDM"] + cases.BASE["cosmology_Omegab"]
    for i in range(200):
        z = 0.5 + 20.0 * i / 199
        f.write(f"{z} {(Om * (1 + z)**3 + 1 - Om)**0.5}\n")
rejected("tabulated_H_without_z0", dict(cosmology_model="Tabulated", cosmology_tabulated_filename=htable,
                                        cosmology_tabulated_format="H"), "must reach z = 0")

# Exact LPT reconstruction (2LPT) from particles that are not on a regular N^3 lattice: a 16^3 snapshot with one
# particle removed (Npart_1D was taken as round(cbrt(4095)) = 16 and the reconstruction went on as for a lattice)
rc, out = run("snapshot_16", dict(output_redshifts=[20.0, 19.0], timestep_nsteps=[0, 1]))
check(rc == 0, "snapshot_16 (input for the next check) runs", out)


def copy_gadget_without_one_particle(prefix_in, prefix_out):
    """Copy a (multi-file) GADGET-1 snapshot (one particle type) and remove the last particle of the first file"""
    files = sorted(f for f in os.listdir(os.path.dirname(prefix_in))
                   if f.startswith(os.path.basename(prefix_in) + "."))
    os.makedirs(os.path.dirname(prefix_out), exist_ok=True)
    for ifile, fn in enumerate(files):
        with open(os.path.join(os.path.dirname(prefix_in), fn), "rb") as f:
            blocks = []
            while True:
                size = f.read(4)
                if not size:
                    break
                blocks.append(bytearray(f.read(struct.unpack("<I", size)[0])))
                f.read(4)
        header, count = blocks[0], sum(struct.unpack("<6I", blocks[0][:24]))
        itype = [i for i, n in enumerate(struct.unpack("<6I", header[:24])) if n > 0][0]
        # npartTotal (offset 96) in all files, npart (offset 0) and the blocks in the first file
        struct.pack_into("<I", header, 96 + 4 * itype, struct.unpack_from("<I", header, 96 + 4 * itype)[0] - 1)
        if ifile == 0:
            struct.pack_into("<I", header, 4 * itype, count - 1)
            for b in range(1, len(blocks)):
                del blocks[b][len(blocks[b]) // count * (count - 1):]
        with open(prefix_out + "." + fn.split(".")[-1], "wb") as f:
            for b in blocks:
                f.write(struct.pack("<I", len(b)) + bytes(b) + struct.pack("<I", len(b)))


snap = os.path.join(outdir, "snapshot_16", "output", "snapshot_Reg_z20.000", "gadget_z20.000")
copy_gadget_without_one_particle(snap, os.path.join(outdir, "snapshot_16_minus_one", "gadget_z20.000"))
rejected("read_exact_not_a_cube", dict(ic_random_field_type="read_particles", ic_reconstruct_exact=True,
                                       ic_reconstruct_gadgetfilepath=os.path.join(outdir, "snapshot_16_minus_one",
                                                                                  "gadget_z20.000"),
                                       ic_reconstruct_assigment_method="PCS", ic_reconstruct_interlacing=True,
                                       ic_reconstruct_smoothing_filter="sharpk",
                                       ic_reconstruct_dimless_smoothing_scale=0.01),
         "is not Npart_1D^3", before_ic=False)

# Lightcone Healpix maps: Healpix_Map uses int pixel indices, so nside <= 8192 (npix = 12 nside^2 overflowed when the
# first map was allocated). Checked before the check for Healpix, so this also works without USE_HEALPIX
plc = dict(lightcone=True, plc_pos_observer=[0.0, 0.0, 0.0], plc_boundary_mpch=20.0, plc_z_init=0.05,
           plc_z_finish=0.0, plc_use_replicas=False, plc_ndim_rep=3, plc_output_gadgetfile=False,
           plc_output_asciifile=False, plc_output_in_batches=False, plc_make_onion_density_maps=True, plc_nside=16,
           plc_da_maps=0.03, plc_use_chunkpix=False)
rejected("plc_nside_too_large", dict(plc, plc_nside=16384), "plc_nside = 16384 must be <= 8192")
rejected("plc_nside_chunks_too_large", dict(plc, plc_use_chunkpix=True, plc_nside_chunks=16384),
         "plc_nside_chunks = 16384 must be <= 8192")
features = subprocess.run(mpirun + [exe, "--features"], capture_output=True, text=True, env=env).stdout
if "healpix" not in features:
    # nside = 8192 is allowed: the run goes on to the Healpix check (made when the lightcone is set up)
    rejected("plc_nside_max_without_healpix", dict(plc, plc_nside=8192),
             "plc_make_onion_density_maps = true needs Healpix", before_ic=False)

# Now working
geff = os.path.join(outdir, "geff_with_header.txt")
with open(geff, "w") as f:
    f.write("# a GeffG\n\n")
    for i in range(100):
        f.write(f"{0.001 + i / 99.0} 1.0\n")
rc, out = run("geff_header", dict(gravity_model="Geff", gravity_model_geff_geffofa_filename=geff,
                                  output_redshifts=[19.0], timestep_nsteps=[1]))
check(rc == 0, "Geff file with a header line is read (the reader looped forever)", out)
rc, out = run("jbd_cout", dict(cosmology_model="JBD", cosmology_JBD_wBD=1000.0, output_redshifts=[19.0],
                               timestep_nsteps=[1]))
as_lines = [l for l in out.splitlines() if l.startswith("# As ")]
check(rc == 0 and as_lines and "0.00000000" not in as_lines[-1],
      "JBD: As is printed with the normal format after the JBD init (" + (as_lines[-1] if as_lines else "") + ")",
      out)

# Command line
for args, what in [([], "no parameterfile"), (["--help"], "unknown option --help"),
                   (["--parameters", os.path.join(outdir, "no_such_dir", "P.md")], "--parameters to a bad path")]:
    r = subprocess.run(mpirun + [exe] + args, cwd=outdir, capture_output=True, text=True, env=env, timeout=120)
    check(r.returncode != 0, f"command line: {what} gives a non-zero exit code", r.stdout + r.stderr)

print("All validation checks passed" if fails == 0 else f"{fails} validation check(s) FAILED")
sys.exit(1 if fails else 0)
