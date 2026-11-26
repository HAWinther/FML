#!/usr/bin/env python3
"""Reference runs of SimulationCode's common-timestep PM + short-range N-body (force_short_range), frozen for TreePMCode.

SimulationCode's short-range force is to be removed once TreePMCode reproduces these runs (README.md here). This script
records exactly how the data were made; it needs a SimulationCode that still has force_short_range.

Usage: python3 make_simcode_reference.py --nbody path/to/SimulationCode/nbody [--work /tmp/simcode_reference] [--mpi N]
Writes the parameter files, the initial conditions (GADGET), the z = 0 particles of the main run, the P(k) of every
case and summary.json into this folder.
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
simcode = here.parents[2] / "SimulationCode"
spec = importlib.util.spec_from_file_location("cases", simcode / "tests" / "regression" / "cases.py")
cases_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases_module)

BOX, NPART, NMESH = 64.0, 32, 64
SPACING = BOX / NPART  # mean particle separation, 2 Mpc/h
BASE = dict(cases_module.BASE)
BASE.update(
    all_parameters_must_be_in_file=False,
    simulation_name="Ref",
    simulation_boxsize=BOX,
    simulation_use_cola=False,
    particle_Npart_1D=NPART,
    particle_allocation_factor=3.0,
    ic_nmesh=NPART,
    ic_initial_redshift=19.0,
    ic_LPT_order=2,
    ic_random_seed=1234,
    ic_use_gravity_model_GR=True,
    output_redshifts=[19.0, 1.0, 0.0],
    output_particles=True,
    output_fileformat="GADGET",
    timestep_method="Quinn",
    timestep_algorithm="KDK",
    timestep_scalefactor_spacing="logarithmic",
    force_nmesh=NMESH,
    force_density_assignment_method="CIC",
    force_greens_function_kernel="continuous",
    force_gradient_kernel="continuous",
    force_short_range=True,
    force_short_range_backend="cell",
    force_short_range_split_scale=1.25 * BOX / NMESH,  # 1.25 force cells (GADGET-2)
    force_short_range_softening_support=0.07 * SPACING,  # h = 2.8 epsilon, epsilon = 1/40 of the separation
    force_short_range_cutoff_tolerance=1e-6,
    force_short_range_table_points=16384,
    pofk=True,
    pofk_every_step=False,
    pofk_nmesh=64,
    pofk_interlacing=True,
    pofk_subtract_shotnoise=False,
    pofk_density_assignment_method="PCS",
)
def steps(n):  # log spacing: 3/4 of the steps from z = 19 to 1, 1/4 from z = 1 to 0
    return [0, 3 * n // 4, n // 4]


# 256 -> 512 -> 1024 common steps is the time-step convergence; treepm_1024 is the reference
CASES = {
    "treepm_256": dict(timestep_nsteps=steps(256)),
    "treepm_512": dict(timestep_nsteps=steps(512)),
    "treepm_1024": dict(timestep_nsteps=steps(1024)),
    # PM only with SimulationCode's fiducial PM settings (the short-range force needs continuous kernels; without it
    # the fiducial 4-point gradient on this 2x grid is the better PM): the size of the short-range contribution
    "pm_fiducial_256": dict(timestep_nsteps=steps(256), force_short_range=False, force_greens_function_kernel="fiducial",
                            force_gradient_kernel="fiducial"),
    # Unscreened Hu-Sawicki f(R) (n = 1, fR0 = 1e-5): the model-owned Yukawa short-range force
    "treepm_fofr_512": dict(
        timestep_nsteps=steps(512),
        gravity_model="f(R)",
        gravity_model_fofr_fofr0=1e-5,
        gravity_model_fofr_nfofr=1.0,
        gravity_model_screening=False,
        gravity_model_fofr_exact_solution=False,
    ),
}
MAIN = "treepm_1024"


def run(nbody, work, name, overrides, mpi, reuse=False):
    d = work / name
    if reuse and (d / "output" / "snapshot_Ref_z0.000" / "pofk_z0.000.txt").is_file():
        return d / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p = dict(BASE)
    p.update(overrides)
    p["output_folder"] = str(d / "output")
    text = "\n".join(k + " = " + cases_module.lua_value(v) for k, v in p.items()) + "\n"
    (d / "parameterfile.lua").write_text(text)
    # The parameter file as stored here (the output folder relative, the input power spectrum relative to FML/)
    p["output_folder"] = "output"
    p["ic_input_filename"] = str(Path(p["ic_input_filename"]).relative_to(simcode.parent))
    (here / "parameters" / (name + ".lua")).write_text(
        "\n".join(k + " = " + cases_module.lua_value(v) for k, v in p.items()) + "\n"
    )
    cmd = (["mpirun", "-np", str(mpi)] if mpi > 1 else []) + [str(nbody), str(d / "parameterfile.lua")]
    with (d / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS="2"))
    return d / "output"


def read_pofk(path):
    k, pk = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        k.append(float(f[0]))
        pk.append(float(f[1]))
    return k, pk


def md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ratio_stats(num, den, kmax):
    k, a = num
    _, b = den
    rel = [(ki, ai / bi - 1) for ki, ai, bi in zip(k, a, b) if ki <= kmax]
    at = lambda kk: min(rel, key=lambda r: abs(r[0] - kk))[1]
    return dict(max_abs=max(abs(r) for _, r in rel), at_k_0p2=at(0.2), at_k_0p5=at(0.5), at_k_1=at(1.0),
                at_kmax=rel[-1][1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbody", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/tmp/simcode_reference"))
    parser.add_argument("--mpi", type=int, default=1)
    parser.add_argument("--reuse", action="store_true", help="only collect the outputs of earlier runs in --work")
    args = parser.parse_args()
    (here / "parameters").mkdir(exist_ok=True)
    (here / "pofk").mkdir(exist_ok=True)
    nbody = args.nbody.resolve()
    outputs = {name: run(nbody, args.work, name, o, args.mpi, args.reuse) for name, o in CASES.items()}
    # The same run on 2 MPI ranks (P(k) must agree to round-off)
    outputs["treepm_256_mpi2"] = run(nbody, args.work, "treepm_256_mpi2", CASES["treepm_256"], 2, args.reuse)
    (here / "parameters" / "treepm_256_mpi2.lua").unlink(missing_ok=True)
    print("runs done", flush=True)

    # Initial conditions: identical in every case (GR IC, same seed)
    # (the 2-rank run writes one file per task)
    ics = {name: out / "snapshot_Ref_z19.000" / "gadget_z19.000.0" for name, out in outputs.items() if "mpi" not in name}
    sums = {name: md5(p) for name, p in ics.items()}
    assert len(set(sums.values())) == 1, sums
    shutil.copy(ics[MAIN], here / "ic_z19.000.gadget")
    shutil.copy(outputs[MAIN] / "snapshot_Ref_z0.000" / "gadget_z0.000.0", here / f"{MAIN}_z0.000.gadget")
    pk = {}
    for name, out in outputs.items():
        for z in ["1.000", "0.000"]:
            src = out / f"snapshot_Ref_z{z}" / f"pofk_z{z}.txt"
            if "mpi" not in name:
                shutil.copy(src, here / "pofk" / f"{name}_z{z}.txt")
            pk[(name, z)] = read_pofk(src)
    kny = math.pi * NPART / BOX  # particle Nyquist frequency
    summary = dict(
        description="SimulationCode N-body (Quinn KDK, log steps) with force_short_range, 32^3 in 64 Mpc/h, z = 19 -> 0",
        particle_knyquist_hmpc=kny,
        split_scale_mpch=BASE["force_short_range_split_scale"],
        softening_support_mpch=BASE["force_short_range_softening_support"],
        reference_run=MAIN,
        md5={"ic_z19.000.gadget": sums[MAIN], f"{MAIN}_z0.000.gadget": md5(here / f"{MAIN}_z0.000.gadget")},
        comparisons={},
    )
    for z in ["1.000", "0.000"]:
        c = summary["comparisons"][f"z{z}"] = {}
        ref = pk[(MAIN, z)]
        c["timestep_256_vs_1024"] = ratio_stats(pk[("treepm_256", z)], ref, kny)
        c["timestep_512_vs_1024"] = ratio_stats(pk[("treepm_512", z)], ref, kny)
        c["pm_fiducial_vs_treepm"] = ratio_stats(pk[("pm_fiducial_256", z)], ref, kny)
        c["fofr_vs_gr_512"] = ratio_stats(pk[("treepm_fofr_512", z)], pk[("treepm_512", z)], kny)
        c["mpi2_vs_serial_256"] = ratio_stats(pk[("treepm_256_mpi2", z)], pk[("treepm_256", z)], 2 * kny)
    (here / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["comparisons"], indent=1))


if __name__ == "__main__":
    main()
