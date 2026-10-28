#!/usr/bin/env python3
"""Small end-to-end tests; build nbody with USE_INTERNAL_COSMOLOGY=true.
Usage: python3 check_internal_transfers.py /path/to/nbody /path/to/scratch-output
"""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "regression"))
import cases

exe = str(Path(sys.argv[1]).resolve())
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="2", OMP_WAIT_POLICY="passive")
features = subprocess.check_output([exe, "--features"], text=True, env=env)
if "internal_cosmology" not in features:
    raise SystemExit("Build with USE_INTERNAL_COSMOLOGY=true to run this test")
ini = """model=lcdm
h=.671
OmegaB=.049
OmegaCDM=.2685
OmegaK=0
m_nu=0
Neff=3.046
TCMB=2.7255
A_s=2.215e-9
n_s=.966
k_pivot=.05
Yp=.245
pert_k_min=.00001
pert_k_max=.12
pert_n_k=100
"""

def run(name, overrides=None, solver=ini, error=None):
    folder = root / name
    folder.mkdir(exist_ok=True)
    (folder / "solver.ini").write_text(solver)
    values = dict(cases.BASE)
    values.update(all_parameters_must_be_in_file=False, particle_Npart_1D=8, ic_nmesh=8,
                  force_nmesh=8, pofk_nmesh=8, simulation_boxsize=1000.0,
                  simulation_use_cola=False, output_redshifts=[49.0], timestep_nsteps=[0],
                  ic_initial_redshift=49.0, ic_input_redshift=49.0,
                  ic_type_of_input="internal_transfer", ic_input_filename=str(folder / "solver.ini"),
                  output_folder=str(folder / "output"), pofk=True)
    values.update(overrides or {})
    (folder / "params.lua").write_text("\n".join(f"{k} = {cases.lua_value(v)}" for k, v in values.items()))
    with (folder / "log.txt").open("w") as log:
        result = subprocess.run([exe, "params.lua"], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (folder / "log.txt").read_text()
    if error:
        assert result.returncode and error in text, f"{name}: expected {error}; see {folder}/log.txt"
    else:
        assert result.returncode == 0, f"{name}: see {folder}/log.txt"
        assert list((folder / "output").rglob("*")), f"{name}: no output"
    print(f"[PASS] {name}", flush=True)

run("converted_ic")
run("converted_two_species", {"particle_baryons": True})
run("converted_backscaled", {"ic_input_redshift": 0.0})
run("short_sigma8_range", {"ic_sigma8_normalization": True}, error="sigma8 normalization requires transfer coverage")
run("simple_pk", {"ic_type_of_input": "powerspectrum", "ic_input_filename": cases.INPUT + "/example_power_spectrum_cb_z0.000.txt", "ic_input_redshift": 0.0})
run("wrong_cosmology", solver=ini.replace("h=.671", "h=.7"), error="parameters differ")
run("short_k_range", solver=ini.replace("pert_k_max=.12", "pert_k_max=.001"), error="range does not cover")
