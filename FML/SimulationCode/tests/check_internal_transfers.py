#!/usr/bin/env python3
"""Small end-to-end tests of ic_type_of_input = internal (each runs the solver, ~30 s).
Usage: python3 check_internal_transfers.py /path/to/nbody /path/to/scratch-output
"""
import os
import re
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "regression"))
import cases

exe = str(Path(sys.argv[1]).resolve())
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="2", OMP_WAIT_POLICY="passive")
# A solver .ini with other settings: its cosmology and k-range are replaced by those of the simulation
ini = """model=lcdm
h=.7
OmegaB=.05
m_nu=0.06
Yp=.245
z_reion=8
pert_k_max=.12
pert_n_k=100
"""

def run(name, overrides=None, solver=None, error=None):
    folder = root / name
    folder.mkdir(exist_ok=True)
    values = dict(cases.BASE)
    values.update(all_parameters_must_be_in_file=False, particle_Npart_1D=8, ic_nmesh=8,
                  force_nmesh=8, pofk_nmesh=8, simulation_boxsize=1000.0,
                  simulation_use_cola=False, output_redshifts=[49.0], timestep_nsteps=[0],
                  ic_initial_redshift=49.0, ic_input_redshift=49.0,
                  ic_type_of_input="internal", ic_input_filename="",
                  output_folder=str(folder / "output"), pofk=True)
    if solver is not None:
        (folder / "solver.ini").write_text(solver)
        values["ic_input_filename"] = str(folder / "solver.ini")
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
    return text

run("converted_ic")
run("converted_two_species", {"particle_baryons": True})
run("converted_backscaled", {"ic_input_redshift": 0.0})
run("sigma8_normalization", {"ic_sigma8_normalization": True})
run("old_name", {"ic_type_of_input": "internal_transfer"})
log = run("solver_ini", solver=ini)
assert "Model: lcdm (solver settings:" in log and "Massive neutrinos: none" in log, "solver_ini: see its log.txt"
log = run("neutrino_particles", {"cosmology_OmegaMNu": 0.0095, "cosmology_OmegaCDM": 0.259, "particle_neutrinos": True,
                                 "force_linear_massive_neutrinos": False, "timestep_method": "Quinn"})
# 3 degenerate species (as in the simulation)
assert re.search(r"# Massive neutrinos: ([0-9.]+), \1, \1 eV", log), "neutrino_particles: see its log.txt"
run("w0wa_curved", {"cosmology_model": "w0waCDM", "cosmology_w0": -0.9, "cosmology_wa": 0.1, "cosmology_OmegaK": 0.02})
run("tabulated", {"cosmology_model": "Tabulated", "cosmology_tabulated_filename": cases.INPUT + "/class_background_w0wa.dat",
                  "cosmology_tabulated_format": "CLASS"}, error="has no model of the solver")
run("simple_pk", {"ic_type_of_input": "powerspectrum", "ic_input_filename": cases.INPUT + "/example_power_spectrum_cb_z0.000.txt", "ic_input_redshift": 0.0})
run("no_input_file", {"ic_type_of_input": "powerspectrum"}, error="ic_input_filename is needed")
