#!/usr/bin/env python3
"""Small end-to-end tests of ic_type_of_input = internal (each runs the solver, ~30 s) and of its tables
(./nbody --write-transfers and ic_type_of_input = tables).
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
# A solver .ini with settings that are not the simulation's (its cosmology and k-range are; setting them is an error)
ini = """Yp=.245
z_reion=8
"""

def params(folder, overrides=None, solver=None):
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


def run(name, overrides=None, solver=None, error=None, args=None):
    folder = root / name
    folder.mkdir(exist_ok=True)
    params(folder, overrides, solver)
    with (folder / "log.txt").open("w") as log:
        result = subprocess.run([exe] + (args or ["params.lua"]), cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    text = (folder / "log.txt").read_text()
    if error:
        assert result.returncode and error in text, f"{name}: expected {error}; see {folder}/log.txt"
    else:
        assert result.returncode == 0, f"{name}: see {folder}/log.txt"
        if args is None:
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
# Settings in the parameter file, after those of the .ini (z_reion set in both: the parameter file wins)
log = run("solver_parameters", {"ic_internal_solver_parameters": ["z_reion=7", "precision=fast"]}, solver=ini)
assert "# Solver parameters: z_reion=7, precision=fast" in log, "solver_parameters: see its log.txt"
# The parameters the simulation sets are errors, in the .ini file and in the parameter file
run("solver_ini_cosmology", solver=ini + "h=.7\n", error="solver parameter h is set by the simulation (cosmology_h)")
run("solver_parameters_k", {"ic_internal_solver_parameters": ["pert_k_max=1"]},
    error="solver parameter pert_k_max is set by the simulation")
run("solver_parameters_model", {"cosmology_model": "w0waCDM", "cosmology_w0": -0.9, "cosmology_wa": 0.1,
                                "ic_internal_solver_parameters": ["w0=-0.8"]},
    error="solver parameter w0 is set by the simulation (the parameters of cosmology_model)")
run("solver_parameters_syntax", {"ic_internal_solver_parameters": ["Yp"]},
    error="ic_internal_solver_parameters: expected name=value")
run("solver_parameters_unknown", {"ic_internal_solver_parameters": ["Yp_typo=1"]}, error="Yp_typo")
log = run("neutrino_particles", {"cosmology_OmegaMNu": 0.0095, "cosmology_OmegaCDM": 0.259, "particle_neutrinos": True,
                                 "force_linear_massive_neutrinos": False, "timestep_method": "Quinn"})
# 3 degenerate species (as in the simulation)
assert re.search(r"# Massive neutrinos: ([0-9.]+), \1, \1 eV", log), "neutrino_particles: see its log.txt"
run("w0wa_curved", {"cosmology_model": "w0waCDM", "cosmology_w0": -0.9, "cosmology_wa": 0.1, "cosmology_OmegaK": 0.02})
run("tabulated", {"cosmology_model": "Tabulated", "cosmology_tabulated_filename": cases.INPUT + "/class_background_w0wa.dat",
                  "cosmology_tabulated_format": "CLASS"}, error="has no model of the solver")
run("simple_pk", {"ic_type_of_input": "powerspectrum", "ic_input_filename": cases.INPUT + "/example_power_spectrum_cb_z0.000.txt", "ic_input_redshift": 0.0})
run("no_input_file", {"ic_type_of_input": "powerspectrum"}, error="ic_input_filename is needed")

# internal_solver_default.ini (./nbody --write-solver-defaults) is the solver's defaults: the same P(k) as no .ini
default_ini = str(Path(__file__).resolve().parent.parent / "internal_solver_default.ini")
run("default_ini", {"ic_input_filename": default_ini})
import numpy as np  # noqa: E402
pk = [np.loadtxt(root / n / "output" / "snapshot_Reg_z49.000" / "pofk_z49.000.txt") for n in ["converted_ic", "default_ini"]]
assert np.array_equal(pk[0], pk[1]), "default_ini: P(k) differs from the run without a .ini"
print("[PASS] default_ini: P(k) = that of the run without a .ini", flush=True)

# The tables of the solver in a file (--write-transfers) and a run from them (ic_type_of_input = tables): the same
# P(k) as the run with the solver (converted_ic); a file for another cosmology, gauge or initial redshift is an error
tables = str(root / "tables_lcdm.txt")
run("write_transfers", {"ic_internal_solver_parameters": []}, args=["--write-transfers", "params.lua", tables])
header = open(tables).read(3000)
assert "# cosmology_model = lcdm" in header and "# columns = a k_hmpc d_cdm d_b d_cb d_m t_cdm" in header, \
    "write_transfers: the header of " + tables
from_file = {"ic_type_of_input": "tables", "ic_input_filename": tables}
run("tables", from_file)
pk = [np.loadtxt(root / n / "output" / "snapshot_Reg_z49.000" / "pofk_z49.000.txt") for n in ["converted_ic", "tables"]]
assert np.allclose(pk[0], pk[1], rtol=1e-8, atol=0), "tables: P(k) differs from the run with the solver"
print("[PASS] tables: P(k) = that of the run with the solver", flush=True)
run("tables_other_h", dict(from_file, cosmology_h=0.7), error="Made for cosmology_h = 0.67100000000000004")
run("tables_other_neutrinos", dict(from_file, cosmology_OmegaMNu=0.0048),
    error="Made for cosmology_m_nu_eV = none")
run("tables_other_As", dict(from_file, cosmology_As=2.0e-9), error="another primordial spectrum")
run("tables_nbody_gauge", dict(from_file, force_nbody_gauge=True), error="needs N-body gauge tables")
run("tables_zini", dict(from_file, ic_initial_redshift=199.0, ic_input_redshift=199.0, output_redshifts=[199.0]),
    error="The tables start at z = ")
run("tables_not_a_file", dict(from_file, ic_input_filename=str(root / "converted_ic" / "params.lua")),
    error="Not an FML transfer tables file")
run("write_transfers_not_internal", {"ic_type_of_input": "powerspectrum",
                                     "ic_input_filename": cases.INPUT + "/example_power_spectrum_cb_z0.000.txt"},
    args=["--write-transfers", "params.lua", str(root / "x.txt")], error="needs ic_type_of_input = internal")

# Ultralight axions in the IC only (ic_axion_fraction): the solver's model axion, the particles start from CDM + axion.
# A small box resolves the suppression of the power by an axion of 1e-24 eV (Jeans scale ~1 h/Mpc at zini)
small = {"simulation_boxsize": 32.0, "particle_Npart_1D": 16, "ic_nmesh": 16, "force_nmesh": 16, "pofk_nmesh": 16}
run("axion_lcdm_reference", small)
log = run("axion_ic", dict(small, ic_axion_fraction=0.3, ic_axion_log10_m=-24.0))
assert "ultralight axion in the IC only" in log and "Model: axion" in log, "axion_ic: see its log.txt"
pk = [np.loadtxt(root / n / "output" / "snapshot_Reg_z49.000" / "pofk_z49.000.txt") for n in
      ["axion_lcdm_reference", "axion_ic"]]
ratio = pk[1][:, 1] / pk[0][:, 1]
assert abs(ratio[0] - 1) < 0.02 and ratio[-1] < 0.9 and np.all(np.diff(ratio) < 0.01), \
    f"axion_ic: P/P_LCDM = {ratio} (1 on large scales, suppressed on small)"
print(f"[PASS] axion_ic: P/P_LCDM from {ratio[0]:.3f} to {ratio[-1]:.3f}", flush=True)
run("axion_ic_too_light", dict(small, ic_axion_fraction=0.1, ic_axion_log10_m=-28.0),
    error="the axion is a matter-like fluid only from")
run("axion_ic_w0wa", dict(small, ic_axion_fraction=0.1, cosmology_model="w0waCDM", cosmology_w0=-0.9,
                          cosmology_wa=0.1), error="ic_axion_fraction > 0 needs cosmology_model = LCDM")
run("axion_ic_nbody_gauge", dict(small, ic_axion_fraction=0.1, force_nbody_gauge=True),
    error="not implemented with force_nbody_gauge")
axtables = str(root / "tables_axion.txt")
axion = dict(small, ic_axion_fraction=0.3, ic_axion_log10_m=-24.0)
run("write_transfers_axion", axion, args=["--write-transfers", "params.lua", axtables])
assert "# cosmology_model = axion" in open(axtables).read(3000), "write_transfers_axion: the header"
run("tables_axion", dict(axion, ic_type_of_input="tables", ic_input_filename=axtables))
pk = [np.loadtxt(root / n / "output" / "snapshot_Reg_z49.000" / "pofk_z49.000.txt") for n in ["axion_ic", "tables_axion"]]
assert np.allclose(pk[0], pk[1], rtol=1e-8, atol=0), "tables_axion: P(k) differs from the run with the solver"
print("[PASS] tables_axion: P(k) = that of the run with the solver", flush=True)
run("tables_axion_other_mass", dict(axion, ic_type_of_input="tables", ic_input_filename=axtables, ic_axion_log10_m=-23.0),
    error="Made for cosmology_model_ax_log10_m")
