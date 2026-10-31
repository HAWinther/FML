#!/usr/bin/env python3
"""ic_type_of_input = class (native CLASS output through FML/TransferProviders) against camb (CAMB transfer files).

Usage: python3 check_class_input.py /path/to/nbody /path/to/scratch-output [transfer data folder]

Uses the reference runs of FML/TransferProviders/tests (CLASS in the synchronous and the Newtonian gauge and CAMB, one
0.1 eV massive neutrino; make them with CLASS=/path/to/class ./make_test_data.sh there). Runs the regression case
scaledep_cola_neutrinos (scale-dependent COLA with massive neutrinos, 32^3) from each input and checks that P_cb and
P_total agree with the CAMB run to 2e-3 at all outputs (they agree to ~1e-3: CLASS vs CAMB), and that wrong
declarations are errors. Prints SKIP without the data.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "regression"))
import cases

exe = str(Path(sys.argv[1]).resolve())
root = Path(sys.argv[2]).resolve()
data = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else here.parent.parent / "TransferProviders" / "tests" / "data"
if not (data / "class_synchronous_00_background.dat").exists():
    print(f"SKIP check_class_input: no reference data in {data}")
    sys.exit(0)
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="2")
MPIRUN = os.environ.get("MPIRUN", "mpirun -np 2").split()

h = 0.67
common = dict(cases.BASE)
common.update(cases.CASES["scaledep_cola_neutrinos"])
common.update(cosmology_h=h, cosmology_Omegab=0.0224 / h**2, cosmology_OmegaCDM=0.119 / h**2,
              cosmology_OmegaMNu=float(open(data / "camb_parameters.txt").read().split()[2]),
              cosmology_As=2.1e-9, cosmology_ns=0.965, ic_input_redshift=20.0, ic_initial_redshift=20.0,
              output_folder="output")


def run(name, overrides, error=None):
    folder = root / name
    folder.mkdir(exist_ok=True)
    values = dict(common)
    values.update(overrides)
    (folder / "parameterfile.lua").write_text("\n".join(f"{k} = {cases.lua_value(v)}" for k, v in values.items()))
    with (folder / "log.txt").open("w") as log:
        result = subprocess.run(MPIRUN + [exe, "parameterfile.lua"], cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    text = (folder / "log.txt").read_text()
    if error:
        assert result.returncode and error in text, f"{name}: expected [{error}]; see {folder}/log.txt"
    else:
        assert result.returncode == 0, f"{name}: see {folder}/log.txt"
    print(f"[PASS] {name} runs" + (" (error as expected)" if error else ""), flush=True)
    return folder / "output"


camb = run("camb", dict(ic_type_of_input="camb", ic_input_filename=str(data / "camb_transferinfo.txt")))
for gauge in ["synchronous", "newtonian"]:
    out = run("class_" + gauge, dict(ic_type_of_input="class", ic_input_filename=str(data / f"class_{gauge}_00_"),
                                     ic_input_gauge=gauge))
    worst = 0.0
    for f in sorted(camb.glob("pofk_*.txt")):
        a, b = np.loadtxt(f), np.loadtxt(out / f.name)
        worst = max(worst, np.max(np.abs(b[:, 1:] / a[:, 1:] - 1)))
    assert worst < 2e-3, f"class ({gauge}) vs CAMB: P(k) differs by {worst:.1e}"
    print(f"[PASS] class ({gauge}) vs CAMB: P_cb and P_total agree to {worst:.1e}", flush=True)

run("wrong_gauge", dict(ic_type_of_input="class", ic_input_filename=str(data / "class_newtonian_00_"),
                        ic_input_gauge="synchronous"), error="Declared synchronous gauge")
run("wrong_h", dict(ic_type_of_input="class", ic_input_filename=str(data / "class_synchronous_00_"),
                    ic_input_gauge="synchronous", cosmology_h=0.7), error="has h = ")
run("no_neutrinos", dict(ic_type_of_input="class", ic_input_filename=str(data / "class_massless_synchronous_00_"),
                         ic_input_gauge="synchronous"),
    error="has no massive neutrinos")
