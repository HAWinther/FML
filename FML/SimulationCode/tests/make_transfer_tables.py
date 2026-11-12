#!/usr/bin/env python3
#==========================================================================
# Regenerate the transfer tables of the tests (input/transfer_tables/*.txt) with the internal
# Einstein-Boltzmann solver: ./nbody --write-transfers for the cosmology of the regression cases
# (tests/regression/cases.py BASE) with 0, 0.1, 0.2 and 0.4 eV of massive neutrinos (three degenerate
# species), in the N-body gauge (0 eV) and with dark scattering, for ic_initial_redshift = 20 as in the tests (the tables cover
# z <= 41, as those of ic_type_of_input = internal for these runs).
# The tests read them with ic_type_of_input = tables, which checks that the cosmology is the same.
#
# Usage: python3 make_transfer_tables.py <path/to/nbody> [name ...]   (default: all)
#   MPIRUN : how to launch (default ""; the solver runs on task 0 only)
# Rerun after a change of the solver, then make a new reference of the regression tests.
#==========================================================================
import os
import shlex
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "regression"))
import cases  # noqa: E402

OUTDIR = os.path.join(cases.INPUT, "transfer_tables")

# name: overrides of cases.BASE
TABLES = {
    "lcdm_nu0.0": dict(),
    "lcdm_nu0.1": dict(cosmology_OmegaMNu=0.0024, cosmology_OmegaCDM=0.2661),
    "lcdm_nu0.2": dict(cosmology_OmegaMNu=0.0048, cosmology_OmegaCDM=0.2637),
    "lcdm_nu0.4": dict(cosmology_OmegaMNu=0.0095, cosmology_OmegaCDM=0.259),
    "lcdm_nbody_gauge": dict(force_nbody_gauge=True),
    # Dark matter - dark energy scattering (the regression cases with CDM and baryon particles)
    "dark_scattering": dict(cosmology_model="DarkScattering", cosmology_ds_w=-0.9, cosmology_ds_xi=50.0,
                            cosmology_ds_cs2=1.0),
}


def filename(name):
    return os.path.join(OUTDIR, name + ".txt")


def main():
    exe = os.path.abspath(sys.argv[1])
    names = sys.argv[2:] or list(TABLES)
    mpirun = shlex.split(os.environ.get("MPIRUN", ""))
    os.makedirs(OUTDIR, exist_ok=True)
    for name in names:
        p = dict(cases.BASE)
        p.update(TABLES[name])
        p.update(ic_type_of_input="internal", ic_input_filename="", ic_internal_solver_parameters=[],
                 ic_initial_redshift=20.0, ic_input_redshift=20.0)
        # The IC-only axion parameters (read with ic_type_of_input = internal; all parameters must be in the file)
        p.setdefault("ic_axion_fraction", 0.0)
        p.setdefault("ic_axion_log10_m", -22.0)
        p.setdefault("ic_axion_switch_mH", 0.0)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "parameterfile.lua"), "w") as f:
                for k, v in p.items():
                    f.write(f"{k} = {cases.lua_value(v)}\n")
            res = subprocess.run(mpirun + [exe, "--write-transfers", "parameterfile.lua", filename(name)], cwd=d,
                                 capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:], res.stderr[-3000:])
            sys.exit(f"Failed to make {filename(name)}")
        print(f"Wrote {filename(name)}")


if __name__ == "__main__":
    main()
