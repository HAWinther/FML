> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Opt-in TSC PM reconstruction (#52)

This bounded experiment follows the [CIC endpoint diagnostic](../pm_stable_stationarity_validation/RESULTS.md).
It changes the discretized PM assignment to TSC, with matching source deposition,
force gather, reported density, pullbacks and force-window compensation. CIC and
its checkpoint identity remain the defaults.

The physical initial modes, seeds, power, noise power, observation band, time
nodes, reference/fit resolutions, fixed nuisance parameters, initialization and
raw convergence tolerance follow the earlier fit. Generate **new TSC data** at
N128/M64/32 steps, then fit with TSC at N32/M16/16 steps. Use the same M16 CIC
observation-window convention, implemented as coarse-CIC/fine-TSC in Fourier
space; this is a fixed signal response, not a changed noise model. The driver
checks an explicit assignment-specific data identity. The targets differ between
CIC and TSC, so the comparison is a matched-assignment experiment, not an
identical-target optimizer benchmark.

Use stable-change L-BFGS, history 5, cap 128, unchanged raw tolerance 2e-5,
seed-300/.05 initialization, fixed spectrum/noise parameters, four pencil ranks
and one OpenMP thread. The truth-point finite differences retain eight scales in
each of two directions, with the existing final-four-scale gates (best scaled
error <2e-5 and at least two <1e-4). Unit tests independently check TSC knots and
periodic seams; PM tests cover full adjoints, volume scaling, replay budgets,
independent Fourier windows and slab/pencil consistency.

```sh
make -C FML/FieldLevel/pm_nonlinear_validation USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_tsc_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --np 4 --mpirun 'mpirun --oversubscribe'
```

Use suitable compiler/FFTW/launcher overrides and separate builds. Each child has
a 900s default deadline with process-group cleanup. `--phase cases` excludes
analysis; `--phase analyze` reuses the native outputs and plan in OUT. NumPy and
Matplotlib are required. Published measurements can be reanalyzed without jobs:

```sh
python3 FML/FieldLevel/pm_tsc_validation/analyze.py --input FML/FieldLevel/pm_tsc_validation --output /tmp/fml-tsc-report
```

The driver adds optional final `ASSIGNMENT` (2=CIC default, 3=TSC) after
`STABLE_CHANGE`. `FORCE_WINDOW` is 0=none, 1=CIC-compensated, 2=TSC-compensated;
nonzero compensation must match the assignment. TSC writes `.assignment.csv` and
adds `assignment_tsc` to the data identity. Existing CIC numerical file formats
and default calls are preserved.

See [RESULTS.md](RESULTS.md). All statuses and failed gates are retained. Broader
seed/amplitude/refinement, survey support and posterior calibration remain
deferred; a converged finite MAP is not a calibrated posterior or continuum test.
