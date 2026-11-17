> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Stable objective changes for Armijo (#52)

This opt-in comparison implements the change calculation identified by the
[stalled-state diagnostic](../pm_stationarity_validation/README.md). It preserves
the force, assignment, gradient, target normalization and convergence tolerance.
Legacy `map()` and the options overload's default comparisons remain unchanged.

First replay all 40 original step sizes at the frozen 83-iteration endpoint.
The public stable-change API must reproduce the independent diagnostic and
accept the previously hidden descent candidate. Then repeat the same compensated
N32/M16/16-step fit to frozen N128/M64 data, common M16 observation window,
fixed nuisance parameters, seed-300/.05 initialization and raw tolerance 2e-5.
Run stable L-BFGS (history 5, cap 128) and stable GD (cap 24). Keep every optimizer
status and any failed convergence criterion.

The final native argument after HISTORY is STABLE_CHANGE (0 by default, 1 enables
it for OPTIMIZER=1/2). `.fit_change.csv` reports extra reference predictions;
`.fit_change_trace.csv` records each accepted stable change and Armijo bound.
Actual prediction counts include reference predictions. Rounded absolute objective
values may rise by roundoff even when the stable comparison reports descent.

```sh
make -C FML/FieldLevel/pm_stable_armijo_validation USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_stable_armijo_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --np 4 --mpirun 'mpirun --oversubscribe'
```

Use fresh output/separate builds and suitable compiler/FFTW/launcher overrides.
Each child has a 900s default deadline with process-group cleanup. NumPy and
Matplotlib are required. `--phase cases` and `--phase analyze` separate execution
and reporting. The frozen state and data are retained in the preceding study
folders. All new native outputs are retained here for analysis without rerunning:

```sh
python3 FML/FieldLevel/pm_stable_armijo_validation/analyze.py --input FML/FieldLevel/pm_stable_armijo_validation --output /tmp/fml-stable-armijo-report
```

See [RESULTS.md](RESULTS.md) and [MAP API](../../../MAP.md). A scientific convergence
failure produces exit 1 after saving all reports; engineering/consistency checks
are separate. Broader seed/amplitude/refinement and posterior calibration remain
deferred. Wall times are indicative under uncontrolled host activity.
