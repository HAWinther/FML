> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Stable-Armijo endpoint diagnostic (#52)

This bounded follow-up freezes the iteration-98 `LineSearchFailed` endpoint of the
[stable-comparison reconstruction](../pm_stable_armijo_validation/RESULTS.md).
It retains the compensated N32/M16/16-step model, seed-300/.05 initialization,
frozen N128/M64 data, fixed nuisance parameters, and raw tolerance 2e-5.
No force, assignment, optimizer, stopping rule or target change is made.

The shared capture executable takes an optional final `STABLE_CHANGE` argument
(default 0). In stable mode it skips exactly one reference prediction following
each full gradient, then captures the first actual line-search trial. Captured
fields and gradients must reproduce the published stable endpoint bitwise before
probing. `.capture.csv` records the option and reference-prediction count. All
ranks must agree on the option. The snapshot format and loader are unchanged.

The shared probe measures the same three directions, fourteen perturbation sizes,
forty original alpha values, corresponding particle/node/axis CIC stencils, and
fresh/reused checkpoint budgets 0/1/4/8 as the
[original diagnostic](../pm_stationarity_validation/README.md). It retains both
one-sided slopes and central differences, including changed-stencil and
roundoff-dominated points. Only same-stencil pairs in 1e-10..1e-5 enter the
pre-existing derivative gates. A gate can fail because insufficient branch-
preserving pairs exist; that alone does not establish an adjoint error.

The failed direction is reconstructed from the accepted point minus the first
unit trial. Subtraction rounding means the smallest reconstructed trials are
measurements, not guaranteed bitwise replays of every internal trial. Predictions
remain double precision even when target differences use long double locally.

```sh
make -C FML/FieldLevel/pm_stationarity_validation USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_stationarity_validation/run.py --stable-changes --bin-dir /path/to/binaries --output /fresh/OUT --phase all --np 4 --mpirun 'mpirun --oversubscribe'
```

Use separate builds, suitable FFTW/compiler/launcher overrides, shared files and
one OpenMP thread. Each child has a bounded 900s default deadline. `--phase probe`
uses the retained full snapshot; `--phase analyze` reuses measurements in OUT.
The analyzer's `--reference` selects the stable endpoint coefficients;
`--line-expectation measure` records acceptance counts without imposing the
old endpoint's hidden-descent outcome. Existing defaults retain the original
study's gates. Reanalyze published files with NumPy and Matplotlib:

```sh
python3 FML/FieldLevel/pm_stationarity_validation/analyze.py --input FML/FieldLevel/pm_stable_stationarity_validation --output /tmp/fml-stable-stationarity-report --reference FML/FieldLevel/pm_stable_armijo_validation/science/stable-lbfgs.fit_coefficients.csv --line-expectation measure
```

Generate the compact endpoint figure after that analysis:

```sh
python3 FML/FieldLevel/pm_stable_stationarity_validation/plot_endpoint.py --input FML/FieldLevel/pm_stable_stationarity_validation --output /tmp/fml-stable-stationarity-report
```

See [RESULTS.md](RESULTS.md). Broader seed/amplitude/refinement and posterior
calibration remain deferred. This is a finite diagnostic, not a convergence proof
or a production inference checkpoint facility.
