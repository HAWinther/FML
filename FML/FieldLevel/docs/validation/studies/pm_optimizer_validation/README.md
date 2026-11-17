> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Frozen-data MAP optimizer comparison (#52)

Use exactly the compensated N128/M64/32-step mock and N32/M16/16-step model from
[the reconstruction study](../pm_reconstruction_validation/README.md), with the same
M16 CIC observation window, fixed nuisance parameters, seed300/.05 start and raw
gradient tolerance 2e-5. The broader refinement campaign remains deferred.

Compare legacy gradient descent (24 iterations), options-based gradient descent
with value-only line searches (24), and L-BFGS with history 5 (up to 128). Report
actual prediction/pullback calls around MAP, wall time, accepted iterations,
status, raw gradient, recovery and residual diagnostics. L-BFGS has a larger
iteration cap to test whether it can reach the same tolerance; compare computational
work and accuracy, not iteration count alone. The legacy24 coefficients/metadata
must replay bitwise from the published study. Value-only GD coefficients must
match the legacy accepted endpoint bitwise.

The new native optional arguments after CIC_REFERENCE are OPTIMIZER (0 legacy,
1 value-only GD, 2 L-BFGS) and HISTORY (1..32, meaningful for L-BFGS).
`.fit_optimizer.csv` reports actual MAP-only work and time; `.fit_trace.csv` records
accepted iterates for options-based algorithms. Additional post-fit diagnostics
are outside these counters/timer. The legacy path retains all old numerical output.

Memory reports distinguish retained history vector payload from summed per-rank
process peak RSS. RSS includes the process lifetime and is not simultaneous
cluster memory. Timing is indicative under uncontrolled host activity. Call-count
savings are more reproducible. No production-scaling claim is made.

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_optimizer_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --np 4 --mpirun 'mpirun --oversubscribe'
```

Clean/separate compiler/backend configurations; supply the normal FML overrides.
NumPy/Matplotlib are required. Each native job is bounded to 600s with process-group
cleanup. `--phase cases` and `--phase analyze` separate execution/report generation.
The analyzer reuses the preceding reconstruction diagnostics for each algorithm;
all published inputs are retained so analysis can run without native trajectories.
See [RESULTS.md](RESULTS.md) for actual results and failed criteria. This is one
fixed-parameter fit, not uncertainty, coverage or broader physical calibration.
