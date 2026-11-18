# Frozen field-metric exploration probe

`plan.json` was saved before the comparison. Native version 1 fixes all physics,
data and initial seeds, projection choices and integrator/draw budgets in Main.cpp.
It samples fields with fixed scalars: four chains, 16 warmup, 128 draws, eight
leapfrog steps of size 0.15, no adaptation/jitter. This isolates FL-7's mass change;
it is not sigma8 inference or a production-calibration campaign.

```sh
make -C FML/FieldLevel/validation/galaxy/metric metric_probe metric_diagnostics USE_MPI=false
/path/to/metric_probe /external/corrected/metric
/path/to/metric_diagnostics /external/corrected/metric
```

Create the output parent first. Keep traces, diagnostics, target snapshots and logs
outside FieldLevel. Main writes each chain's six projection traces and timing,
acceptance and divergence counts. The target JSON records data, objective and
field/scalar gradients at a common point for exact comparison. Diagnostics.cpp
uses FML's existing rank-normalized split/folded Rhat and bulk/tail ESS. A diagnostic
status `ok` means computable, not converged. All four declared chains are retained.

The legacy comparison requires a genuinely pre-FL-7 executable, built before
applying the correction or retained with its original sources. The recorded study
pins both binary hashes and the before-fix headers outside the library. Current
source builds the corrected metric; the library gains no legacy-metric mode switch.
Use the same prefix CLI and plan for the old executable. Draw/step budgets are
fixed; preserve failures instead of extending until a diagnostic passes.

[Recorded results](../../../docs/validation/studies/eulerian_metric/RESULTS.md)
separate the algebraic fix, improved short-run exploration and remaining mixing
gates. The old chains are unmixed; ESS ratios are not a calibrated production
speedup. A larger joint field/scalar pilot and scientific validation remain open.
