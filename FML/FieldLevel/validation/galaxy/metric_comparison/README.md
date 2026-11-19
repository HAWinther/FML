# Frozen nonlinear field-metric chain comparison

The [local Gauss-Newton screen](../gauss_newton/README.md) passes with modest mixed
projection-jump improvements. This separate fixed-budget test determines whether
that translates into useful joint field/scalar exploration on dataset 203.

A fresh four-chain candidate uses 256 discarded warmup sweeps and 512 retained
cold draws per chain. Reuse the completed thirteen-replica refined-ladder cohort
as a read-only baseline. Freeze its full checkpoints/records/diagnostics and never
rewrite its analysis. Both use identical targets, starts, seed domains, ladder,
step 0.075 ×16, scalar/rescaling updates, priors, filters and twelve-coordinate/
transport/divergence gates. Only the field kinetic metric changes. Candidate metric
files are copied and pinned before launch; M_beta=I+beta K is fixed per ensemble.
There is no extension, tuning until pass, chain exclusion or hot/replay pooling.

Compare measured primary HMC gradient counts including warmup. The cached baseline
shared eight concurrent native workers across two cohorts; the candidate uses four.
Raw active wall seconds are reported separately and do not certify a speedup.
Auxiliary validation replay is excluded from gradient counters and included in time.
All failed-chain ESS and posterior summaries remain provisional. This selected-data
experiment is not independent calibration or a real-catalog analysis.

## Native integration and identities

make_driver.py generates a driver from the unchanged original tempering/Main.cpp
using four explicit, unique source hooks: include the metric adapter, load the
ensemble metric, dispatch the counted field update, require explicit settings,
and add the metric-byte checksum to the existing checkpoint target identity.
(The include is prepended; four original-source anchors must match exactly.)
The generated GNMain.cpp stays in the external build directory. This reuses the
existing scalar/transport/checkpoint code without copying a maintained driver or
editing frozen historical sources. Review the generated native-hooks.patch.

Native.h applies the existing Gaussian-target HMC with the fixed low-rank mass;
legacy mode delegates to the exact old counted kernel. Native preflight checks
legacy draw/summary/target identity; candidate full-checkpoint/summary/draw restart;
changed metric eigenvalues/kind rejection; missing/truncated metric rejection;
and actual warmup/sampling gradient counts. New native checkpoint identities are
intentionally distinct even in legacy mode. Metric copies and SHA256 digests are
also included in runner identity and the frozen parent manifest.

```sh
make gn_tempering
python3 preflight.py /absolute/empty-controls --exe /absolute/gn_tempering --old-exe /absolute/old_tempering_pilot --metrics /absolute/gauss-newton-study
python3 study.py prepare /absolute/empty-comparison --source /absolute/ladder-study/refined/mock203 --metrics /absolute/gauss-newton-study --controls /absolute/controls/controls.json --exe /absolute/gn_tempering --diagnostics /absolute/calibration_diagnostics
python3 study.py run /absolute/comparison --exe /absolute/gn_tempering --diagnostics /absolute/calibration_diagnostics --max-rounds 1
python3 study.py run /absolute/comparison --exe /absolute/gn_tempering --diagnostics /absolute/calibration_diagnostics --resume
python3 study.py analyze /absolute/comparison --exe /absolute/gn_tempering --diagnostics /absolute/calibration_diagnostics
```

The study and cohort writer locks exclude concurrent writers. SIGTERM/SIGINT or
PAUSE drains active native checkpoint chunks; --resume clears pause receipts.
A cumulative two-hour sampling cap reserves 1200 seconds per batch before launch;
crashed batches retain the reservation. Missing budget state beside native work
rejects resume. Each chain performs the existing full-ensemble restart replay.
No production MPI replica orchestration or distributed metric construction is added.
See [results](../../../docs/validation/studies/metric_comparison/RESULTS.md).
