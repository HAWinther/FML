# Dataset-204 actual-chain integrator comparison

This is a targeted, paired diagnostic on an already examined failed dataset.
It is not an independent calibration campaign. The original eight-dataset pilot
and its failed outcomes stay intact.

The two prospective plans fix four chains each, 256 discarded warmup sweeps and
512 retained cold draws, the original eleven replicas and four exchange passes.
The baseline uses 0.15/8 leapfrog steps; the half-step case uses 0.075/16. Both
have integration time 1.2 and the original frozen-per-ensemble field mass.
All data-generation controls, starts, sampler seeds, scalar kernels, physical
priors, masks and noise are identical. Common seeds pair the designs; they are
not independent replications. No current-scalar mass is substituted.

Every original coordinate gate applies: five scalars, six field projections and
likelihood Q, Rhat<1.01 and bulk/tail ESS>=400, plus transport and zero-divergence
gates. The 512-draw budget is deliberately a first bounded actual-chain test, not
a claim of full 2048-draw calibration. Stop even if gates fail; no extension until
pass. The baseline is replayed so its cost can be measured directly; every one
of its 2048 retained records must equal the actual original 512-draw prefix.

## Explicit native settings and work counters

The native tempering executable accepts an optional final settings line:

```text
FML_FIELD_HMC_1 0.075 16
```

It configures the field step/count and enables actual potential/gradient-call
counters. Settings are registered in checkpoint identity. This opt-in uses serial
replica checkpoint schema 2, which includes total and post-warmup field-gradient
counts. Legacy files without the extra line retain schema 1 and exact original
output/trajectories. Changing or disabling explicit settings rejects resume.
No public production-HMC defaults are changed.

`FieldHMC.h` invokes the existing HMC kernel with a counted field-only potential.
The metric remains constant during each conditional trajectory. Counts include
initial gradient evaluation, actual leapfrog evaluations and early nonfinite
termination; they are not inferred from nominal leapfrog counts. Scalar/cache/
metric work is outside this counter and included in wall time instead.

The new run.py maps the explicit JSON field settings to this native record and
checks returned metadata. The original frozen runner is unchanged.

## Freeze and run outside the library

Build the native tempering executable and calibration_diagnostics as usual; build
integrator_test here for independent harmonic-oscillator/count controls.

```sh
make integrator_test
./integrator_test
python3 compare.py prepare /absolute/comparison --source /absolute/pilot/mock204 --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
python3 compare.py run /absolute/comparison --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
python3 compare.py analyze /absolute/comparison --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
```

Preparation hashes every original prefix record, original checkpoints, all runtime
sources, plans and binaries before launching new chains. A study writer lock also
excludes concurrent analysis. Both four-worker cohorts run concurrently, one
checkpoint chunk per chain per round. `--max-rounds N` provides a bounded pause;
SIGTERM/SIGINT or the comparison PAUSE marker drains active native chunks. Resume
with `--resume`. A conservative cumulative four-hour wall budget uses the existing
2400-second reservation/refund ledger; a missing ledger beside native work rejects
restart. Failed or incomplete work stays visible.

Active cohort wall time sums each cohort's four-worker runner invocation; it
includes startup and replay overhead but excludes waiting for the other cohort
between rounds. Cohorts share hardware, so ESS/second is specific to this concurrent
run. If an unacknowledged crash loses a cohort time receipt, the cumulative ledger
keeps its reservation and analysis does not certify wall-time efficiency.

Analysis reuses the existing independent transport and twelve-coordinate checks.
It reports actual total/sampling field-gradient calls and bulk ESS per total field
gradient and per active cohort second for each coordinate. ESS/cost and posterior
summaries remain provisional when mixing gates fail. All failures and credible-
interval misses are kept; no rank/SBC conclusion is drawn from this selected case.

[Results](../../../docs/validation/studies/integrator_comparison/RESULTS.md) and
[the separate dataset-3 audit](../mixing_audit/README.md) describe their distinct
purposes. Raw chains, binaries and plots stay outside the library.
