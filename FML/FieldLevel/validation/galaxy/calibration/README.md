# Bounded prior-predictive calibration pilot

The study runner and analysis use the existing serial eleven-replica forward/sampler
driver. They add fixed prior-predictive datasets, cumulative sampling-time accounting,
likelihood-Q convergence diagnostics and independent-endpoint ranks. This is a small
same-model computational pilot, not precise coverage or real-catalog validation.

`plan.json` declares eight independent datasets (IDs 201–208), four chains each,
256 warmup and 2048 retained cold draws. No scalar/field/noise priors or kernels
change. Five scalars, six fixed field projections and the full weighted likelihood
quadratic Q must each satisfy Rhat<1.01 and bulk/tail ESS>=400, together with the
existing transport and zero-divergence gates. Failures never trigger extra draws,
new truths, weaker gates or a selected passing-subset calibration claim.

## Freeze before generating data

Build `calibration_diagnostics` here and `tempering_pilot` in `../tempering`, using
the normal external build configuration. Keep run outputs outside the library.

```sh
python3 check.py --diagnostics /absolute/calibration_diagnostics
python3 study.py prepare /absolute/study --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
python3 study.py run /absolute/study --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
python3 analyze.py /absolute/study --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
```

Preparation validates the entire mock set, generates eight immutable per-dataset
plans, enumerates generation/sampler/exchange key domains, and hashes the study,
plans, runtime analysis/runner sources and both executables into FROZEN.json. It
generates no nonlinear data. Changed source, binary, study or per-mock plans reject
resume. The native runner separately validates numerical controls and checkpoint
identity. Four initial scalar configurations remain independent of each true draw.

## Bounded sampling and pause/resume

Datasets run sequentially with four workers each. Each invocation of the existing
native runner advances at most one 16-sweep chunk per chain. A study writer lock
also excludes concurrent analysis; the child runner inherits that lock descriptor.
`--max-batches N` stops after N additional four-chain chunks, useful for checks.

SIGTERM/SIGINT, or touching the study's PAUSE marker, requests a graceful pause.
The parent signals the active runner, waits for its native checkpoint chunks and
replay operations to drain, then records paused.json. Resume the same study with:

```sh
python3 study.py run /absolute/study --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics --resume
```

The declared cumulative **16-hour sampling wall budget** includes native sampling,
refresh and replay time across invocations, but excludes offline analysis and idle
paused time. Before launch, the ledger reserves 2400 seconds for a batch; successful
acknowledged batches charge their measured wall time and refund the remainder.
A crash/unacknowledged batch keeps its entire reservation. A missing ledger alongside
existing native work rejects restart rather than silently resetting the budget. A full new reservation
must fit before another batch launches, so sampling can stop with less than 40
minutes of nominal budget unused. The budget is checked between batches; an active
batch may drain past its reservation, with the actual overrun charged. It is not
a hard wall-time kill. No automatic cap increase or budget-until-pass extension
is permitted. Retain incomplete datasets when capped, and report them explicitly.

## All-outcome analysis

Analysis can run on a frozen but unstarted, paused or completed study. Every
predeclared mock appears, with incomplete progress counts or the full diagnostics.
Completed mocks reconstruct every cold quadratic from saved predictions, data,
weights and noise power. Their targets must be byte-identical across chains and
match the frozen prior-predictive generation controls. Existing transport analysis
independently reconstructs all swaps, refreshed-prior trips and counters.

Native twelve-coordinate diagnostics use all retained cold draws. Per-dataset
calibration-summary.json retains every Rhat/ESS/status, gate, truth, central
50/90/95% interval and inclusion/miss, with source-record/checkpoint digests.
Nonfinite diagnostic values become JSON null with the failure status retained.

For each quantity, ranks use only the predeclared final endpoint from each of the
four independent chains. This gives ranks 0..4, with ties randomized using the
separate `[0x55, analysis_seed, mock_id, coordinate]` NumPy seed domain. Endpoint
stationarity still depends on the full-chain checks. Hot draws and replay records
are never included. Parameter and data-dependent likelihood ranks are both kept;
parameter-only ranks can miss a data-ignoring sampler, as the analytic control
in `../tempering/check_sbc.py` demonstrates.

Study calibration-summary.json and calibration-overview.png retain failed and
incomplete datasets. Histograms include every completed dataset, with failed-gate
ranks marked provisional. Eight datasets and five possible ranks do not support
a precise coverage assessment or an asymptotic chi-square uniformity decision.
The analysis deliberately makes no passing-subset or calibrated-recovery claim,
even if all computational gates pass. No new MPI execution or larger target is
validated by this driver.

See [implementation/control results](../../../docs/validation/studies/calibration_runner/RESULTS.md)
and [the statistical design](../prior_sensitivity/CALIBRATION_PLAN.md). The full
nonlinear pilot is separately frozen before launch; preparation itself is not
a calibration result.
