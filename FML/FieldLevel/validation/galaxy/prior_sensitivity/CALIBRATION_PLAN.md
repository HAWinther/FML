# Proposed next validation: prior-predictive calibration

Status: the bounded eight-dataset study is now frozen but unstarted; see
[the runner controls and exact manifest](../../../docs/validation/studies/calibration_runner/RESULTS.md). The
[truth-generation/legacy/restart and analytic-rank controls](../../../docs/validation/studies/prior_predictive_controls/RESULTS.md)
are now implemented and validated. Campaign orchestration/Q diagnostics and the
final frozen manifest are now implemented; launching that exact pilot is next. The completed
[prior-sensitivity analysis](../../../docs/validation/studies/prior_sensitivity/RESULTS.md)
does not replace this stage. Keep its baseline bounded-uniform physical priors;
choosing an alternative using known-truth recovery would invalidate the rationale
for an independent check.

## End goal and experiment

Check whether the implemented simulator and posterior computation agree when the
*entire truth* is drawn from the fitted prior. For each independent dataset, draw
all five physical scalars from the exact prior, all 64 white coordinates from
their standard Gaussian prior, then Gaussian data noise from an independent RNG
domain. Generate data with the unchanged 2LPT model, cutoffs and noise power.
Initial sampler states must not use any generating truth. Keep inactive white
coordinates in the generation and sampling contract.

Two fixed-scalar mocks cannot be repurposed as prior-predictive draws. Nor can
importance reweighting an existing posterior create a new prior-predictive dataset.
This stage checks same-model computation; galaxy/HOD model mismatch, the actual
catalog, a larger discretization and production MPI replica orchestration remain
separate tasks.

## Implementation before freezing a campaign

1. Add an explicit opt-in prior-predictive truth mode. Preserve the current native
   metadata/checkpoint format and fixed-truth trajectories. Domain-separate scalar
   truth, white truth, noise, initial fields and every sampler key. Reserve distinct
   mock IDs from the original and two recovery mocks. Validate integer bounds and
   enumerate the full declared sampler key budget for collisions.
2. Serialize mode, prior identity, seeds and generated physical truth in the target
   record and bind all controls to checkpoint identity. Changed generation controls
   must reject restart. Every ensemble must produce identical data independently.
3. Check scalar draws against the exact declared prior, white draws against their
   Gaussian law, and simulation noise against its covariance. Check independent
   seed effects, cached/full targets and actual short positive-beta updates on the
   new generated datasets. Require fresh-process exact replay and pause/resume.
4. Add independent analytic SBC controls (normal-normal model) and an intentionally
   incorrect data-ignoring posterior. Include data-dependent test quantities so the
   data-ignoring control is detectable; marginal parameter ranks alone can miss it.
5. Save final protocol, source/binary/input hashes, IDs and budgets *before*
   generating nonlinear outcomes. Do not silently freeze this proposal as a run.

## Suggested bounded nonlinear pilot

Use eight datasets with four independent replica-exchange ensembles per dataset,
256 warmup and 2048 retained cold draws each, with the existing eleven temperatures,
four exchange passes and kernels. Run one dataset at a time (four workers) so
runtime and thermal load stay bounded. At prior-mock timing this is several hours;
prior-predictive extremes can be harder. First measure implementation preflight
cost; set and record an explicit wall-time cap before the run. Pause at checkpoint
boundaries, retain partial outcomes, and resume only within the declared budget.

Use the unchanged eleven monitored convergence gates: Rhat<1.01, bulk/tail ESS>=400,
zero measured positive-beta divergences and the existing transport requirements.
Add the data-dependent likelihood discrepancy to the monitored quantities and
apply the same scalar diagnostics. Record every dataset whether gates pass or
fail. Never replace a truth, extend until pass, discard chains, or declare SBC on
a selected passing subset. Failed-mixing ranks may be shown as provisional, but
an overall calibration conclusion stays open. Eight datasets are a debugging
pilot, not a precise coverage assessment or a powerful uniformity test.

## Ranks without treating correlated cold records as independent

For a fixed and coarse rank diagnostic, use only the final retained draw (index
2047) from each of the four independently seeded ensembles for each dataset.
This gives L=4 draws and ranks on {0,1,2,3,4}. With exact posterior endpoints,
conditional independence follows from independent chains; no within-chain draws
are pooled into this rank. Approximate stationarity remains subject to the full
chain gates. The index must be frozen in advance and not selected by its value.
This trades rank resolution for avoiding a questionable thinning assumption.

Use these same endpoint configurations for each of the five scalars, six fixed
field projections, and the full data likelihood discrepancy Q. Compute Q at the
truth and each endpoint on the *same generated data* and noise convention. For
exact ties randomize the truth's position among equal values with an independent,
predeclared analysis RNG; retain tie counts and seeds. Never resample endpoints
or use the rank RNG in simulation or sampling.

Show each dataset's rank and diagnostics. Any rank histogram has only eight counts
across five possible ranks, so omit asymptotic chi-square tests and a binary
uniformity gate. A simultaneous finite-sample band would require a separately
specified design with enough independent datasets. If more resolution is needed,
freeze an autocorrelation-aware thinning protocol in a later study; an overlap
index from importance weights is not an effective sample size for SBC.

Use all retained cold draws for descriptive intervals, with inclusion/misses
reported per dataset and coordinate; do not call the endpoints a high-precision
credible-interval estimate. Keep MC precision and prior-averaged coverage distinct
from fixed-truth frequentist coverage. Do not infer scientific accuracy from a
same-model calibration pass.

## References behind the design

[Talts et al., Validating Bayesian Inference Algorithms with Simulation-Based
Calibration](https://arxiv.org/abs/1804.06788) explains prior-predictive ranks and
the issue with correlated MCMC samples. The independent-endpoint choice above is
our deliberately coarse pilot design, not a recommendation attributed to that
paper.

[Modrak et al., Simulation-Based Calibration Checking for Bayesian Computation:
The Choice of Test Quantities Shapes Sensitivity](https://paulbuerkner.com/publications/pdf/2023__Modrak_et_al__Bayesian_Analysis.pdf)
explains why parameter-only checks can miss incorrect inference and motivates
including a data-dependent likelihood quantity.
