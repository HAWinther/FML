# Validation records

These are historical, bounded studies. Reports retain their original conclusions,
including failures. Large inputs/outputs and exploratory source are in the
[verified archive](ARCHIVE.md); compact outcome excerpts are in
[outcomes.json](validation/outcomes.json). Maintained driver entry points are
[listed separately](../validation/README.md).

## pm_force_reference_validation

[Recorded report](validation/studies/pm_force_reference_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_full_validation

[Recorded report](validation/studies/pm_full_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_hmc_validation

[Recorded report](validation/studies/pm_hmc_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_nonlinear_validation

[Recorded report](validation/studies/pm_nonlinear_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_observation_validation

[Recorded report](validation/studies/pm_observation_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_optimizer_validation

[Recorded report](validation/studies/pm_optimizer_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_oversampling_validation

[Recorded report](validation/studies/pm_oversampling_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_reconstruction_validation

[Recorded report](validation/studies/pm_reconstruction_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_resolution_validation

[Recorded report](validation/studies/pm_resolution_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_stable_armijo_validation

[Recorded report](validation/studies/pm_stable_armijo_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_stable_stationarity_validation

[Recorded report](validation/studies/pm_stable_stationarity_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_stationarity_validation

[Recorded report](validation/studies/pm_stationarity_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_tsc_joint_validation

[Recorded report](validation/studies/pm_tsc_joint_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_tsc_posterior_validation

[Recorded report](validation/studies/pm_tsc_posterior_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## pm_tsc_validation

[Recorded report](validation/studies/pm_tsc_validation/RESULTS.md). Recorded overall flag: `passed=true`.

## pm_window_validation

[Recorded report](validation/studies/pm_window_validation/RESULTS.md). Recorded overall flag: `passed=false`.

## survey_validation

[Recorded report](validation/studies/survey_validation/RESULTS.md). Consult the report for its scope and individual verdicts.

## Recent posterior milestones

[Repeated mocks](validation/studies/pm_tsc_posterior_validation/REPEATED_MOCK_RESULTS.md):
three fresh N16 mocks pass their sampling/consistency checks; amplitude intervals
contain truth in 2/3 and noise intervals in 3/3. The fourth was user-skipped.
This small fixed-truth sample does not establish coverage.

![Repeated-mock summary](images/pm-repeated-mocks.png)

[Long fixed-schedule run](validation/studies/pm_tsc_posterior_validation/LONG_VALIDATION_RESULTS.md)
and [independent mock](validation/studies/pm_tsc_posterior_validation/INDEPENDENT_MOCK_RESULTS.md)
retain their original results. Earlier mixing failures remain in the archive.

## Fixed-IC amplitude pilot

[Three-mesh synthetic field result](validation/studies/galaxy_fixed_ic/RESULTS.md):
all 12 declared chains completed, but the bulk-ESS mixing gate is not passed.
Observed mean shifts are small; their Monte Carlo uncertainty does not certify
the 0.1 SD numerical budget. Raw chains/plots remain outside the library.

The [separate joint-proposal follow-up](validation/studies/galaxy_fixed_ic_joint/RESULTS.md)
passes the same gates on the same data and budgets, preserving the original
failed pilot. Alpha ESS improves 1.75–2.40 times per forward call. This is still
a small fixed-IC Gaussian-field validation, not a galaxy-catalog/free-IC result.

## Periodic catalog observations

[Streaming catalog adapter results](validation/studies/periodic_catalog/RESULTS.md):
serial and slab/pencil 1/2/4-rank direct-sum checks pass, with 17 CSV/HDF5 interface
checks. Maximum absolute mode error is 5.55e-9 at requested epsilon=1e-8.
This is fixed-observation ingestion, not NUFFT matter evolution or a new
galaxy-catalog amplitude posterior.

## PSWF matter deposition and position adjoint

[Recorded numerical results](validation/studies/nufft_matter/RESULTS.md):
serial and slab/pencil 1/2/4-rank checks pass for PSWF deposition and the matched
position/weight reverse through 1LPT/2LPT, Lagrangian bias/RSD and Eulerian bias.
Smooth, checkpoint and catalog regressions pass. Compact-stencil boundaries
and derivative accuracy are measured separately from Fourier value accuracy.
This adds no new posterior campaign or production convergence claim.

## Compact latent/product meshes

[Recorded compact-mesh validation](validation/studies/compact_meshes/RESULTS.md):
serial and slab/pencil 1/2/4-rank checks pass for separate sampled and product
meshes, with a fixed-resolution padded reference. Predictions agree to roundoff;
field/scalar derivatives, prior, mock noise, metric and short HMC checks pass.
The fixture has 512 rather than 1728 sampler field coordinates; total RSS and
production mixing are not measured. Shared-mesh, fixed-IC and catalog regressions pass.

## Eulerian checkpoint and restart

[Recorded restart validation](validation/studies/eulerian_restart/RESULTS.md):
serial and slab/pencil 1/2/4-rank fresh-process continuation reproduces final
state/metrics/statistics and draw suffixes byte-for-byte. Failed generation writes,
corrupt shards and changed targets reject collectively. The indexed-output example
replays orphan records correctly. Legacy/PM suites and a pre-refactor checkpoint
writer/current reader compatibility check pass. These are bounded restart checks,
not a new scientific posterior campaign.

## Bounded Eulerian numerical/resource screen

[Recorded screen](validation/studies/numerical_screen/RESULTS.md): all 17 serial/MPI
cases pass frozen mesh/tolerance/gradient/smoke gates on one synthetic data vector.
The default N8/N12/N24, epsilon=1e-8 has noise-weighted distance 0.00123 and median
forward+gradient time 0.037 s on this host. Actual process RSS establishes no total
memory saving from compact storage on this small fixture. Fisher quantities are
local proxies, not posterior-mean convergence. A separate field-metric noise/response
issue (FL-7) should be addressed before longer sampling.

## FL-7 posterior-aware Eulerian field metric

[Recorded fix and comparison](validation/studies/eulerian_metric/RESULTS.md):
78 serial and 156/160/160 MPI metric checks pass, including independent linear
Hessian, actual Gaussian noise/mask/volume, compact geometry, cache, MAP and
restart checks. The unchanged target compares byte-for-byte with the pre-fix
probe. Six field-projection bulk ESS values improve from 4.5--4.9 to 171--261
at the same fixed budget; max corrected Rhat=1.023 still needs longer mixing
validation. Pre-fix Eulerian checkpoints are rejected by the new metric identity;
legacy backend checkpoint compatibility is retained.

## Joint Eulerian initial-field/amplitude/bias pilot

[Recorded result](validation/studies/joint_pilot/RESULTS.md): four chains each
complete 256 warmup and 1024 retained draws using the FL-7 metric. The frozen
eleven-coordinate mixing gate is **FAIL**; no extension or retuning. A separate
64-draw fresh-process continuation replays byte-for-byte and is excluded from
diagnostics. This is a synthetic same-model test, not coverage or real data.

## Blocked field/cached-scalar comparison

[Recorded result](validation/studies/blocked_pilot/RESULTS.md): four chains,
256 warmup and 1024 retained draws; eleven-coordinate mixing gate **FAIL**.
The target/initial states/gates match the preceding joint pilot. Cache/full-target,
bias derivatives, independent scalar quadrature and exact restart checks pass.
No extension, retuning or baseline/replay pooling. This is a serial validation
prototype, not a production MPI blocked-sampler API or scientific calibration.

### Amplitude/field interweaving (18 November 2026)

[Recorded comparison](validation/studies/interweave_pilot/RESULTS.md): 26 dedicated kernel tests, 18 full-target/cache checks, 72 bias derivatives and exact native replay pass. Four chains each complete 256/1024 discarded/retained sweeps. Frozen mixing gate: **FAIL**. No extension, retuning, excluded chains or pooling; serial validation prototype, not production MPI support or scientific calibration.

### Geometry audit (18 November 2026)

[Geometry audit](validation/studies/geometry_audit/RESULTS.md): 1010 deterministic points, exact convex bias profiling, three independent optimizer references, 32 endpoint derivative comparisons and a rescaling control. Profiled barriers remain on nine selected paths. A reuse-only temperature scan motivates a separate replica-exchange test; no new Monte Carlo draws, evidence or coverage claim.

### Replica-exchange connectivity (18 November 2026)

[Replica-exchange connectivity](validation/studies/tempering_connectivity/RESULTS.md): eleven kernel checks, positive-beta target/cache consistency, actual beta=0 prior refresh and byte-exact ensemble replay pass. Short fixed-budget connectivity gate **FAIL**. Only beta=1 records are cold states; no posterior Rhat/ESS, coverage, production MPI support or real-catalog claim.

### Configurable prior-bridge refinement (18 November 2026)

[Configurable prior-bridge connectivity](validation/studies/tempering_bridge/RESULTS.md): 25 kernel checks, ten beta contexts, thirty cache comparisons, exact legacy retained continuation and whole-ensemble replay pass. Fixed 64/256 two-ensemble gate **PASS**. Prior bounces without refresh do not count. No posterior Rhat/ESS, coverage, production MPI or real-catalog claim.

### Four-cold-chain posterior validation (18 November 2026)

[The separate frozen posterior campaign](validation/studies/tempering_posterior/RESULTS.md) completes four 256/1024 chains with eleven replicas. Posterior gate: **FAIL** (2/11 coordinate gates fail); connectivity: **PASS**. Complete replay, legacy continuation, pause/SIGTERM/stale-summary and exclusive-writer checks pass. Numerical posterior stability, independent-mock calibration and real-catalog scientific validation remain separate.

### Longer four-cold-chain posterior validation (18 November 2026)

[The separate frozen posterior campaign](validation/studies/tempering_long/RESULTS.md) completes four 256/2048 chains with eleven replicas. Posterior gate: **PASS** (0/11 coordinate gates fail); connectivity: **PASS**. Complete replay, legacy continuation, pause/SIGTERM/stale-summary and exclusive-writer checks pass. Numerical posterior stability, independent-mock calibration and real-catalog scientific validation remain separate.

### Posterior numerical-refinement check (18 November 2026)

[The frozen six-case posterior sensitivity check](validation/studies/posterior_refinement/RESULTS.md) is **PASS** on all 8192 existing cold states. Same latent coordinates, data, priors, noise and physical bands. Max paired mean-shift bound: 1.62e-07 posterior SD; max sampled model distance: 1.58e-05. Importance overlap remains adequate; this is not new refined-target MCMC or independent-mock calibration. Source checkpoints are unchanged. Next: independently frozen mock recovery/calibration.

### Independent fixed-truth mock recovery (18 November 2026)

[Both predeclared independent-mock budgets](validation/studies/independent_mocks/RESULTS.md) complete four fresh 256/2048 chains. Computational gate: **PASS**; all outcomes and interval misses preserved. Exact namespace, legacy, data-seed, whole-study pause and eight-chain final checkpoint checks pass. Recovery/interval ranks are descriptive; two fixed-truth mocks are not prior-predictive SBC or coverage calibration.

### Bounded prior sensitivity (18 November 2026)

[Four frozen prior cases on all three completed datasets](validation/studies/prior_sensitivity/RESULTS.md)
pass every overlap/mean-shift precision screen. Amplitude mean shifts are -0.23 to
-0.27 baseline SD for log-amplitude priors, +0.12 to +0.41 SD for nonlinear-bias
regularization. Source chains/checkpoints are unchanged. This measures prior
sensitivity, not coverage; prior-predictive truth controls and a separately frozen
calibration campaign are next.

### Prior-predictive truth controls (18 November 2026)

[Opt-in prior-predictive generation controls](validation/studies/prior_predictive_controls/RESULTS.md)
pass independent scalar/white/noise references, actual Fourier noise/target checks,
three-layout byte-exact legacy trajectories, actual completed-checkpoint reloads
and a four-chain 64-sweep pause/replay smoke. Analytic SBC controls include a
data-ignoring counterexample detected by a likelihood quantity. No long nonlinear
SBC campaign is started; campaign orchestration/Q diagnostics and frozen budgets
are next. No production MPI orchestration or coverage claim.

### Bounded calibration orchestration (18 November 2026)

[Study orchestration and likelihood-Q diagnostics](validation/studies/calibration_runner/RESULTS.md)
pass fresh two-dataset 4x48 controls, actual SIGTERM/lock/resume, immutable identity
and crash/time-budget checks. Both short datasets retain all twelve failed mixing
gates and provisional ranks. The full eight-dataset four-chain 256/2048 pilot is
frozen with a cumulative 16-hour sampling cap, but is **not started**. Next: run
that exact pilot; no nonlinear calibration or production MPI claim yet.
