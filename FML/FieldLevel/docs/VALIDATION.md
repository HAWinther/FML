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
