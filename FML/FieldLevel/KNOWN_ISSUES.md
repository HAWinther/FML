# FieldLevel review issues and deferred follow-ups

Recorded 29 October 2026; safety status updated 4 November 2026. **FL-1 and FL-2
were resumed and fixed at the user's request.** FL-4, FL-5 and FL-6 remain deferred.
The general-framework plan and initial periodic stages 1–3 are recorded in
[FRAMEWORK_PLAN.md](FRAMEWORK_PLAN.md). Safety fixes do not establish production readiness.

The original review examined commit `6edc763`. Engineering #17 was subsequently
implemented in `9205c5f`; its diagnostics fix is recorded separately below.
The relevant code is [FieldLevelBAO.h](FieldLevelBAO.h),
[the tests](tests/test_field_level.cpp) and [HMC.h](../MCMC/HMC.h).

## Correctness and verification status

- [x] **FL-1 (P1): Reject unsupported multi-rank use before serial indexing.**
  Resolved 4 November 2026: constructor validation queries the actual FML
  communicator before constructing the FFT grid. Unsupported multi-rank use
  throws deliberately, including when cached task metadata is stale. The test,
  example and calibration executable use the FMLSetup singleton; this works with
  automatic setup and NO_AUTO_FML_SETUP without a second MPI_Init. Frontends
  report unsupported rank counts and return failure before model construction.
  Bounds-checked tests pass on 1/2/4 ranks, with automatic and manual setup.
  The separate `DistributedLinearModel` added 12 November 2026 supports MPI
  linear inference; the original BAO and particle-LPT backends keep this guard.
  The historical evidence motivating the fix follows:

  The constructor enumerates all global cells/modes as local FFTWGrid entries.
  In a multi-rank host application, local slabs are smaller. A two-rank,
  mesh-8 build with `BOUNDSCHECK_FFTWGRID` aborts in `get_index_real`; without
  checks the same indices exceed local storage. Add an early communicator-size
  guard before the index/mode loops. Full MPI ownership/adjoints remain a
  separate extension; changing only loop bounds is insufficient.
  Acceptance: a supported serial run passes, and an unsupported multi-rank
  construction fails deliberately before local indexing or payload access.
  Also remove redundant raw `FML::init_mpi` calls in the test/example under
  default automatic FML setup: MPI-enabled builds currently attempt MPI_Init
  twice. The class-level reproduction used automatic setup alone to reach the
  independent indexing problem.

- [x] **FL-2 (P2): Validate public field/filter vector dimensions.**
  Resolved 4 November 2026: public conversions/filter validate dimensions before
  FFT scratch or mode access; to_grid has a shared field-size guard. Empty, short
  and oversized fields/filters throw invalid_argument. Valid conversion/filter
  round trips and posterior gradients are retained. Constructor settings are
  validated before grid allocation, including nonfinite values, invalid meshes
  and overflowing default density meshes. The focused suite passes 104 serial
  checks under Clang, GCC and ASan/UBSan. The existing serial suite also passes
  all 37 checks, including adjoints and HMC, under Apple Clang.
  The historical evidence motivating the fix follows:

  `forward` validates field length, while `filter`, `sampled_from_white` and
  `white_from_sampled` can reach `to_grid` without that check. `filter` also
  indexes `per_mode` without checking its length. On an otherwise valid model,
  `white_from_sampled(Field(1), 1.)` triggers an AddressSanitizer heap-buffer
  overflow; `filter(valid_field, {})` triggers undefined behavior/null access.
  Acceptance: invalid field and filter lengths throw `invalid_argument` before
  access; add negative tests for the public conversions and retain valid-input
  round-trip and gradient checks.

- [ ] **FL-4 (P2): Make the fixed-step CIC gradient test portable.**
  A clean Apple Clang serial double build fails the white 1LPT directional
  check. The reproduced analytic derivative was 43.2638 versus 43.2913 at
  h=1e-5, relative error 6.34e-4 against a 1e-5 tolerance. A step sweep gave
  errors 0.129 at 1e-3, 6.76e-3 at 1e-4, 6.34e-4 at 1e-5 and 3.87e-8 at
  1e-6; cancellation grew again at 1e-7. This is consistent with crossing
  piecewise-CIC cell boundaries, not evidence of an incorrect adjoint formula.
  Use step-convergence checks and perturbations away from knots to test local
  derivatives, with separate continuity/crossing tests. Seeded normal draws
  differ between standard libraries. Acceptance: meaningful checks pass under
  both Apple Clang and GCC without a blanket tolerance relaxation. GCC passed
  the full suite after #17; the Clang failure remains unresolved.

  Update (29 October 2026): the first 1LPT model started every particle exactly on
  a density-grid node, where CIC responds through |Psi| and the likelihood has a
  kink at Psi = 0 for every particle; this is the likely cause of the fragile
  fixed-step check. CIC particles now start mid-cell, and the default assignment
  is a smooth deconvolved kernel with a continuous gradient (see the README). The
  Apple Clang run has not been repeated, so this item stays open.

## Scientific and performance follow-ups

- [ ] **FL-5: Strengthen inference validation beyond same-model recovery.**
  The current 1LPT smoke check accepts truth within 3.5 posterior standard
  deviations and acceptance above 0.4. A prior-only uniform beta posterior on
  [0.8,1.2] has mean 1 and standard deviation about 0.115, satisfying that
  location criterion for beta_true=0.97. Thus it does not establish calibrated
  inference or detect every lost-likelihood failure. Add an explicit negative
  control for this case, posterior-information checks, multiple dispersed
  chains and repeated-mock coverage. Use higher-resolution/higher-cutoff mocks
  and mesh/CIC convergence to expose model mismatch; later include bias/noise
  nuisance inference. Monitor relevant field summaries as well as beta.
  The two-chain diagnostic checks added in #17 are regression checks, not
  certification of the full field posterior or scientific coverage.
  Progress: repeated-mock coverage is now measured by simulation-based
  calibration (`example/Calibration.cpp`, README "Simulation-based
  calibration"): 24 1LPT and 30 linear mocks with beta drawn from the prior give
  z-scores consistent with N(0,1) and uniform posterior quantiles. This covers
  same-model coverage only; the negative control, information checks,
  dispersed chains and model-mismatch tests above remain open.

- [ ] **FL-6: Measure and reduce force-call workspace churn.**
  `potential_and_gradient` recreates its workspace, displacement/adjoint fields
  and several mode vectors on each HMC force evaluation; `sample` also copies
  the full field into separate storage each time. Profile first, then consider
  reusable per-chain scratch and view-based inputs. Preserve gradient accuracy
  and measure allocation, time and peak RSS before claiming improvements.
  One model instance contains mutable FFT scratch: concurrent chains need
  separate instances/workspaces. No speedup was measured in the review.

## Resolved diagnostics issue

- [x] **FL-3 (P2): Frozen chains must not report maximal effective sample size.**
  Resolved by Engineering #17, commit `9205c5f`. Previously a constant
  1,000-element chain returned tau=1 and implied ESS=1,000; identical constant
  chains produced NaN legacy Rhat. Modern diagnostics now expose invalid-chain
  statuses, rank-normalized split/folded Rhat and bulk/tail ESS, with shared
  Metropolis/HMC adapters. Legacy IAT/Gelman-Rubin and Metropolis stopping fail
  closed on degenerate inputs. See [diagnostic contracts](../MCMC/README.md#convergence-diagnostics).
  Modern estimates are explicit post-run checks; the automatic stopping rule
  remains the legacy statistic. Longer chains and coverage validation are
  still part of FL-5.

## Resuming the work

The FL-1/FL-2 prerequisite is resolved. The initial periodic implementation of stages 1–3
is described in [the general-framework plan](FRAMEWORK_PLAN.md) and [its guide](FRAMEWORK.md). The deferred FL-4 numerical
verification, FL-5 scientific validation and measured FL-6 workspace work remain
separate follow-ups. The current Clang suite passes, but FL-4's portable
step-convergence acceptance criteria have not been completed. Recheck current
source before implementing the open entries; historical evidence can become stale.

## General framework sampler follow-up

The initial framework checks joint linear field/bias/noise inference against exact
field-marginalized quadrature with two dispersed chains and shared Rhat/ESS diagnostics.
Wide log-noise priors expose a narrow-noise geometry: rare divergent proposals remain
in the tested joint sampler despite good acceptance, Rhat/ESS and mean/width agreement.
These are explicitly reported in `tests/FRAMEWORK_RESULTS.md`; the regression accepts
a small measured proposal rate, not a claim of divergence-free production sampling.
Production users must investigate divergences, tune or reparameterize as needed,
and validate field summaries/tails and nonlinear/model-mismatch coverage. The field-only
linear Gaussian sampler is checked separately against its exact conditional covariance.
