# FieldLevel review issues and deferred follow-ups

Recorded 29 October 2026. **Remaining work is deferred at the user's request.**
Resume from this checklist before extending the prototype. This document records
review findings, not fixes or evidence of production readiness.

The original review examined commit `6edc763`. Engineering #17 was subsequently
implemented in `9205c5f`; its diagnostics fix is recorded separately below.
The relevant code is [FieldLevelBAO.h](FieldLevelBAO.h),
[the tests](tests/test_field_level.cpp) and [HMC.h](../MCMC/HMC.h).

## Open correctness and verification issues

- [ ] **FL-1 (P1): Reject unsupported multi-rank use before serial indexing.**
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

- [ ] **FL-2 (P2): Validate public field/filter vector dimensions.**
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

Suggested order: FL-1 and FL-2 safety guards, FL-4 numerical verification,
FL-5 scientific validation, then measured FL-6 workspace changes. Recheck the
current source before implementing: the evidence above describes the reviewed
versions and may become stale. Do not treat the open entries as implemented.
