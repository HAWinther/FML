# FieldLevel review issues and deferred follow-ups

Recorded 29 October 2026; safety status updated 4 November 2026. **FL-1 and FL-2
were resumed and fixed at the user's request.** FL-4 is now resolved by portable
step-sweep and CIC-knot tests (13 November 2026). FL-5 has a bounded new
validation suite; production scientific calibration and FL-6 profiling remain open.
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

- [x] **FL-4 (P2): Make the fixed-step CIC gradient test portable.**
  Resolved 13 November 2026: the nine existing field/beta gradient cases now
  require a resolved interval of central-difference steps and improvement from
  the coarse step, rather than agreement at one hand-picked step. Two adjacent
  steps must agree within 2e-6 and the best within 1e-6 (relative with unit
  absolute floor); no blanket tolerance relaxation was made. Dedicated CIC
  tests use small deterministic fields away from cell boundaries, including
  centred/white 1LPT and biased/RSD 2LPT cases. A manufactured cell crossing
  tests continuity and converged, distinct one-sided slopes separately.
  A deliberately biased derivative fails the sweep criterion. Both Apple Clang
  and GCC validation are recorded in [tests/BAO_VALIDATION.md](tests/BAO_VALIDATION.md).
  Historical evidence follows:

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
  the full suite after #17; the original Clang failure was unresolved at that time.

  Update (29 October 2026): the first 1LPT model started every particle exactly on
  a density-grid node, where CIC responds through |Psi| and the likelihood has a
  kink at Psi = 0 for every particle; this is the likely cause of the fragile
  fixed-step check. CIC particles now start mid-cell, and the default assignment
  is a smooth deconvolved kernel with a continuous gradient (see the README). The
  Apple Clang run had not been repeated at that point, so this item stayed open
  until the step-sweep validation above.

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

  Update (13 November 2026): `tests/test_bao_validation.cpp` adds an explicit
  no-wiggle/prior-only negative control and posterior-width information criterion,
  two dispersed chains monitoring beta plus low-mode real/imaginary white-field
  projections and field power, divergence checks and configurable prior-drawn
  1LPT/2LPT mock ensembles. A test-only field-summary recorder is checked against
  the public BAO sampler for exactly unchanged trajectories. Independent direct
  particle Fourier sums test assignment/mesh convergence and reveal low-band
  changes from omitted higher-k initial modes. The NumPy summary reports empirical
  credible-interval coverage with binomial uncertainty. See
  [the validation guide](tests/BAO_VALIDATION.md) for measured results and limits.
  These stronger checks resolve the old prior-only smoke-test failure mode;
  the small strengthened-wiggle ensembles, manufactured fields and direct
  deposition references do not certify production cosmological/model-mismatch
  inference, joint bias/noise inference or field-posterior coverage.

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
is described in [the general-framework plan](FRAMEWORK_PLAN.md) and [its guide](FRAMEWORK.md).
FL-4's portable step-convergence acceptance criteria are now completed. FL-5
production scientific validation and measured FL-6 workspace work remain separate
follow-ups; the bounded FL-5 progress is documented above. Recheck current source
before implementing open entries; historical evidence can become stale.

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

## PM-S1: spatial force/particle accuracy (open, 15 November 2026)

The [matched-field resolution study](pm_resolution_validation/RESULTS.md) fails
its declared spatial accuracy criteria. At fixed 8^3 particles, changing force
mesh M=32 to 64 changes observed-band density by 21.9% and the projected posterior
gradient by 68.5%. At fixed force mesh M=32, changing particle lattice N=16 to 32
changes them by 16.8% and 50.9%. Time refinement and the adjoint directional tests
pass within the fixed spatial model.

An independent NumPy CIC/spectral-force/gather probe and a native tiny momentum
kick agree on the normalized longitudinal force response: for N=8, M=8/16/32/64
it is 0.900/1.322/1.788/3.023, versus continuum linear response 1. The agreement
localizes the discrepancy to the current discrete force/assignment setup. These
bounded fixtures do not select an optimal replacement kernel. Recovery from mocks
generated by the same discrete model does not establish continuum accuracy.

Progress: **native N>M support is implemented** with an IC band below both
Nyquists and aligned, supported process layouts. Forward/direct-DFT, adjoint,
posterior normalization and same-layout restart pass on slabs/pencils and sanitizers.
The [dense-sampling follow-up](pm_oversampling_validation/RESULTS.md) passes the
original finest-correction gates: at N=64, M=32 vs 64 changes density by 0.735%
and projected gradient by 6.198%. Particle refinements at M=16 and M=32 and time
refinement also pass. Native N=64/M=32 confirms the independent force response
0.991973. The original failed sparse-load study is retained.

The subsequent [122-mode study](pm_nonlinear_validation/RESULTS.md) uses two
initial-field seeds and three strength/time fixtures, with final density RMS
0.156/0.497/0.641 at N64/M32. Particle/time finest corrections pass, but M32 vs
M64 at N64 changes the full projected gradient by 16.3%/75.1%/60.6%. Both stronger
fixtures also exceed the 0.1-noise-SD criterion. Derivative/backend checks pass;
the spatial failures are retained. M64 has only one particle per force cell here,
which motivated a controlled force-reference follow-up before choosing a
numerical-law correction. Refined-data MAP is a
bounded optimization diagnostic, not posterior calibration: all three 12-iteration
fits improve recovery but remain unconverged. The expensive original 80-iteration
attempts were interrupted; optimizer work budgeting/conditioning remains open.

The [controlled forward follow-up](pm_force_reference_validation/RESULTS.md) now
checks N64/96/128 on both M32 and M64 for strong91. The last particle corrections
are 0.0648%/0.0424% in observed-band density and the N128/M64 time correction is
0.0119%. M32→M64 at N128 still changes density by 3.7749% (0.1063 noise SD per
observed degree, failing 0.1). Its stability across particle loads points to
force-grid discretization as the main density sensitivity in this band. These
finite, forward-only checks resolve the earlier particle-load ambiguity for this
fixture; they do not close posterior-gradient, full-mesh/high-cutoff or MAP issues.
No force law or checkpoint identity is changed.

An [explicit compensated-force option](pm_window_validation/RESULTS.md) now has a
verified adjoint and distinct law-tag-2 checkpoint identity. In the controlled
N128 comparison it reduces the raw density correction to 1.768% (0.0501 noise
SD), and a common-window postprocessing diagnostic to 0.0309%. Raw joint-gradient
changes remain 34.9% versus the 15% criterion. Sparse compensation strongly
amplifies alias/lattice errors; dense linear-force controls improve, not a general
accuracy guarantee. A coarse tiny-kick measurement failed and is retained;
1e-9/1e-10 refinements pass unchanged tolerances. The explicit fixed observation response is now implemented (see below); broader
gradient refinement and posterior validation remain open.

Follow-ups before a broad PM accuracy claim:

1. Extend these finite-reference, one-realization checks to more physical modes,
   amplitudes and nonlinear regimes, with refined/independent data and inference
   checks. Dense sampling costs N^3 particle/white storage; a universal N/M ratio
   or production-accuracy guarantee has not been established.
2. Evaluate any assignment or force-window/Green-function change as an explicit
   numerical-law choice with matched physics, derivative tests and external/analytic
   references. Dense particles do not eliminate finite force-window errors.
3. Bump the PM checkpoint numerical-law identity if the force/assignment law changes.
   Recheck inference against refined or independent data before adding coverage
   or production claims.

The default numerical law is preserved; no hidden filter or rescaling has been
introduced. Its default checkpoint tag remains 1: old valid targets are unchanged, and
new mesh orderings are encoded by existing layout metadata. Changed-layout
restart and survey/tracer/RSD remain separate.

## Fixed PM observation convention (15 November 2026)

The [fixed-response study](pm_observation_validation/RESULTS.md) now implements the
actual signal response, transpose and checkpoint identity. With a common M32 CIC
observation window, compensated M32→M64 density and joint-gradient corrections
are 0.0309% and 0.476%, passing the original bounded criteria. The uncompensated
joint-gradient baseline still fails at 41.6%. Custom real coefficients, noise after
response, both distributed backends and legacy bitwise compatibility are tested.
This is one finite-reference realization; adequate broader gradient sampling,
posterior calibration, MAP conditioning and PM survey/tracer/RSD work remain open.
