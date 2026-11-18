# FieldLevel review issues and deferred follow-ups

Recorded 29 October 2026; safety status updated 4 November 2026. **FL-1 and FL-2
were resumed and fixed at the user's request.** FL-4 is now resolved by portable
step-sweep and CIC-knot tests (13 November 2026). FL-5 has a bounded new
validation suite; production scientific calibration and FL-6 profiling remain open.
The general-framework plan and initial periodic stages 1–3 are recorded in
[FRAMEWORK_PLAN.md](FRAMEWORK_PLAN.md). Safety fixes do not establish production readiness.

The original review examined commit `6edc763`. Engineering #17 was subsequently
implemented in `9205c5f`; its diagnostics fix is recorded separately below.
The relevant code is [FieldLevelBAO.h](../FieldLevelBAO.h),
[the tests (archived)](ARCHIVE.md) and [HMC.h](../../MCMC/HMC.h).

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
  and GCC validation are recorded in [tests/BAO_VALIDATION.md](validation/tests/BAO_VALIDATION.md).
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
  [the validation guide](validation/tests/BAO_VALIDATION.md) for measured results and limits.
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
  closed on degenerate inputs. See [diagnostic contracts](../../MCMC/README.md#convergence-diagnostics).
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

The [matched-field resolution study](validation/studies/pm_resolution_validation/RESULTS.md) fails
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
The [dense-sampling follow-up](validation/studies/pm_oversampling_validation/RESULTS.md) passes the
original finest-correction gates: at N=64, M=32 vs 64 changes density by 0.735%
and projected gradient by 6.198%. Particle refinements at M=16 and M=32 and time
refinement also pass. Native N=64/M=32 confirms the independent force response
0.991973. The original failed sparse-load study is retained.

The subsequent [122-mode study](validation/studies/pm_nonlinear_validation/RESULTS.md) uses two
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

The [controlled forward follow-up](validation/studies/pm_force_reference_validation/RESULTS.md) now
checks N64/96/128 on both M32 and M64 for strong91. The last particle corrections
are 0.0648%/0.0424% in observed-band density and the N128/M64 time correction is
0.0119%. M32→M64 at N128 still changes density by 3.7749% (0.1063 noise SD per
observed degree, failing 0.1). Its stability across particle loads points to
force-grid discretization as the main density sensitivity in this band. These
finite, forward-only checks resolve the earlier particle-load ambiguity for this
fixture; they do not close posterior-gradient, full-mesh/high-cutoff or MAP issues.
No force law or checkpoint identity is changed.

An [explicit compensated-force option](validation/studies/pm_window_validation/RESULTS.md) now has a
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

The [fixed-response study](validation/studies/pm_observation_validation/RESULTS.md) now implements the
actual signal response, transpose and checkpoint identity. With a common M32 CIC
observation window, compensated M32→M64 density and joint-gradient corrections
are 0.0309% and 0.476%, passing the original bounded criteria. The uncompensated
joint-gradient baseline still fails at 41.6%. Custom real coefficients, noise after
response, both distributed backends and legacy bitwise compatibility are tested.
This is one finite-reference realization; adequate broader gradient sampling,
posterior calibration, MAP conditioning and PM survey/tracer/RSD work remain open.

## Refined-data compensated PM reconstruction (15 November 2026)

The [new reconstruction study](validation/studies/pm_reconstruction_validation/RESULTS.md) generates
a compensated N128/M64/32-step mock and fits at N32/M16/16 steps with a common
M16 observation window and fixed nuisance parameters. At 24 accepted iterations,
initial coefficient error is 0.2696 of the zero-field baseline,
correlation 0.9643, and residual 0.7881 noise SD. Optimizer
status is MaximumIterations; raw and projected gradients are retained with the plots.
Serial/MPI slab/pencil/sanitizer replay and legacy bitwise comparisons pass.
A costlier N64/M32 attempt is retained; the public optimizer is unchanged.
Broader seed/amplitude/refinement checks remain explicitly deferred by request;
this is reconstruction validation, not posterior calibration. Item 52 remains open.

## MAP optimizer and frozen-data comparison (15 November 2026)

`MAP.h` adds an options overload with limited-memory L-BFGS and value-only
gradient descent; legacy `map()` is unchanged. `BasicPosterior::value` omits the
forward-model adjoint while retaining the normalized target and priors. Distributed
field/scalar reductions, controls, curvature and history limits are tested.
[The frozen-data comparison](validation/studies/pm_optimizer_validation/RESULTS.md) preserves legacy24
coefficients bitwise; value-only GD cuts pullbacks from 144 to 25.
L-BFGS status is LineSearchFailed, raw gradient 0.000116038 at
83 iterations with tolerance2e-5. Its recovery/error and work/memory
measurements are reported separately. Broader refinement remains deferred, and
posterior calibration/PM survey/tracer/RSD work remains open.

## Stalled PM reconstruction diagnosis (16 November 2026)

[Read-only stationarity probes](validation/studies/pm_stationarity_validation/RESULTS.md) reproduce
iteration83 and capture its full state. Fresh/reused caches and replay budgets
0/1/4/8 give identical gradients/values; three stencil-preserving directional
checks pass. A particle lies3.19e-10 force cells from a final CIC face. Larger
trials cross stencils, while smaller smooth-branch decreases around2e-12 are
masked by ordinary objective evaluation/subtraction errors around1e-11. The
public value/full base targets agree; no cache/adjoint discrepancy was observed
in these tests. Solver and assignment are unchanged. Next test stable objective
change comparisons for Armijo before another full reconstruction. Broader
refinement remains deferred; convergence and general calibration remain open.

## Stable Armijo comparisons (16 November 2026)

`BasicPosterior` gains an opt-in stable change API for likelihoods implementing
local differences; Gaussian variance/normalization and field/scalar priors are
included. Defaults and ordinary target evaluations remain unchanged.
[The frozen-state/full-fit check](validation/studies/pm_stable_armijo_validation/RESULTS.md) accepts
the diagnosed hidden descent. Stable L-BFGS uses 594 value trials versus
1311, but ends at iteration 98 with raw gradient 0.00013902
and status LineSearchFailed: convergence remains unresolved. Stable GD
retains its 24-step coefficients bitwise. MPI/sanitizer and default-compatibility
checks pass. Broader refinement/calibration remains deferred; next inspect CIC
regularity/forward precision at the new endpoint before changing assignment.

## Stable endpoint regularity diagnostic (16 November 2026)

[Iteration-98 probes](validation/studies/pm_stable_stationarity_validation/RESULTS.md) reproduce the
stable fit bitwise. A final-node CIC coordinate lies 7.47e-13 grid cells from a
face. One-sided negative-gradient slopes differ across two stencil axes: the
branch-preserving side agrees with the adjoint, while the nominal descent side
increases the target. All 40 reconstructed search trials have positive stable
changes; ordinary subtraction would accept 17 tiny apparent improvements.
Fresh/reused caches and checkpoint budgets agree exactly. Two central derivative
gates remain unresolved because no eligible same-stencil pairs exist; the report
retains passed=false and consistency_passed=true. Convergence remains open.
Next consider opt-in TSC with matched deposit/gather derivatives and adjoint,
preserving CIC defaults. No assignment or stopping-rule change was made. Broader
refinement/calibration remains deferred.

## Opt-in TSC PM assignment (16 November 2026)

[Matched-assignment reconstruction](validation/studies/pm_tsc_validation/RESULTS.md) converges in
16 iterations with raw gradient 5.6502e-06 at the unchanged
2e-5 tolerance. TSC covers force deposition/gather, reported density, matching
adjoint and optional force compensation. CIC defaults and checkpoint identities
remain compatible. Targeted serial/MPI/pencil/sanitizer and checkpoint checks pass.
The matched TSC reference changes the discrete target; this is one finite
realization, not general convergence or posterior calibration. Broader refinement
remains deferred. Stable objective comparisons remain opt-in.

## Joint TSC MAP validation (16 November 2026)

[Initial field, amplitude and noise](validation/studies/pm_tsc_joint_validation/RESULTS.md) converge
jointly in 54 iterations with joint norm 9.84911e-06
at tolerance 2e-5. Scalar gradient identities and six refined directional gates
pass; serial/slab/pencil/sanitizer checks agree and fixed mode reproduces the
committed TSC driver bitwise. No library or optimizer code changed. This study
explicitly widens amplitude support to [-4,4]; Gaussian mean/scale remain fixed.
The high joint-MAP amplitude follows the whitened-coordinate density mode and
must not be interpreted as an unbiased or marginalized amplitude estimate.
Physical initial-field correlation is 0.964540. General scalar
posterior inference, coverage and broader refinement remain deferred.

## Bounded TSC posterior pilot (16 November 2026)

[Active-field/amplitude/noise HMC](validation/studies/pm_tsc_posterior_validation/RESULTS.md) completes
four N16 particle chains on the frozen data, M16 force mesh and 16 time steps,
as selected by the user. Exact inactive-Gaussian marginalization and physical
coordinates include the 122*A/2 Jacobian; direct target/gradient checks pass.
All Rhat values are <1.05 and there are no retained divergences, but bulk/tail ESS
criteria remain failed (minima 93.3/180.9); the report
keeps passed=false. Scalar intervals and predictions are exploratory. N32 mixing
and coverage remain unvalidated. A <=16-mesh bounded dense lookup reduces a short
case's CPU time about 21% with bitwise-identical outputs; larger meshes stay sparse.
PM/LPT, MPI/pencil, sanitizer and old-checkpoint checks pass. Broader refinement
remains deferred; more effective draws are needed before calibrated inference.

## Longer N16 TSC posterior mixing validation (16 November 2026)

[The fixed-schedule 512-draw run](validation/studies/pm_tsc_posterior_validation/LONG_VALIDATION_RESULTS.md)
clears all original finite-chain gates for the declared N16 periodic PM target:
four chains, maximum 6 leapfrog steps, 64 warmup and 512 retained draws each.
Across all 248 active-field/squared-field/scalar/potential/norm features, maximum
Rhat is 1.0071, minimum bulk ESS 471.1 and minimum tail ESS 726.0, with zero
retained divergences. The 192-draw prefix and saved analysis reproduce exactly.
The earlier 128/192-draw reports retain their failed flags; longer sampling with
an increased declared budget resolves this bounded target's mixing gate.
N32 mixing, repeated-mock coverage and a scalable production full-grid metric
remain open. The dense metric and scalar-cache adapter remain validation code;
production sampler defaults are unchanged.

## Independent same-model N16 posterior validation (17 November 2026)

[The independent mock](validation/studies/pm_tsc_posterior_validation/INDEPENDENT_MOCK_RESULTS.md)
uses fresh IC191/noise192 with matching N16/M16/16-step generation and inference.
The frozen four-chain 64-warmup/512-retained protocol passes all original 248
feature gates: maximum Rhat 1.0064, minimum bulk/tail ESS 463.9/552.9 and zero
retained divergences. Both injected scalar parameters lie within their 95%
marginal intervals; initial-field mean correlation is 0.9775. Recovery is
descriptive on one mock, not repeated-mock coverage. A data-only MAP and fresh
fixed metric precede sampling; no post-result retuning. New generator/seed
plumbing passes serial, slab/pencil MPI 1/2/4-rank and sanitizer checks, with
legacy default outputs preserved. The full science chains run in serial.
The original refined-data reference remains separate. N32 mixing, physical
convergence, multiple-mock calibration and scalable full-grid metrics remain open.

## Repeated same-model N16 posterior validation (17 November 2026)

[Results and plots](validation/studies/pm_tsc_posterior_validation/REPEATED_MOCK_RESULTS.md).

The original same-model N16 plan declared IC/noise seeds 291/292, 391/392,
491/492 and 591/592, excluding the previous 191/192 case. The user requested
skipping the fourth after two results were known and the third was running.
Three datasets participate; the fourth is user-skipped, not missing or failed. The frozen
four-chain 64-warmup/512-retained schedule and all 248 strict feature gates stay
unchanged. Completed: 3/3; mixing passes: 3/3;
consistency passes: 3/3; campaign passed=True.
No failed case is replaced, extended or dropped. Each dataset has a data-only
MAP and a fresh metric frozen before sampling. The report separates individual
mixing verdicts from scalar interval inclusion and field recovery. Three datasets
give broad coverage uncertainty; correlated field-mode counts are not pooled.
This is limited repeated-data recovery at fixed true hyperparameters, not SBC.
Native sources/binaries remain unchanged; synthetic reporting and raw/aggregate
replay checks are recorded. N32 mixing, physical convergence, larger calibration
ensembles and scalable full-grid metrics remain open. This validation pilot
buffers chain output until completion and has no mid-chain restart checkpoint;
finished mocks are persistent, but interrupted unfinished chains must restart.


## FL-7 (P2), resolved: Eulerian metric noise and tracer response

Resolved 18 November 2026. The posterior-aware metric now uses the actual
Gaussian noise policy, covariance volume and likelihood mask with linear response
`b_delta-b_laplacian*k^2`; inactive modes retain identity precision. HMC freezes the
operator at its initial scalar snapshot; MAP uses its current point. Precision
caching is collective across MPI mask changes and uses the sampled grid directly.
The Eulerian restart identity versions the new metric, rejecting pre-fix Eulerian
checkpoints. Existing backend identities/kinetics retain compatibility.
See [the implementation guide](EULERIAN_METRIC.md) and
[recorded validation](validation/studies/eulerian_metric/RESULTS.md).

The original path reused `matter_parameters`: unbiased b1=1, other legacy
coordinates (including log noise) zero. It supplied unit noise to the inherited
metric instead of the posterior's fixed noise and tracer response. Its precision
exceeded a noise-aware linear surrogate by 394--462 times in the earlier screen.
That was an efficiency defect, not evidence of an incorrect corrected target.

Analytic precisions, an independent zero-field likelihood Hessian, metric powers,
symmetry/positivity, cache/tape/MAP behavior and serial/slab/pencil/restart checks
pass. In a frozen four-chain field-only probe, projection bulk ESS changes from
4.5--4.9 to 171--261, with unchanged data/objective/gradient bytes and zero sampling
divergences. Corrected maximum Rhat is 1.023: this short probe does not pass every
production mixing gate or establish marginal-sigma8/full-field calibration.
No extra draws or retuning were used. Next: a declared longer joint field/scalar
pilot and numerical/scientific posterior validation.
