# General field-level forward modeling and Bayesian reconstruction (#52)

Saved 4 November 2026 at the user's request. This is the agreed design direction
and staged work plan, not a claim that the general framework is implemented.
The current BAO prototype will become its first application. Backlog #52 remains
open. The user authorized the MPI/input-safety prerequisite first, followed by stages
1–3 as the first implementation milestone.

## Initial milestone status (4 November 2026)

- Safety prerequisite completed in `9f88b6c`: early single-rank rejection, once-only
  startup and public field/filter dimension checks, with serial/MPI/sanitizer tests.
- Stages 1–3 have an initial periodic implementation in [Framework.h](Framework.h),
  documented in [FRAMEWORK.md](FRAMEWORK.md). It includes generic spectra and scalar
  priors, distinct cutoffs, linear/1LPT/2LPT with smoothed bias/RSD adjoints,
  Gaussian likelihood with noise normalization, exact linear conditionals, MAP,
  joint field/bias/noise HMC and streamed/reconstructed field outputs.
- Shared BAO kernels are retained behind a compatibility adapter; full extraction
  of independent evolution/tracer numerical components remains a refactoring option.
  High-cutoff padding, full cosmological growth derivatives, distributed execution,
  survey/count observations, PM/COLA and production nonlinear coverage are not claimed.
- Validation and sampler limitations, including rare divergent proposals with wide
  noise priors, are recorded in [tests/FRAMEWORK_RESULTS.md](tests/FRAMEWORK_RESULTS.md).

## MPI priority and stages 4a–4e (updated 13 November 2026)

At the user's request, prioritize MPI before the proposed survey likelihood.
The survey likelihood was deferred during the MPI work; its initial stage 5a
Cartesian implementation is now recorded below.
The first MPI increment is a distributed periodic linear-Gaussian model in
[DistributedLinear.h](DistributedLinear.h), using the generic posterior/MAP/HMC
with global reductions, local FFT slabs and counter-based field momenta. Exact
linear references and 1/2/4-rank checks anchor this increment. Stage 4b adds distributed 1LPT predictions,
bias/RSD adjoints, sparse particle/mesh routing, MAP and joint-HMC execution.
Stage 4c extends that backend to EdS 2LPT with its full-grid quadratic source,
reverse path and second-order RSD factor. The original single-rank implementation
remains a numerical compatibility reference. Stage 4d adds owning local
initial/matter/displacement/velocity outputs and observed tracer modes for all
three MPI models, with explicit units and cache invalidation. Stage 4e adds complete
same-rank trajectory-boundary HMC checkpoint/restart, including warmup adaptation,
RNG/distribution state, cached forces and the original frozen field metric.
This is the initial periodic MPI engineering milestone; nonlinear scientific
validation and performance/memory profiling remain open. Survey observations can
now follow as the next planned extension. See [FRAMEWORK.md](FRAMEWORK.md) for ownership, collective-call contracts and current
limits.

## Bounded nonlinear validation and stage 5a (13 November 2026)

A bounded same-model ZA/2LPT inference pass now checks two dispersed chains for
initial-field information, reconstruction error and field-summary mixing on
1/2/4 ranks. It keeps nuisance values fixed and is not production posterior
calibration. See [tests/SURVEY_RESULTS.md](tests/SURVEY_RESULTS.md).

[SurveyLikelihood.h](SurveyLikelihood.h) now composes MPI forward models with
Cartesian masked/selected voxel observations and distinct normalized Gaussian
field or positive-link Poisson count likelihoods, including their adjoints and
MAP/HMC integration. [SURVEY.md](SURVEY.md) specifies coordinates, exposure,
noise and normalization, the unused Gaussian-noise scalar for counts, ownership
and limits. Wider survey calibration, pixelization/mean-density nuisance work,
additional likelihood nuisances remain open. Same-rank survey checkpoints now
register full likelihood identities and prior families in identity schema 2;
this initial stage 5a does not complete #52.

## Bounded survey calibration (13 November 2026)

The first scientific survey pass is recorded in
[survey_validation/RESULTS.md](survey_validation/RESULTS.md): longer dispersed
full-grid production chains, twelve independent initial-field mocks per Gaussian
and Poisson likelihood using an exact active-mode reference, masked exact-Gaussian
posterior checks, future-data predictions and clustered coverage, plus separate
selection/mesh convergence. Fixed spectrum/bias/growth/noise/exposure and a small
26-mode band limit make this conditional same-model evidence. It does not close
production calibration, nuisance inference or PM/COLA evolution. The convergence
scan exposes appreciable boundary errors from coarse point-sampled selection.

## Aim and posterior

Build composable physics, observation, likelihood and inference components around
one shared latent initial field. Support both perturbative galaxy-field inference
and initial-condition reconstruction, with posterior predictions of evolved density
and velocity. Keep the framework independent of a particular BAO template or sampler.

A representative posterior is

```
p(s, theta, b, nu | d) proportional to
    exp(-s^T s / 2) L[d | R F_(theta,b)(L_theta s), nu] p(theta,b,nu)
```

Here s is the whitened initial field, theta controls the spectrum/evolution, b the
tracer model, nu the noise, and R the observation operator. F includes the chosen
evolution and tracer prescription, including advected initial bias operators when
using Lagrangian bias. Maintain explicit normalization and parameterization
Jacobians; do not count a prior determinant twice when working in whitened variables.

## Components

| Component | Responsibility | Initial implementation |
|---|---|---|
| Initial-field prior | White-to-correlated field map and parameter derivatives | General P(k;theta), replacing the hard-coded BAO spectrum |
| Evolution | Displacements, velocities and matter fields | Linear, Zel'dovich and 2LPT |
| Tracers | Biased tracer fields | Linear, quadratic, tidal and higher-derivative operators; Eulerian or advected Lagrangian bases |
| Observations | Predictions in the actual data representation | Periodic real/RSD fields first; masks, selection and survey geometry later |
| Likelihood | Data probability and derivatives | Gaussian field likelihood, followed by a separately defined galaxy-count likelihood |
| Posterior | Compose data terms, field prior and parameter priors | Multiple datasets/tracers sharing one initial field |
| Inference | Optimize or sample the posterior | MAP, then field HMC with nuisance-parameter updates |

Use a small explicit composition system initially. Do not start by implementing a
general automatic-differentiation engine. Existing numerical routines can supply
forward calculations, but their adjoints still need implementation and verification.

## Required contracts

- Every differentiable component supplies a forward operation and reverse/adjoint
  operation. Declare forward-only components explicitly; unavailable derivatives
  must not be silently approximated or treated as zero. Tangent operations can
  follow where useful for curvature/preconditioning.
- Separate initial-field cutoff, bias smoothing and likelihood cutoff. Numerical
  and physical model convergence must be testable independently. Parameter changes
  should not silently change the latent-space dimension or a hard mode selection.
- Specify units, real/Fourier layout, FFT normalization, Hermitian multiplicities,
  zero/Nyquist modes and real-space versus Fourier inner products.
- Keep model configuration immutable and mutable scratch in reusable per-chain
  workspaces. Reuse FFT plans, fields and adjoint buffers; define cache invalidation
  for cosmology and nuisance changes. Concurrent chains have independent workspaces.
- Establish distributed ownership during design: local field slabs, globally
  reduced scalar products, and explicit particle/mesh communication. Migration,
  deposition and interpolation need corresponding adjoint communication; changing
  forward loop bounds alone does not implement distributed inference.
- Distinguish perturbative Gaussian tracer fields from positive count intensities.
  A potentially negative bias expansion is not automatically a Poisson intensity.
  Specify selection, exposure and missing-region behavior in the observation model.
- Include normalization/log-determinant terms when sampling noise parameters.
  Analytical marginalization of bias coefficients must retain determinant terms;
  fitting/profile likelihood and Bayesian marginalization are distinct operations.
- Parameter priors and their transforms belong in the posterior, independently of
  the sampler. Preserve prior-whitened fields; treat partial centering as an explicit
  optional parameterization with its Jacobian.

## First useful milestone

A periodic-box reconstruction jointly inferring:

- the initial field;
- bias coefficients b1, b2, b_s2 and b_laplacian_delta;
- a positive noise amplitude;
- optionally one spectrum parameter, such as amplitude or BAO dilation.

Use 2LPT with real-space and fixed-line-of-sight RSD predictions. Return samples of
initial density, evolved density and velocity, alongside parameter samples. Recreate
the existing BAO analysis as an application through the same generic interfaces.
The milestone is stages 1–3 below; it does not require survey geometry or PM adjoints.

## Implementation order and acceptance

0. **Safety prerequisite: FL-1 and FL-2.** Reject unsupported multi-rank construction
   before serial FFT/index access, remove redundant raw MPI initialization in the
   test/examples, and validate public field/filter dimensions before access. Verify
   supported serial/one-rank operation, deliberate multi-rank rejection, negative
   input tests, valid conversion/filter round trips and existing gradients.
1. **Generic interfaces and exact linear-Gaussian problem.** Verify MAP solutions,
   posterior covariance and sampling against analytical results. Establish field
   layout, inner-product, parameter and workspace contracts here.
2. **Differentiable 1/2LPT, tracers and RSD.** Test each adjoint with dot products and
   step-converged directional finite differences. Check assignment, mesh, smoothing
   and cutoff convergence; compare predictions against independent references.
3. **Joint field/bias/noise inference.** Reproduce the BAO application; test negative
   controls, information gain, dispersed chains and repeated-mock coverage. Use the
   shared MCMC diagnostics on nuisance parameters and relevant field summaries.
4. **Complete MPI execution and adjoints.** Extend the ownership contracts established
   in stages 1–2. Verify full posterior values, directional gradients and predictions
   across rank counts, with bounded memory and checkpoint/restart behavior.
5. **Survey observations and count likelihoods.** Add selection, exposure, masks and
   survey geometry explicitly, with separate Gaussian-field and positive-intensity
   models and independent validation of observation adjoints.
6. **PM/COLA and richer physical models.** Implement their adjoints and a tested
   memory/recomputation strategy. Add higher-order LPT/bias and additional data types
   through the established interfaces rather than changing the posterior/sampler API.

The prototype scientific-validation and measured-workspace follow-ups are tracked
in [KNOWN_ISSUES.md](KNOWN_ISSUES.md); portable step-converged gradient tests are
now resolved there. The bounded checks do not complete production calibration.

## Existing starting points and references

`FieldLevelBAO.h` already contains the spectrum mapping, 1/2LPT, Lagrangian bias,
RSD, assignment/adjoint chain and an HMC adapter. Reuse verified components while
separating their roles. General LPT/growth, transfer-provider, hybrid-bias,
interpolation, survey and MCMC modules can supply interfaces or numerical building
blocks; their conventions and derivatives must be checked before integration.

- [LEFTfield forward modeling](https://arxiv.org/abs/2409.10937): perturbative galaxy-field modeling and numerical/model convergence.
- [BORG reconstruction](https://arxiv.org/abs/1203.3639): Bayesian initial-condition reconstruction with dynamical evolution and survey observations.
- [Field-level BAO inference](https://arxiv.org/abs/2407.01524): an application of joint initial-field and parameter inference.

These are scientific reference directions, not claims of native compatibility or
complete reimplementations of the named codes.

## PM forward milestone (14 November 2026)

[ParticleMesh.h](ParticleMesh.h) and [its guide](PARTICLE_MESH.md) implement the
first periodic PM milestone: fixed flat matter/Lambda background, growing ZA
initialization from the shared whitened spectrum map, CIC mesh forces and KDK
evolution with explicit canonical time/velocity units. Slab/pencil sparse
scatter/gather retains Lagrangian ownership and exposes final matter, displacement
and momentum with separate particle/mesh layouts. Independent dense-DFT and native
NBody comparisons, conservation, rank/backend replay and fixed-mesh timestep
refinement are the bounded validation goals. This is a forward API, not a PM
posterior. Next: reverse discrete map with checkpoint/recomputation, then MAP and
small posterior experiments. Existing LPT/survey inference remains unchanged.

## PM discrete adjoint milestone (14 November 2026)

[ParticleMeshAdjoint.h](ParticleMeshAdjoint.h), documented in [PM_ADJOINT.md](PM_ADJOINT.md),
now reverses CIC gather/deposition, the spectral force, KDK and the shared ZA
initial spectrum/whitening map. Real-space output cotangents produce local white
gradients and globally reduced spectrum gradients. Bounded binary midpoint
checkpoints/prefix replay expose retained-state bytes and recomputation counters.
The original public forward result is preserved and inputs are cached by value.
Directional finite-difference/transpose, component, rank/thread/backend and sanitizer
checks validate this increment. This does not add a PM likelihood or posterior.
Next: connect the PM map/pullback to a periodic likelihood and MAP, then small
HMC scientific experiments; cosmological derivatives and survey/tracer/RSD follow
as distinct extensions.

## Periodic PM likelihood and MAP milestone (14 November 2026)

[ParticleMeshPosterior.h](ParticleMeshPosterior.h), documented in [PM_POSTERIOR.md](PM_POSTERIOR.md),
adapts the PM map/pullback to the existing normalized Fourier-Gaussian likelihood,
white/scalar priors and MAP optimizer. It separates IC/particle and force/observation
meshes, supports observed nonlinear modes above the IC Nyquist, and correctly
bridges global raw PM spectrum derivatives to the generic posterior reduction.
Full posterior derivatives, independent DFT/noise normalization, volume scaling,
serial/MPI/pencil replay and a conservative-band fixed-parameter MAP mock check
validate the increment. A bounded joint fit exercises spectrum/noise updates.
An exploratory broader-band line-search limit remains documented. Next: bounded
HMC recovery, divergence/step-size and mixing tests; wider coverage and PM-specific
restart/survey/tracer/cosmological derivatives remain separate.

## Bounded periodic PM HMC validation

[The validation driver](pm_hmc_validation/README.md) compares fixed-step and
adapted HMC with refined two-dimensional quadrature for two active IC modes,
conditioning on the other white coordinates and scalar parameters. Both ensembles
pass the recorded moment/Monte Carlo error checks with no post-warmup divergences.
Energy/reversibility probes and serial/MPI/pencil/thread/sanitizer replays pass.
Short full-field and joint spectrum/noise traces exercise the public sampler and
canonical replay; they do not establish full-posterior mixing or calibration.
See [results and limitations](pm_hmc_validation/RESULTS.md). Larger fields, longer
joint runs, posterior predictive checks, repeated-mock coverage and PM-specific
restart/survey/tracer/cosmological derivatives remain separate; item 52 stays open.

## Bounded full-field PM posterior validation (15 November 2026)

[The full-field campaign](pm_full_validation/README.md) samples all 64 white
coordinates in four dispersed fixed-parameter and four joint spectrum/noise chains.
For this N=4/M=8 mock, six IC directions affect the likelihood and 58 have an exact
unit-Gaussian posterior. All declared feature-mixing, analytic null-moment,
normalization and posterior-predictive checks pass. Backend/thread/sanitizer replay
also passes. [Results and plots](pm_full_validation/RESULTS.md) separate latent
uncertainty from noisy predictions. This is bounded single-mock evidence;
larger fields/mode support, repeated-mock coverage and numerical convergence remain
open, along with PM checkpoint/restart, tracer/RSD/survey and cosmological derivatives.
Item 52 remains open.

## Same-layout PM inference checkpoint/restart (15 November 2026)

[ParticleMeshCheckpoint.h](ParticleMeshCheckpoint.h) registers explicit PM target
identity with the existing trajectory-boundary HMC checkpoint API. It includes both
IC/observation layouts, copied grid/domain geometry, background/time nodes, versioned
force/ZA/CIC/KDK/adjoint/metric conventions and replay budget. Complete sampler,
RNG/adaptation and frozen-metric state is reused. Fixed and joint fresh-process
continuation is bitwise exact on serial, 1/2/4-rank slabs and pencils (including
unequal boxes), with sanitizer and incompatible-configuration/failure checks.
Existing non-PM fingerprints/file schemas are preserved and old-serializer linear/
LPT checkpoints resume under the new adapter. [The guide](PM_CHECKPOINT.md) and
[results](tests/PM_CHECKPOINT_RESULTS.md) state the callback-versioning and same-build
contract. Changed-layout restart, larger/multi-mock calibration and PM survey/tracer/
RSD/cosmological extensions remain open; item 52 is not complete.

## PM convergence and more active modes (15 November 2026)

[The bounded resolution campaign](pm_resolution_validation/README.md) matches one
physical 18-mode Fourier field across particle/force grids and time schedules.
Time refinement and four directional derivative sweeps pass. Four joint chains
on N=6/M=12 sample all 216 white coordinates and spectrum/noise; all 653 monitored
features pass the declared mixing checks with no post-warmup divergences.

**Spatial accuracy remains unresolved:** four force/particle criteria fail, with
last-refinement field/gradient differences of 21.9%/68.5% (force mesh) and
16.8%/50.9% (particle lattice). Independent NumPy and native tiny-kick probes reproduce
strong sparse-load/force-grid dependence. [PM-S1](KNOWN_ISSUES.md) prioritizes dense
particle support and controlled force/assignment investigation before accuracy or
coverage claims. The numerical force law is unchanged. [Results and plots](pm_resolution_validation/RESULTS.md)
retain the failed criteria. Item 52 remains open.

## PM dense-particle sampling (15 November 2026)

Native N>M now supports more IC/particle cells than force cells, with initial
support below both Nyquists and compatible slab/pencil geometry. The force law,
adjoint and checkpoint codec are unchanged. Serial, 1/2/4-rank MPI, sanitizer,
normalized posterior and fresh-process restart checks pass, including old PM
checkpoint continuation. The [dense-sampling study](pm_oversampling_validation/RESULTS.md)
passes the original bounded finest-correction criteria: N=64, M=32 vs 64 changes
density/gradient by 0.735%/6.198%; particle and time refinements also pass. This
completes native oversampling support and this refinement experiment, not broad
PM calibration. The original sparse-load failures remain recorded; more modes,
nonlinear regimes, refined-data inference and repeated-mock coverage are open.
Particle/white memory grows as N^3. Item 52 remains open.

## Many-mode PM accuracy and refined-data reconstruction (15 November 2026)

[The broader PM campaign](pm_nonlinear_validation/RESULTS.md) increases initial support to
122 real directions and observations to 256 real degrees, using two seeds and
three strength/time fixtures. Particle/time finest-correction checks and
forward/adjoint/backend checks pass. Force-grid projected-gradient changes of
16.3%/75.1%/60.6% fail the unchanged 15% criterion; both stronger fixtures also
fail the noise-scaled field criterion. Finite force references still need
controlled particle sampling. Fixed-parameter MAP diagnostics now use data from
a finer discretization, with optimizer status and recovery measures explicit.
All three 12-iteration fits improve reconstruction but remain unconverged; initial
error ratios are 0.585/0.560/0.290 with correlations 0.816/0.834/0.969. The original
80-iteration attempts were interrupted for runtime; criteria are unchanged.
No force law, posterior, sampler or optimizer is changed. The original and dense
18-mode studies remain preserved. Broad PM calibration and item 52 remain open.

## Particle-sampled PM force reference (15 November 2026)

[The controlled forward study](pm_force_reference_validation/RESULTS.md) extends strong91 to
N64/96/128 on M32 and M64, with a N128/M64 timestep check. Particle corrections
N96→N128 are 0.0648%/0.0424% in observed-band density; the N128 time correction
is 0.0119%. Force M32→M64 at N128 still changes density by 3.7749% and 0.1063
noise SD per observed degree, retaining the failed 0.1-noise criterion. The density
force-grid difference is stable across particle loads. This establishes a better
sampled finite reference in the tested band, not an exact continuum solution.
A forward-only validation CLI avoids adjoint/MAP cost, preserving original full-mode
outputs bitwise; serial/MPI/sanitizer, N128 backend and launcher cleanup checks pass.
No production physics is changed. Posterior-gradient failures and unconverged MAP
remain open; next compare explicit force/assignment-window choices against this
controlled setup. Item 52 remains open.

## Explicit PM force-window convention (15 November 2026)

`PMForceWindow::CICCompensated` is an opt-in inverse squared CIC force window,
with a matching discrete adjoint and PM checkpoint law tag 2. Default `None`
retains tag-1 identity bytes and old numerical outputs/checkpoint compatibility.
[The controlled comparison](pm_window_validation/RESULTS.md) reduces the N128 M32→M64 raw density
correction from 3.775% to 1.768%; after smoothing outputs to a common observation
window, the descriptive density correction falls from 2.068% to 0.0309%.
The raw joint-gradient correction improves from 75.1% to 34.9% but still fails
15%. Dense linear-force controls, derivatives, serial/MPI/sanitizer and restart
checks pass. Sparse-load compensation is poor and remains unvalidated for use.
Refining the tiny-kick measurement resolves its amplified sparse-load truncation
error; the coarse-step failure is retained. This raw-observation study motivated the fixed response implemented below.
Broader gradient refinement and validation of actual posterior behavior remain open. MAP conditioning, broad calibration and
PM survey/tracer/RSD/cosmological work remain open; item 52 is not complete.

## Fixed PM observation convention (15 November 2026)

The [fixed-response study](pm_observation_validation/RESULTS.md) now implements the
actual signal response, transpose and checkpoint identity. With a common M32 CIC
observation window, compensated M32→M64 density and joint-gradient corrections
are 0.0309% and 0.476%, passing the original bounded criteria. The uncompensated
joint-gradient baseline still fails at 41.6%. Custom real coefficients, noise after
response, both distributed backends and legacy bitwise compatibility are tested.
This is one finite-reference realization; adequate broader gradient sampling,
posterior calibration, MAP conditioning and PM survey/tracer/RSD work remain open.
