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
