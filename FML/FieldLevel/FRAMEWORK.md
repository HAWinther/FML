# Periodic field-level framework: initial stages 1–3

`Framework.h` provides a first periodic-box implementation of the
[saved #52 plan](FRAMEWORK_PLAN.md): a generic power-spectrum input, separate
forward model / Fourier response / Gaussian likelihood / posterior, exact linear
conditionals, MAP and joint field/nuisance HMC. The BAO prototype remains available.
This is not the complete survey/PM reconstruction scope of backlog #52. The
validated configuration uses double-precision FFTW; other FFT precisions are not
in this inference validation matrix.

## A small reconstruction

```cpp
#include <FML/FieldLevel/Framework.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
FF::Configuration config;
config.mesh = 8;
config.density_mesh = 16;
config.boxsize = 100;                  // Mpc/h
config.initial_cutoff = .115;          // h/Mpc
config.bias_cutoff = .08;
config.likelihood_cutoff = .10;
config.evolution = FF::Evolution::LPT2;
config.growth_rate = .6;               // Fixed plane-parallel RSD parameter.
FF::Model model(config, FF::Spectrum::bao(
    FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200)));
auto work = model.workspace();
auto parameters = model.initial_parameters();
auto truth = model.white_noise(100);
auto data = model.mock(truth, parameters, 200, work);
FF::Posterior posterior(model, data);
auto fit = FF::map(posterior, truth, parameters, false); // Field MAP at fixed parameters.
FML::MCMC::HMCOptions options;
options.warmup = 400;
options.samples = 1000;
options.target_acceptance = .95;
auto chain = FF::sample(posterior, fit.white, parameters, options, true,
    [](const auto & white, const auto & params, std::size_t draw) {
        // Optional streamed field samples: save summaries or selected snapshots here.
    });
```

The example is a deliberately small mock calculation, not a cosmological
calibration or proof that a short chain has converged. Inspect MAP status,
divergences, dispersed-chain diagnostics and posterior predictive behavior.

## Parameters and probability model

The sampled field is a real cell-wise white Gaussian vector s, with prior
`-log p(s)=sum s^2/2`. Its full grid, including modes absent from the likelihood,
remains part of the state. Fixed cutoffs determine which modes affect predictions;
unconstrained latent modes retain their prior. The real-field FFT convention is
`r2c/N^3`, so correlated initial modes are

```
delta_L(k) = sqrt[P(k;theta) N^3 / V] s(k), 0 < k < initial_cutoff
```

P uses (Mpc/h)^3 and k uses h/Mpc. Initial DC and excluded modes are zero. There is
no extra parameter-dependent determinant from changing P in this whitened
parameterization. `Spectrum` supplies scalar parameter specifications and an
`evaluate(k,theta)` callback returning positive P and one derivative of ln P per
parameter. Derivatives are explicit; missing or malformed derivatives are rejected.
`Spectrum::fixed` and `Spectrum::bao` are convenience factories. A transfer-provider
power callback can supply a fixed cosmology; parameterized cosmology also needs
consistent growth/evolution derivatives, which are not supplied automatically.

The scalar state is ordered as:

1. spectrum parameters;
2. b1 for linear evolution, or b1, b2, b_s2 and b_laplacian for LPT;
3. log_noise_power.

Each scalar has a named Gaussian or bounded-uniform prior, an initial value and
optional hard bounds. Gaussian is the default; uniform priors require finite bounds.
Constructor prior overrides replace entries by name; unknown/duplicate names and
invalid prior specifications fail. For example:

```cpp
FF::Model model(config, spectrum, response,
    {{"b1", 1.7, 1.6, .2, 0., 5.}, // initial, mean, sigma, lower, upper
     {"alpha_BAO", 1., 1., .1, .8, 1.2, FF::Prior::Uniform}});
```

The BAO factory defaults to a Gaussian alpha prior; the uniform override above
reproduces the original BAO application's prior. Uniform priors use zero potential
inside their support; their finite width supplies the optimizer/sampler scale.

The prior on log_noise_power is a prior **on the logarithm**, not a linear-power
prior with an omitted Jacobian. Noise is positive through P_eps=exp(log_noise_power).
Changing to a prior on P_eps would require the appropriate transformation.

`GaussianFieldLikelihood` evaluates

```
U_data = 1/2 sum_full [ |prediction - data|^2 / (P_eps/V) + log(P_eps/V) ]
```

The normalization term is retained when sampling noise. Half-complex multiplicities
are explicit, including conjugate pairs on the kz=0 plane; data and fixed response
must preserve the required Hermitian symmetry. Parameter-independent constants are
omitted. Noise is added after the observation response. This is a Gaussian tracer
field model, not a Poisson-count model; biased fields/particle weights may be negative.

## Forward physics and observation

`Evolution::Linear` predicts `(b1 + f mu^2) delta_L`. `Zeldovich` and `LPT2` use
advected Lagrangian weights and smooth mass assignment with its explicit Fourier
correction. The initial reference implementation shares verified numerical kernels
with `FieldLevelBAO`; it does not rebuild a BAO model per force call or use a BAO
spectrum internally for the generic initial field. An identity-transfer compatibility
backend accepts the separately correlated delta_L.

The LPT weight is

```
1 + (b1-1) delta_b
  + b2 (delta_b^2 - mean delta_b^2)
  + b_s2 (s_ij s_ij - mean s_ij s_ij)
  + b_laplacian laplacian(delta_b)
```

Here delta_b uses the fixed bias smoothing cutoff, not necessarily the initial
cutoff. b2 has no extra factor of 1/2. The shared legacy kernel uses
G2=t_ij t_ij-delta_b^2, so the adapter explicitly maps
`b2_legacy=b2+2 b_s2/3`, `bG2_legacy=b_s2`, including the corresponding adjoint.
b_laplacian has units (Mpc/h)^2; its Laplacian uses physical k, not inverse-box k.
All bias derivatives and smoothing/mean-subtraction terms are included in the
reverse operation. The matter evolution uses the unsmoothed initial field.

Plane-parallel RSD acts along z, with `f(Psi1_z+2 Psi2_z)` for 2LPT. The 2LPT growth
factorization is EdS-like at the supplied linear epoch. f is a fixed configuration
parameter in this milestone; it is not jointly sampled. GrowthHistory, higher-order
LPT, time-dependent cosmological parameters and their derivatives remain extensions.

The three cutoffs are distinct: initial field, bias smoothing and likelihood.
Meshes and hard selections remain fixed during inference. For this first backend,
meshes must be even and `initial_cutoff < half the initial-grid Nyquist` for LPT.
This resolves quadratic source/bias products without silently enabling an
unimplemented padded adjoint. Higher-cutoff padded operators require a separate
verified extension. Particle sampling, assignment support and output mesh still
need convergence checks. `density_mesh >= mesh` and likelihood modes must lie
below the supported Nyquist band.

`FourierResponse` is a separate fixed real multiplier per retained mode. Empty means
identity. It can represent a declared periodic transfer/smoothing response and has
its own adjoint. It is not a survey mask, a completeness probability or an implicit
noise correction. General survey/count observations belong to stage 5.

## Composition and workspaces

`Model` is the built-in periodic forward implementation. `BasicPosterior<Forward,
Likelihood>` composes a forward implementation and likelihood without depending on
the internal LPT algorithm; `Posterior` is its default alias. MAP and HMC are generic
over that posterior. A replacement forward must provide:

- cells, layout, modes, volume, parameter specifications and likelihood configuration;
- workspace creation, support checking, prediction and a matching pullback;
- a positive field-metric operation for optimization/HMC preconditioning.

A replacement likelihood supplies `evaluate(prediction,last_scalar,cotangent,
last_scalar_gradient)`. The initial interface has one scalar likelihood nuisance
parameter; more general noise/count models need an explicit extension rather than
silently interpreting several scalar parameters as one.

Configuration and parameter metadata are separate from mutable per-chain workspaces.
`workspace()` makes a workspace owned by that exact Model instance. Models must
outlive posteriors and workspaces; rvalue-model posterior construction is rejected.
Pullbacks require the matching successful forward evaluation and parameter state.
Reacquire returned prediction views after another prediction. Construct FFT
workspaces outside concurrent regions; no concurrent-chain thread-safety claim is
made in this initial release. The shared kernel still allocates some intermediates;
FL-6's profiling/reuse work remains open. No allocation/time speedup is claimed.

Current layout metadata explicitly identifies global mesh/cells and owned start,
width/cells. This implementation owns a full grid and supports one FML rank only;
MPI-enabled one-rank hosts work, and multi-rank use rejects before allocation.
Distributed scalar reductions, HMC state, particle exchange and communication
adjoints remain stage 4. A local field loop alone is not distributed inference.

## Exact linear inference, MAP and HMC

At fixed scalar parameters `linear_conditional` returns the exact real-space mean,
Fourier covariance multipliers of the whitened field, and the exact field-marginalized
log likelihood. Inactive modes have unit prior covariance. `linear_log_marginal`
evaluates the marginal without constructing the mean, useful for low-dimensional
reference quadrature. These methods require linear evolution.

`map` uses a Fourier preconditioner and backtracking line search. It reports
Converged, MaximumIterations or LineSearchFailed with objective, gradient norm and
iteration count. `vary_parameters=false` optimizes just the field; true includes
all scalar priors and parameter support. Nonlinear MAP is local optimization, not
a global optimum or a posterior uncertainty estimate.

`sample` uses the shared HMC engine. Its field mass is a Fourier-diagonal linear
approximation **frozen at the initial scalar parameters**, avoiding an unaccounted
state-dependent mass. Scalar masses/step size use the existing warmup adaptation.
LPT's metric is an approximation; Metropolis correction targets the full posterior.
Recorded rows contain scalar parameters; `last` contains the final white field and,
when varied, the scalar block. A field callback can stream samples or field summaries
without retaining every full field in memory. When scalars are held fixed, recorded
scalar rows are constant and must not be treated as informative chain diagnostics.
Monitor reconstructed-field summaries through the callback instead.

A usable sampler is not established by acceptance alone. Use dispersed chains,
rank-normalized split/folded Rhat and bulk/tail ESS on parameters and field summaries,
plus divergences, posterior predictive checks and repeated-mock/model-mismatch tests.
The tests compare joint linear nuisance means/widths to independent quadrature and
use measured Monte Carlo uncertainty. The deliberately wide noise-prior case still
has rare divergent proposals; these are counted and reported, not described as zero.
The regression requires a low rate plus usable Rhat/ESS and correct moments; that
is not a production acceptance criterion. Investigate divergences and posterior
tails with a suitable parameterization and tuning. Short nonlinear examples are smoke checks.

## Posterior field outputs

`reconstruct(white,parameters,workspace,velocity_factor)` returns:

- initial linear density contrast on the initial mesh;
- real-space matter density contrast restricted to the declared model band;
- Lagrangian displacement samples at q, in box lengths;
- Lagrangian velocity samples with the supplied conversion;
- the observed biased/RSD tracer modes.

For EdS-factorized 2LPT, velocity is `velocity_factor*(Psi1+2 Psi2)`.
Passing `a H f boxsize` gives peculiar velocities for compatible H and length units.
No cosmology is inferred from a default numerical factor. Matter outputs are
independent of tracer bias and are evaluated in real space; tracer modes retain
the configured RSD/response. Velocity samples are at Lagrangian positions, not a
volume-weighted Eulerian velocity mesh. A separate deposition/normalization choice
is needed for Eulerian velocity products. Reconstruction invalidates the LPT
forward cache, so run predict again before its next pullback.

## Build and validation

```
make -C FML/FieldLevel/tests framework
make -C FML/FieldLevel/tests USE_MPI=true framework  # one-rank tests, 2/4-rank rejection
make -C FML/FieldLevel/example framework
./FML/FieldLevel/example/framework 2lpt 200 output_prefix
```

The example writes scalar samples, optionally streamed white-field snapshots, and a
final initial/matter/velocity reconstruction. It is an analytic periodic toy.
See [framework validation results](tests/FRAMEWORK_RESULTS.md). Tests cover field
and scalar adjoints for linear/1LPT/2LPT with RSD, distinct cutoffs, higher-derivative
bias and BAO shape; exact linear MAP/covariance; normalized sampled noise; no-data
information limits; nuisance inference against exact-marginal quadrature; BAO
compatibility; reconstruction semantics; malformed dimensions and unsupported MPI.

Stages 4–6 remain open: distributed inference, survey observations and count
likelihoods, PM/COLA adjoints and broader physical models. Production nonlinear
coverage/calibration, full padding at higher cutoffs, time/growth derivatives,
checkpoint/resume of inference and measured workspace optimization also remain.
The initial stages 1–3 milestone is a reusable periodic implementation with exact
linear verification, not a complete LEFTfield/BORG replacement.
