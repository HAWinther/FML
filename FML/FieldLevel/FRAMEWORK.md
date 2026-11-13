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
width/cells. `Model` owns a full grid and supports one FML rank only; multi-rank
construction still rejects before allocation. For distributed linear inference,
use `DistributedLinearModel` below. Particle exchange and nonlinear communication
adjoints remain stage 4.

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
is needed for Eulerian velocity products. Run predict again before the next pullback after reconstruction. The distributed
backends explicitly invalidate their forward caches for both linear and LPT models.

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

## Distributed linear inference (stage 4a, 12 November 2026)

Include `<FML/FieldLevel/DistributedLinear.h>` and set
`Configuration::evolution = Evolution::Linear` explicitly. Construct
`DistributedLinearModel` and `DistributedLinearPosterior`; use the same `map` and
`sample` functions. [example/FrameworkMPI.cpp](example/FrameworkMPI.cpp) is a minimal
working example. This backend also runs in a serial build for comparison.

The model implements the same normalized periodic linear prediction, generic
spectrum derivatives, scalar priors, fixed Fourier response, Gaussian noise and
plane-parallel RSD as `Model`. Its `workspace`, `white_noise`, `mock`, `predict`,
`pullback`, `apply_field_metric` and `linear_conditional` operate collectively.
`white_noise` generates the local slab without communication; use the same seed
and stream on every rank.
`linear_conditional` returns the local mean/covariance and global log marginal;
the `linear_log_marginal` convenience remains serial-only. Distributed
reconstruction outputs are available in stage 4d below.

- `cells()` is the number of **owned** real cells, in x/y/z order, with no FFT
  padding. `layout()` supplies global cells and local x start/width. The latent
  dimension and whitening factor still use **global** N³. Fourier vectors contain
  only locally owned retained modes, ordered by local x, y and nonnegative kz.
  A supplied response vector uses that same local ordering. Zero and Nyquist
  modes are excluded from the modeled band; unconstrained latent modes retain
  their unit Gaussian prior.
- Posterior values and scalar gradients are reduced globally; the scalar priors
  enter once. `pullback` itself returns local field gradients and **local** scalar
  contributions, which the posterior reduces. MAP reduces field slopes and norms.
  `Execution<Forward>` supplies these hooks; serial/custom forwards keep identity
  reductions unless they opt into a distributed specialization.
- HMC keeps the field/momentum/mass operations local. It sums field kinetic energy
  globally and counts replicated scalar kinetic energy once. Counter-based momentum
  draws use global cell IDs and trajectory index, while the shared control RNG
  handles scalar momenta, leapfrog jitter and acceptance. Step/mass adaptation and
  scalar states remain replicated. The field mass is frozen at the initial scalar
  parameters. `last` contains a local field followed by replicated scalars when
  varied; recorded scalar rows and global potentials are replicated on all ranks.
- All ranks in `FML::communicator` must call collective APIs in the same order.
  Configurations, scalar parameters, seeds and inference controls must agree;
  spectrum callbacks must implement the same deterministic function on every rank.
  Config/parameter/control and local state-dimension checks reject collectively.
  Data/response Hermitian checks use a distributed FFT round trip, including pairs
  across rank boundaries. Callback exceptions/missing derivatives remain visible
  errors; unrepresentable numerical proposals return infinite posterior potential.
- Call the field sink on **every rank**. It receives a local field and replicated
  parameters. Compute any reductions before a root-only output branch. FFT
  workspaces and the model must remain alive; do not run collective chains
  concurrently on the same communicator or construct workspaces in worker threads.
- The current FFTWGrid contract requires an even mesh divisible by the communicator
  size and nonempty FFTW slabs. `density_mesh` and bias smoothing do not participate
  in linear evolution. Double-precision FFTW is the validated configuration.

Build and run (use the local MPI compiler and FFTW-MPI paths as needed):

```sh
make -C FML/FieldLevel/tests USE_MPI=true framework-mpi
make -C FML/FieldLevel/example USE_MPI=true framework_mpi
mpirun -np 4 FML/FieldLevel/example/framework_mpi
```

The regression checks 1/2/4 ranks against a direct DFT, finite differences,
adjoint products, exact MAP and exact Gaussian projection moments; a short joint
field/spectrum/bias/noise chain checks synchronized scalar states. This establishes
a linear distributed inference foundation, not nonlinear posterior calibration or
a strong-scaling result. No full fields or mode arrays are gathered by the model.
Small replicated metadata and repeated safety reductions still have overhead.

Distributed 1LPT and 2LPT are available in stages 4b–4c below. Next:
nonlinear inference validation and checkpoint/restart. Reconstruction outputs are
available in stage 4d below. Survey likelihoods remain deferred
until after the prioritized MPI work. MPI BAO compatibility and PM/COLA are not
provided by this linear backend. The original single-rank safety checks remain.

## Distributed 1LPT (stage 4b, 12 November 2026)

Include `<FML/FieldLevel/DistributedLPT.h>`, set `evolution=Evolution::Zeldovich`,
and construct `DistributedLPTModel` / `DistributedLPTPosterior`. They use the same
configuration, spectrum, prior specifications, smoothed Lagrangian bias conventions,
plane-parallel RSD and full-Hermitian mode inner product as the single-rank `Model`.
Supported scalar parameters are spectrum parameters followed by b1, b2, b_s2,
b_laplacian and log noise power. The initial cutoff must remain strictly below
half the initial-grid Nyquist; the density mesh must be even, at least the initial
mesh, and divisible by the number of ranks. Kernel widths 4/6/8 are supported.
The backend now accepts `Zeldovich` and `LPT2`; stage 4c below documents the
second-order source and adjoint extension.

```cpp
#include <FML/FieldLevel/DistributedLPT.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// After FML::FMLSetup::init, on every rank:
FF::Configuration c;
c.evolution = FF::Evolution::Zeldovich;
c.mesh = 8; c.density_mesh = 16; c.boxsize = 100;
c.initial_cutoff = .115; c.bias_cutoff = .08; c.likelihood_cutoff = .10;
FF::DistributedLPTModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
auto work = model.workspace();
auto p = model.initial_parameters();
auto data = model.mock(model.white_noise(10), p, 11, work); // Local modes
FF::DistributedLPTPosterior posterior(model, data);
auto fit = FF::map(posterior, model.white_noise(12), p, false);
```

The field/parameter ownership and collective-call rules from stage 4a still apply.
Particles stay on their Lagrangian owners. Each forward pass computes distributed
initial/bias fields, uses globally centered quadratic/tidal bias operators, and
builds sparse contributions to density nodes on any rank. Node contributions are
aggregated locally, routed to their FFT slab owners and summed there. The cached
communication layout returns mesh cotangents to the same particle owners; interpolation
and displacement/bias adjoints then reverse the forward map. Thus the method handles
arbitrary periodic particle displacements without retaining a full density grid on
every rank or assuming that particles stay within neighboring slabs.

Initial and density meshes have different Fourier ownership. Reusable
`FourierResamplingPlan` objects transfer normalized coefficients between their
layouts; the reverse uses the corresponding upsampling map with the density FFT
adjoint's N_density³ normalization. Smooth assignment and its Fourier correction
match the original serial model. Nonfinite particle proposals reject before
floating-point coordinates are converted to node indices. Routed key/count errors
and invalid local states fail collectively; reverse communication requires a
successful matching scatter.

`map`, `sample`, `mock`, `predict`, `pullback`, `white_noise` and the frozen
linear field metric are supported. There is no exact nonlinear conditional, and
reconstruction outputs are available in stage 4d below. A nonlinear
MAP is a local optimum; the short joint-HMC regression checks execution, not
posterior calibration.

```sh
make -C FML/FieldLevel/tests USE_MPI=true framework-lpt-mpi
make -C FML/FieldLevel/example USE_MPI=true framework_mpi
mpirun -np 4 FML/FieldLevel/example/framework_mpi 1lpt
```

The new sparse exchange uses checked MPI-int counts/displacements and rejects an
exchange exceeding INT_MAX entries on any rank. It is not a large-count transport
implementation. Scratch is bounded by local particles, local grids and distinct
routed stencil nodes (at most (kernel_width+1)³ nodes per local particle before
deduplication), plus incoming requests. Highly clustered particles can concentrate
requests on a rank; this is not a strict balanced-memory or strong-scaling guarantee.
Sparse maps are rebuilt each prediction. Double FFTW and sequential collective
chains are the validated configuration; no additional OpenMP speedup is claimed
for the new routing/particle loops.

Validation covers serial compatibility, actual 1/2/4-rank predictions/gradients,
large displacements, independent sparse scatter/reverse dot products, nonlinear
MAP, joint-HMC synchronization, sanitizers and MPI/OpenMP/manual initialization.
Distributed 2LPT and reconstruction outputs are available in stages 4c–4d below.
Checkpoint/restart, measured workspace/communication optimization and nonlinear
scientific validation remain open.
Survey likelihoods remain deferred until after the prioritized MPI work.

## Distributed 2LPT (stage 4c, 12 November 2026)

The same `DistributedLPTModel` / `DistributedLPTPosterior` now also accept
`Configuration::evolution = Evolution::LPT2`. Ownership, scalar parameter order,
cutoff restrictions, Gaussian likelihood, sparse particle/mesh communication and
MAP/HMC interfaces remain as in stage 4b. The example accepts `2lpt`:

```sh
make -C FML/FieldLevel/tests USE_MPI=true framework-lpt-mpi
make -C FML/FieldLevel/example USE_MPI=true framework_mpi
mpirun -np 4 FML/FieldLevel/example/framework_mpi 2lpt
```

The EdS-factorized second-order displacement uses the **unsmoothed initial field**:

```
T_ij = partial_i partial_j inverse_laplacian(delta_L)
S2 = (delta_L^2 - sum_ij T_ij^2) / 2
Psi2 = (3/7) D[S2],     D(k) = i k_box / k_box^2
x = q + Psi1 + Psi2 + f zhat (Psi1_z + 2 Psi2_z)
```

Here displacements and q are in box units, so `k_box=2 pi n`. This is the same
EdS growth factorization and fixed plane-parallel growth rate as the original
periodic model, not a new general cosmological-growth implementation. Bias smoothing
still affects only the Lagrangian bias operators, independently of the matter source.

The source generates modes beyond the input cutoff. Its derivative/inverse-Laplacian
operator therefore runs over the full distributed initial FFT grid, with the zero
mode and all Nyquist planes removed to preserve the real/Hermitian convention.
The strict initial cutoff below half Nyquist resolves the quadratic products;
high-cutoff padded adjoints remain unsupported. The reverse uses `D^T=-D` and
propagates S2 cotangents through both delta_L and all six independent tensor
components, with the off-diagonal multiplicities. The second-order RSD factor is
`1+2f` along z. Spectrum derivatives include this quadratic evolution path.

Seven additional **local** fields (initial density and six matter tensors) are
cached only for 2LPT, separately from the smoothed bias cache. The quadratic source
and its reverse also need temporary local fields. This increment does not provide
a recomputation-memory policy or a measured speed/scaling improvement.

The combined LPT suite runs both orders on 1/2/4 ranks and checks serial compatibility,
field/scalar finite differences, adjoint products, periodic crossings, MAP and short
joint-HMC execution. Independent two-wave particle positions verify the EdS coefficient,
sign and RSD factor; a single plane wave has zero second-order source. Assignment
widths 4 and 8 receive additional field-adjoint checks alongside the main width-6
cases. Fixed cutoffs and toy meshes do not establish production convergence or
nonlinear posterior calibration.

Distributed reconstruction outputs are available in stage 4d below.
Checkpoint/restart, scientific nonlinear validation and workspace/communication
profiling remain next. Survey likelihoods
are still deferred until after the prioritized MPI work; #52 is not complete.

## Distributed reconstruction outputs (stage 4d, 13 November 2026)

`DistributedLinearModel` and `DistributedLPTModel` now provide the same
`reconstruct(white,parameters,workspace,velocity_factor)` bundle as the serial
model. Each call is collective and returns **owning local arrays**:

- Initial linear density contrast on the initial mesh.
- Unbiased, real-space matter contrast on the initial mesh, restricted to the
  model's retained mode band (max of initial/likelihood cutoffs). Linear matter
  equals the initial density. LPT matter uses unit particle weights and real-space
  positions, with the same smooth assignment/deconvolution as the forward model.
- Three displacement samples at Lagrangian q, in **box units**; multiply by L for
  physical comoving lengths. They include Psi1 plus Psi2 where applicable, without
  the tracer's redshift-space displacement.
- Three velocity samples at Lagrangian q: `velocity_factor*Psi1` for linear/1LPT,
  or `velocity_factor*(Psi1+2 Psi2)` for EdS 2LPT. Supply `a H f L` for peculiar
  velocity units compatible with H and L. This conversion is explicit and separate
  from the fixed RSD growth rate configured on the forward model.
- Locally owned observed tracer modes, retaining the configured bias, RSD and
  Fourier response. A zero response suppresses these modes without suppressing
  the physical density/displacement/velocity products.

Use `model.layout()` for the owned x start/width and global mesh. All real output
vectors have `model.cells()` entries in x/y/z order; there is no FFT padding or
all-gather. Matter is an Eulerian mesh field; displacements and velocities are
Lagrangian samples, **not** volume- or mass-weighted Eulerian velocity meshes.
Bias, noise, RSD and observation response do not enter the physical outputs;
initial-spectrum parameters, the supplied velocity conversion and physical
LPT order still do.

```cpp
auto work = model.workspace();
auto products = model.reconstruct(local_white, parameters, work, a * H * f * L);
// products.initial_density, matter_density, displacement[3], velocity[3], tracer
```

All ranks must call this in the same order with the same scalar parameters and
finite velocity conversion. Per-rank conversion mismatches and nonfinite outputs
fail collectively. The method reuses workspace scratch and, for LPT, performs an
additional unbiased real-space deposition; it invalidates the previous pullback
cache. Call `predict` again before `pullback`. A new posterior evaluation already
performs that fresh prediction. Returned arrays remain owned by the caller.

For fixed-parameter sampling, `sample(...,false)` returns a local white field in
`chain.last`, so it may be passed directly to `reconstruct`. For joint sampling,
`chain.last` also contains the replicated scalar block: split its first
`model.cells()` entries from the parameter tail before reconstructing. A field sink
may reconstruct samples using a separate workspace created outside worker threads;
all ranks must participate before root-only printing or file output.

The MPI example reconstructs its final field in linear/`1lpt`/`2lpt` mode and prints
globally reduced density and velocity RMS summaries. Its velocity conversion of 1
is explicitly a numerical toy. The focused test target is:

```sh
make -C FML/FieldLevel/tests USE_MPI=true framework-reconstruction-mpi
```

The bundle allocates eight owning real fields plus local tracer modes, in addition
to the existing workspace and reconstruction temporaries. Output selection,
recomputation-memory policies and parallel scientific-file export are not added
here. The method does not run automatically inside force/posterior evaluations.
Checkpoint/restart remains the next engineering milestone; nonlinear scientific
validation remains open and survey likelihoods remain deferred.
