# Periodic PM Gaussian posterior and MAP (#52)

`ParticleMeshPosterior.h` supplies `FRAMEWORK::PeriodicPMModelT<Grid>` and
`PeriodicPMPosteriorT<Grid>` (slab defaults `PeriodicPMModel`, `PeriodicPMPosterior`).
The model adapts the [PM forward/adjoint](PM_ADJOINT.md) to the existing normalized
Fourier-Gaussian likelihood, white/scalar priors and `map()` optimizer. It predicts
periodic real-space matter, with a fixed background/schedule and no tracer bias/RSD
or survey observation operator. [Bounded HMC validation](validation/studies/pm_hmc_validation/README.md)
now compares a two-mode conditional posterior with quadrature and exercises the
public full-field and joint-parameter sampler.

The [resolution campaign](validation/studies/pm_resolution_validation/RESULTS.md) records a spatial
force/particle accuracy limitation ([PM-S1](KNOWN_ISSUES.md)). A
[dense-sampling follow-up](validation/studies/pm_oversampling_validation/RESULTS.md) now passes bounded
refinement criteria with N>M; broader calibration remains open. Same-model inference
checks validate the discrete target; broader physical accuracy remains unresolved.

## Use

```cpp
#include <FML/FieldLevel/ParticleMeshPosterior.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// c describes IC mesh N and force/observation mesh M=density_mesh.
// c.evolution=Linear describes the IC generator, not the PM dynamics.
FF::PeriodicPMModel model(c, spectrum, pm_settings, noise_parameter, {4});
auto work = model.workspace();
auto p = model.initial_parameters();
auto data = model.mock(model.white_noise(91), p, 92, work);
FF::PeriodicPMPosterior posterior(model, data);
auto fit = FF::map(posterior, initial_white, p, false, 200, 2e-5);
// false holds ALL spectrum/noise parameters fixed; true fits them jointly.
model.predict(fit.white, fit.parameters, work);
const auto & fields = model.reconstruction(work); // Workspace-owned PM Result.
```

The [standalone MAP example (archived)](ARCHIVE.md) is a small dimensionless
mock-recovery problem on 4^3 particles and 8^3 force cells:

```sh
make -C FML/FieldLevel/example particle_mesh_map USE_MPI=true USE_PENCIL_GRID=true
FML_PENCILS="2 2" mpirun -np 4 FML/FieldLevel/example/particle_mesh_map
```

Clean before changing MPI/backend flags. All methods and example builds retain
slab/pencil support and have no GSL dependency. FML/MPI must already be initialized;
all ranks participate in the same order.

## Likelihood and prior convention

Predictions are normalized Fourier coefficients of the final CIC matter contrast
on the force mesh, restricted to nonzero modes `k < likelihood_cutoff`. The initial
white field and power spectrum use `mesh`; the likelihood and noise use
`density_mesh`. `initial_cutoff` remains below the IC Nyquist. The observation
cutoff may exceed the IC Nyquist but must be below the force-grid Nyquist; the
PM-generated nonlinear modes then contribute. By default no assignment deconvolution
or extra observational filter is applied; optional fixed responses are described below.

Parameters are the spectrum's existing `Parameter` specifications followed by
`log_noise_power`. The default noise prior is Gaussian with initial/mean `log(1000)`,
sigma 2 and bounds [-30,40]; callers should supply a scientifically appropriate
prior. Gaussian and bounded-uniform parameter priors follow the existing framework.
The initial spectrum is defined at a=1 and retains physical power units.

Let `P_noise = exp(log_noise_power)` and `V = boxsize^3`. With full-Hermitian
multiplicities w, the parameter-dependent negative log likelihood is

```text
U_like = 1/2 sum_kept_modes w(k) [ |prediction(k)-data(k)|^2 / (P_noise/V)
                                 + log(P_noise/V) ]
U = U_like + 1/2 sum_all_initial_white_cells white(q)^2 + sum_scalar_priors penalty(p)
```

Parameter-independent constants are omitted, matching the existing likelihood.
Noise is homogeneous Gaussian Fourier noise, not a positive count/intensity model.
The mock generator uses an independent-purpose, global force-cell-ID counter stream
and FFT normalization corresponding to variance `P_noise/V` per Fourier mode. Seeds
are stable across rank/backend changes, within the documented numerical tolerance.

Data must be local retained modes with the model's exact ordering and Hermitian
symmetry; construction collectively validates shape, finiteness and symmetry.
`predict()` supplies a matching forward cache for `pullback()`; malformed dimensions
remain programming errors, while unsupported/numerically unrepresentable proposals
are rejected by the existing posterior path.

## Two layouts and gradient reduction

- `layout()` and `cells()` describe the initial white/particle mesh N.
- `observation_layout()` describes compact real force cells on M.
- `modes()` and their Fourier indices belong to M, never to N.
- `reconstruction(work)` returns the native PM `Result` with both layouts,
  initial/matter density and particle displacement/canonical momentum.

The Fourier likelihood covector becomes a real-density covector through the inverse
transform divided by M^3; full-Hermitian multiplicities are already represented
by the inverse. The raw PM adjoint returns global spectrum derivatives. This
adapter supplies those on rank zero before `BasicPosterior` performs its existing
single parameter reduction. The likelihood's noise derivatives remain local until
that reduction, and scalar priors are added once afterward. Posterior scalar
outputs are replicated global gradients; callers must not reduce them again.

The white-field metric is explicitly identity for every exponent. This is a valid
white-coordinate metric, not a PM Fisher preconditioner. MAP uses the existing
backtracking gradient-descent implementation and its explicit status, iteration
count and gradient norm. A useful reconstruction with lower objective is not by
itself a converged MAP; inspect `MAPStatus`.

## Limits and next step

CIC is piecewise differentiable. Optimization can be sensitive to assignment faces,
mode support, conditioning and the mock/noise realization. The validation includes
a converged conservative-band fixed-parameter recovery and a bounded joint fit;
it does not establish convergence of arbitrary nonlinear objectives. An earlier
broader-band trial stopped at a line-search limit and is retained in the review
artifacts. See [the numerical results](validation/tests/PM_POSTERIOR_RESULTS.md).

The generic Cartesian survey adapters assume one mesh for initial and observed
fields. This model explicitly disables that factory so N/M cannot be misindexed.
PM survey likelihoods need their own observation adapter and validation. Generic
restart support is explicit in [ParticleMeshCheckpoint.h](../ParticleMeshCheckpoint.h),
which fingerprints both layouts and the PM background/time/numerical settings.
See [same-layout PM checkpoint/restart](PM_CHECKPOINT.md); no generic `response()`
accessor or Cartesian survey factory is enabled.

[Conditional HMC results](validation/studies/pm_hmc_validation/RESULTS.md) include moment comparisons,
step-size/reversibility probes and dispersed-chain diagnostics. Short full-field
and joint-parameter traces establish integration and replay. The subsequent
[full-field campaign](validation/studies/pm_full_validation/RESULTS.md) checks dispersed fixed/joint
chains, exact-prior null moments and posterior predictions for one small mock.
Broader scientific calibration, smooth assignment alternatives,
preconditioning, changed-layout PM restart, tracer/RSD/survey models,
cosmological derivatives, 2LPT/COLA starts and production performance remain open.

## Force compensation and observations

`PMForceWindow::CICCompensated` selects opt-in dynamics independently of the
observation convention. The final optional constructor argument, after `domain`,
is `PMObservationSettings`. Its `reference_cic_mesh` applies `W_reference/W_force`
to final retained modes, keeping a common coarser CIC window as the force mesh
changes. Zero selects the legacy raw observation. Positive references must be
even, at least four, no larger than the force mesh, and support the strict
observed band below their Nyquist. This changes a window, not spatial regridding,
mode ownership or the two-layout contract.

`response.multiplier` accepts fixed finite real coefficients in local mode order;
negative/zero values are allowed and compose with the reference window. If any
rank supplies custom values, every rank supplies its owned-mode count (including
empty vectors for zero-mode owners). Coefficients are copied, cached and checked
for cross-rank Hermitian symmetry. They are fixed, not inferred parameters.
Prediction applies the response to the signal; pullback applies its real transpose
before the existing density adjoint. Noise is added **after** the response with
unchanged P_noise/V normalization. Supply data in this declared observation space.
Raw `reconstruction(work)` fields are unchanged. No generic `response()` accessor
or Cartesian survey adapter is enabled.

[The actual common-observation likelihood study](validation/studies/pm_observation_validation/RESULTS.md)
reduces the compensated M32→M64 joint-gradient correction from the raw-window
34.9% to 0.476%, with density correction 0.0309%. The uncompensated baseline still
fails the gradient criterion. This bounded result does not establish posterior
coverage, broad physical accuracy or safe sparse-load compensation.
