# Cartesian survey observations (stage 5a)

`SurveyLikelihood.h` adds a first MPI-capable survey layer to the generic
field-level framework. It composes `DistributedLinearModel` or
`DistributedLPTModel` with a masked Cartesian observation grid and either a
normalized Gaussian field likelihood or a separate Poisson count model.
Both work with the existing `BasicPosterior`, MAP and HMC interfaces. No
periodic forward-model or sampler formulas were changed.

## Geometry, layout and selection

`SurveyWindow` stores one `observed` byte (0/1) and fixed `selection` value in
[0,1] per owned compact real cell. Both vectors have `model.cells()` entries,
with local index `(local_x*N+y)*N+z`, without FFTW padding or ghost cells. The
observation mesh is the forward model's **initial FFT mesh**, not its deposition
mesh. Data and exposure arrays use this same local ordering. Nothing is gathered
into a global field for observations or likelihood evaluation.

`make_survey_window` calls a user callback at physical coordinates in Mpc/h.
`origin` is the physical position of FFT node (0,0,0), with spacing L/N; it can
label the center of voxel zero by choosing the corresponding origin. The callback
can encode a Cartesian, angular/radial or other footprint and completeness. For
example, setting origin=(-L/2,-L/2,-L/2) centers the periodic embedding on the
coordinate origin, where an observer may be placed. No distance/redshift or sky
projection is inferred automatically, and translating origin does not change
the model's plane-parallel RSD direction. The underlying dynamical volume remains
periodic. Geometry/selection/exposure are fixed during inference.

The observation synthesizes the model's tracer Fourier modes below its
`likelihood_cutoff`. It then applies a real-space window through the likelihood.
Masking therefore couples Fourier modes; multiplying by a periodic
`FourierResponse` is not a substitute for this operation. For the synthesis S,

```
S p = sum_full p_k exp(i k.x)
S^T a = N^3 r2c(a)
```

The returned mode cotangent uses the model's full-Hermitian inner product
`sum_half weight Re(conj(g_k) dp_k)`. Zero/Nyquist modes absent from the model
remain absent from the signal. The finite-box tracer mean is fixed by the
forward model; survey-mean/integral-constraint marginalization is not automatic.

`CartesianSurveyObservation` exposes `synthesize` and `adjoint` for custom
likelihoods. Hermitian inputs are checked through its own reusable FFT plan.
Constructor metadata, dimensions, finite values and geometry callback errors
are checked collectively. All ranks must make the same calls in the same order.
A rank may have no observed cells or no retained Fourier modes.

## Gaussian field data

For observed voxel i,

```
mean_i = selection_i delta_i
variance_i = exp(log_noise_power) / Vcell * relative_variance_i
U_i = 1/2 [(data_i-mean_i)^2/variance_i + log(2 pi variance_i)]
```

Here `delta_i` is the band-limited tracer prediction, `Vcell=(L/N)^3`, and
`log_noise_power` is the model's existing last scalar parameter, with noise
power in (Mpc/h)^3. `relative_variance` is a supplied positive dimensionless
array. Selection does not silently modify noise variance or become a statistical
weight: data must be in the declared selection-times-tracer units. Correlated
noise is not implemented by this diagonal voxel likelihood.

Every observed voxel contributes its noise normalization, including noise-only
signal directions outside the modeled Fourier band. This is a voxel-data model;
it is not identical to dropping unretained noisy Fourier modes in the older
periodic likelihood. Masked data and relative variances are ignored and may be
NaN; selection metadata must still be finite and valid everywhere.

```cpp
#include <FML/FieldLevel/SurveyLikelihood.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// model is a living DistributedLinearModel or DistributedLPTModel.
auto window = FF::make_survey_window(model, {-50,-50,-50}, [](auto x) {
    return FF::SurveyVoxel{x[0] + x[1] < 20, 0.8};
});
FF::SurveyGaussianLikelihood likelihood(model, std::move(window),
                                        local_data, local_relative_variance);
FF::BasicPosterior posterior(model, std::move(likelihood));
auto work = model.workspace();
auto parameters = model.initial_parameters();
auto result = FF::map(posterior, model.white_noise(12), parameters, false);
```

The likelihood returns a **local** scalar likelihood and local noise derivative;
`BasicPosterior` reduces these once and adds replicated scalar priors once.
Its FFT adjoint already includes spatial coupling across all observation slabs.

## Positive-intensity Poisson counts

A perturbative density/bias field can be negative. Counts therefore use an
explicit different observation hypothesis, a log-intensity predictor:

```
log lambda_i = log(exposure_i) + log(selection_i) + delta_i - offset
U_i = lambda_i - count_i log lambda_i + lgamma(count_i+1)
dU_i/d delta_i = lambda_i - count_i
```

`exposure` is the expected count per voxel at unit link and unit selection; it can
include a fiducial number density times cell volume and a fixed exposure map.
Selection is included **once**, outside the exponential link. `offset` is a
fixed dimensionless link offset, not an automatically computed variance or mean
renormalization. The link is evaluated at voxel nodes/centers, not integrated
with a sub-voxel quadrature. There is no clipping of negative tracer predictions
and no Jacobian for a new latent intensity variable: intensity is a deterministic
observation map of the existing field.

Counts are `std::vector<std::uint64_t>`; observed counts above 2^53-1 are rejected
to keep their conversion to double exact. Zero exposure or zero selection with
zero count contributes exactly zero. A positive count at zero fixed rate is an
input error; a masked count is ignored. The likelihood uses log rates directly,
so exponential underflow still permits finite low-rate/count contributions.
Overflow rejects the proposal collectively as `UnrepresentableState` rather
than silently capping the rate.

```cpp
FF::SurveyPoissonLikelihood likelihood(model, std::move(window),
                                       local_integer_counts, local_exposure,
                                       /* fixed offset */ 0.0);
FF::BasicPosterior posterior(model, std::move(likelihood));
auto parameters = model.initial_parameters();
auto fit = FF::map(posterior, model.white_noise(12), parameters, false);
```

Poisson counts **do not use the Gaussian log-noise scalar**. Their likelihood
contribution to its derivative is exactly zero. The existing forward-model
parameter vector retains that scalar and its independent prior (and its role in
the frozen approximate field preconditioner). The example fixes all scalars.
Joint bias/spectrum sampling is supported by the generic sampler, but the unused
noise scalar then samples its independent prior; it is not count overdispersion
or a number-density parameter. Inferring selection, exposure normalization,
overdispersion or additional likelihood nuisances requires an explicit parameter
interface extension. No scalar is silently reinterpreted as another parameter.

## Workspace, examples and validation

One observation/likelihood owns mutable FFT scratch and real cotangent storage.
It is move-only; forward models must outlive it and the posterior. Use separate
instances for concurrent chains, construct FFT plans outside concurrent regions,
and call all collective methods on every communicator rank. Real data and scratch
remain local slabs. No timing, scaling or zero-allocation claim is made; existing
forward-model workspace profiling remains open.

```sh
make -C FML/FieldLevel/example USE_MPI=true survey_mpi
mpirun -np 2 FML/FieldLevel/example/survey_mpi gaussian
mpirun -np 2 FML/FieldLevel/example/survey_mpi poisson
make -C FML/FieldLevel/tests USE_MPI=true survey-mpi
# Longer bounded nonlinear periodic validation is deliberately opt-in:
make -C FML/FieldLevel/tests USE_MPI=true framework-nonlinear-mpi
```

The example creates a spherical/radially selected toy footprint, draws voxel
Gaussian noise or Poisson counts, decreases the masked posterior with MAP, and
runs fixed-parameter field HMC. It reports MAP termination status; a decrease is
not a claim of MAP convergence or HMC convergence. Its count generator is only
intended for the small example rates. Use the library API to supply real data.

Validation includes analytic waves, independent direct DFT adjoints and voxel
likelihood formulas, noise normalization, full nonlinear ZA/2LPT field/scalar
finite differences with bias and RSD, MAP/HMC integration, empty footprints,
zero exposure, corrupted dimensions/Hermitian input and one-rank failures. See
[recorded results](tests/SURVEY_RESULTS.md).

This is an initial Cartesian survey likelihood, not a survey ingestion pipeline,
lightcone/wide-angle evolution, correlated-noise model or production survey
calibration. Pixelization/selection convergence, independent-survey mock recovery,
mean-density nuisance treatment and multiple correlated datasets remain work.
The existing exact periodic linear conditionals and the default-Gaussian
checkpoint identity are not survey-likelihood implementations; survey checkpoint
support requires its own target identity extension. Backlog #52 remains open.
