# Periodic Eulerian LPT galaxy model

`DistributedEulerianLPT.h` composes the existing unbiased, real-space
Zel'dovich/EdS-2LPT particle evolution with the [Eulerian bias
operators](EULERIAN_BIAS.md). The scalar vector contains spectrum parameters
followed by `b_delta`, `b_delta2`, `b_K2`, `b_laplacian`. It has **no noise
coordinate**. The caller supplies all four bias priors, including physical units
and bounds for the Laplacian coefficient.

```cpp
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
FF::EulerianLPTConfiguration c;
c.mesh = 16;
c.density_mesh = 32;
c.boxsize = 400;
c.initial_cutoff = .04;
c.matter_band = {.035, E::BandGeometry::Sphere};
c.likelihood_band = {.03, E::BandGeometry::Cube};
// reference_power(k) is fixed P_L(k) at the analysis redshift.
auto spectrum = FF::sigma8_spectrum(reference_power);
FF::DistributedEulerianLPTModelT<Grid> model(c, spectrum, four_bias_priors, domain);
auto work = model.workspace();
auto parameters = model.initial_parameters();
auto prediction = model.predict(local_white_field, parameters, work);
auto target = FF::fixed_noise_posterior(model, observed_modes, poisson_power,
                                      FF::InitialFieldPrior::ConditionOnField);
auto U = target.evaluate(local_white_field, parameters, work,
                         local_field_gradient, scalar_gradient);
```

`ConditionOnField` is for inference of scalars conditional on a specified white
initial realization. Its field gradient is still available for validation, but
the realization must stay fixed in the scalar analysis. For free-IC inference,
use `WhitenedGaussian` (the factory default), which adds the unit-Gaussian field
penalty and gradient. This difference is explicitly tested; it is not an
amplitude-dependent power-spectrum determinant.

`sigma8_spectrum` uses alpha=sigma8/sigma8_fid with an initial uniform prior
on 0.5..1.5, P(alpha)=alpha^2 P_fid and d ln P/d alpha=2/alpha. The input spectrum
is at the analysis epoch; the ratio is epoch-independent only when the fixed
background/growth and amplitude-only assumptions apply. Reporting an absolute
sigma8 at z=0 also requires the corresponding fiducial sigma8 at z=0. This
component does not import a cosmology or calculate that normalization.

For normalized Fourier-series coefficients, use Poisson power P=1/nbar=V/Ngal
and covariance C=P/V=1/Ngal for an unweighted catalog. This power is a constructor
input, not a fitted scalar. Do not subtract shot noise from the observed field.
The target includes both the residual term and the Gaussian log determinant,
weighted by the half-complex multiplicity. Scalar priors are added once after
MPI reduction; model pullbacks return local scalar contributions. White fields
and operator fields remain distributed. No full field or mode all-gather is used.

The initial band remains **spherical** in this first composition. Matter and
likelihood bands each support explicit spherical/cubic selection with strict
boundaries. Cubic likelihood modes are enumerated directly: corners are not lost
to a preliminary radial mask. A radial internal envelope of the matter band
includes its cubic corners before the selected Eulerian matter filter is applied.
DC and Nyquist planes are excluded. The matter/operator mesh currently equals
the white-field mesh by default. An explicit [compact/product separation](COMPACT_MESHES.md)
uses `mesh` for the sampled field and observation modes, `product_mesh` for
LPT/particle evolution and operators, and `density_mesh` for deposition. The
conservative padding guards apply to the product mesh. The likelihood cutoff must not
exceed the matter cutoff in this implementation.

The matter adapter sets unit particle weights in the existing LPT model, with
zero Lagrangian-bias coefficients and zero RSD. Its fixed legacy noise slot is
internal bookkeeping and contributes no parameter or prior to this target.
The old Lagrangian forward model is unchanged. The adapter still computes some
unused legacy bias intermediates; factoring those away remains performance work.
Compact latent storage is now separate from padded product meshes when requested.
Particle deposition defaults to the existing smooth deconvolved mesh kernel.
An explicit [PSWF NUFFT option with its position adjoint](NUFFT_MATTER.md) is now
available through `matter_assignment` and `nufft_epsilon`.

The reverse maps Fourier cotangents to product-grid cells with 1/N_product^3,
applies the Eulerian operator transpose, and maps real matter derivatives back
to the LPT cotangent with N_product^3. The compact white-field transfer has its
own Euclidean transpose. These factors follow the normalized-FFT convention and are checked by
full field/scalar directional derivatives. `validate_modes` checks Hermitian
symmetry across ranks by a distributed FFT round trip; direct `pullback` callers
must also supply Hermitian cotangents. Keep the model at a stable address while
its workspaces or posterior exist.

`mock` produces a Gaussian noisy tracer **field**, not a discrete galaxy catalog.
The small regression covers cubic corners, direct particle Fourier sums,
unbiased-LPT agreement, all five scalar derivatives, the field derivative,
normalization, fixed/free field prior policies, and stable target changes. It
runs in serial and with slabs/pencils on 1/2/4 ranks:

```sh
make -C FML/FieldLevel/tests eulerian-lpt USE_MPI=false
make -C FML/FieldLevel/tests eulerian-lpt USE_MPI=true USE_OMP=true NP_LIST='1 2 4'
```

This establishes a small compositional forward/adjoint test, not a sigma8
posterior, a numerical-convergence campaign or a reproduction of the challenge.
The exact catalog IC/spectrum inputs and challenge intermediate filters remain
open. A [fixed-IC scalar sampler](FIXED_IC.md) and [bounded three-mesh pilot](validation/studies/galaxy_fixed_ic/RESULTS.md) are now available; the original pilot failed its mixing gate and the separate
[joint-proposal follow-up](validation/studies/galaxy_fixed_ic_joint/RESULTS.md) passes
its bounded scalar checks. The general
noise policy and this forward model now have an explicit
[checkpoint/restart adapter](EULERIAN_RESTART.md), including compact meshes.
Scientific convergence, resource and mixing checks are still required before
long-running galaxy chains.

The [posterior-aware Gaussian field metric](EULERIAN_METRIC.md) now uses fixed or
explicitly indexed noise together with the Eulerian linear/Laplacian response.
Call `posterior.apply_field_metric` for direct metric operations; HMC and MAP
route through it automatically. HMC freezes the initial scalar/noise snapshot.
