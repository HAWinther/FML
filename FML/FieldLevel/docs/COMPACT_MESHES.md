# Compact latent and padded product meshes

`DistributedEulerianLPTModelT<Grid>` can sample its white initial field on a
smaller mesh than it uses for LPT evolution and nonlinear Eulerian operators.
Set `EulerianLPTConfiguration::product_mesh` explicitly; zero (the default)
keeps the previous shared-mesh path and does not allocate a compact transfer grid.

```cpp
FF::EulerianLPTConfiguration c;
c.mesh = 8;           // Sampled white field and retained observation modes.
c.product_mesh = 12;  // LPT evolution/particle lattice and Eulerian operator grid.
c.density_mesh = 24;  // Particle deposition/NUFFT fine grid.
c.boxsize = 100;
const double kf = 2 * M_PI / c.boxsize;
c.initial_cutoff = 2.7 * kf;
c.matter_band = {2.8 * kf, E::BandGeometry::Cube};
c.likelihood_band = {2.5 * kf, E::BandGeometry::Cube};
```

This is a small numerical example, not a science resolution recommendation.
All three meshes must fit the same slab/pencil domain. `product_mesh` must be
even and at least `mesh`; `density_mesh` must satisfy the existing matter model
constraints relative to the product mesh. Initial and likelihood cutoffs must
be below the compact Nyquist. The initial cutoff must also be below half the
**product** Nyquist, and the existing conservative quadratic matter-band padding
guard uses the product mesh. Initial filtering remains spherical. No padding
restriction is removed: its cost moves out of the sampled field coordinates.
For PSWF, sigma is `density_mesh / product_mesh` in this configuration.

## Normalization and transpose

Let Nc and Np denote the **numbers of real cells**, not the side lengths, on the
compact and product grids. For normalized Fourier-series coefficients the
padded white field is

```
a = sqrt(Nc / Np)
s_product = a * IFFT_product(copy_non_Nyquist_modes(FFT_compact(s_compact))).
```

The initial-density transfer in the matter model contains `sqrt(P * Np / V)`.
The factor a therefore gives the same physical initial coefficients as
`sqrt(P * Nc / V)` on the compact grid. It is independent of cosmological/scalar
parameters and adds no parameter-dependent Jacobian.

Under the ordinary cell dot products, the Euclidean transpose is

```
g_compact = (1/a) * IFFT_compact(copy_modes(FFT_product(g_product))).
```

Using a for both directions would give a wrong gradient. The transfer is an
isometry on the compact Fourier subspace excluding every compact Nyquist plane.
Those omitted planes are outside the resolved initial band. DC and modes outside
the initial band still have their unit-Gaussian prior. Compact storage here is a
smaller **real grid**, not a fully packed list of active initial Fourier modes.

The likelihood modes and their local indices remain on the compact mesh. The
forward restricts the product tracer's Fourier coefficients without white-noise
scaling; the reverse injects their cotangents and uses the product cell count for
the real-space operator transpose. The Gaussian mock uses the compact cell count
for its `P_noise/V` covariance. The existing periodic catalog adapter continues
to use `configuration().mesh` and `model.modes()` together.

The [posterior-aware field metric](EULERIAN_METRIC.md) acts directly on the
compact mesh using actual Gaussian noise and Eulerian linear/Laplacian response,
preserving identity on discarded compact directions. Field draws and HMC momentum keys use compact
global cell IDs, including on unequal pencils. The public field size/layout,
unit-Gaussian prior and sampler's field/scalar split all refer to the compact grid.

## Memory and cost

One field vector now contains `mesh^3` doubles instead of `product_mesh^3`.
The 8/12/24 fixture uses 512 rather than 1728 field coordinates: 3.375 times fewer,
or 70.4 percent less storage **per sampler field vector**. Product and particle
resolutions are held fixed in that comparison.

The large LPT/operator/deposition workspaces do not shrink automatically. The
compact path adds one compact FFT grid, a reusable product-sized lifted white
field, transfer plans and compact metadata. It also adds FFT/transfer work.
These figures are dimension/storage accounting, not total RSS, speed or mixing
measurements. Choose product/deposition resolutions from numerical convergence;
do not lower them to make a memory comparison look favorable. Model/workspace
objects must keep their existing stable-address and collective-call contracts.

## Validation and remaining work

[Recorded results](validation/studies/compact_meshes/RESULTS.md) compare to a
padded reference sampled independently from analytic Fourier waves. They check
physical predictions, all bias columns, conditional target/scalar gradients,
compact field finite differences, prior and metric, direct-sum mock noise,
workspace moves, catalog mode indices and short HMC trajectories. Serial and
slab/pencil 1/2/4-rank tests cover both smooth and PSWF assignment, all small
pencil orientations and unequal domains. Old shared-mesh regressions remain.

```sh
make -C FML/FieldLevel/tests compact-eulerian-lpt USE_MPI=false
make -C FML/FieldLevel/tests compact-eulerian-lpt USE_MPI=true NP_LIST='1 2 4'
```

This does not add a packed Fourier sampler, independent particle/evolution/operator
resolutions, the existing Cartesian survey-likelihood adapter for this model,
initial cubic filtering or a new posterior campaign.
[Checkpoint/restart support](EULERIAN_RESTART.md) is now available for the periodic
Eulerian HMC target. The
periodic Gaussian Fourier likelihood remains the supported project target.
Challenge inputs and real-data/free-IC scientific validation remain open.
