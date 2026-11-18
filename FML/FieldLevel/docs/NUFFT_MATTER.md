# PSWF NUFFT matter deposition and position adjoint

The distributed Zel'dovich/EdS-2LPT and composed Eulerian LPT models support
PSWF NUFFT particle deposition. The existing smooth assignment remains the default.
Select the new option explicitly on either `Configuration` or `EulerianLPTConfiguration`:

```cpp
c.matter_assignment = FML::FIELDLEVEL::FRAMEWORK::MatterAssignment::PSWF;
c.nufft_epsilon = 1e-8;
c.density_mesh = 2 * c.mesh;
```

The fine/evolution mesh ratio sets the PSWF upsampling factor; it must be in
[1.25,8]. For the Eulerian model with [compact meshes](COMPACT_MESHES.md), this is
`density_mesh/product_mesh`; otherwise it is `density_mesh/mesh`.
Existing geometry and band constraints still apply. Fine meshes need
not be integer multiples of the initial mesh, but both must fit the same domain.
The tolerance must be finite, below one and at least 32 times the grid scalar's
machine epsilon, and its requested kernel width must not exceed 16. Unsupported
controls fail collectively before the kernel's assertion-based constructor.
`assignment_width()` reports the selected PSWF width; `kernel_width` controls only
the smooth option. Float builds require an explicitly looser epsilon than the
double-precision default. Runtime validation here uses double precision.

This option belongs to `DistributedLPTModelT<Grid>` and
`DistributedEulerianLPTModelT<Grid>`, for FFTWGrid slabs and PencilGrid pencils.
The compatibility `Model` and distributed linear/PM initial-condition backend
reject a PSWF selection. It does not change the PM force assignment.

## Numerical operation and reverse

Particles remain on their Lagrangian owners, including when displaced across
several ranks or periodic boundaries. The existing sparse node scatter sends
weighted kernel contributions to fine-grid owners and records their request
order. Its transpose returns each needed node cotangent to the originating rank.
No full particle list or full field is gathered. This reuses the existing sparse
communication, including its checked MPI int count/displacement limits; it is
not a new production-scale communication implementation.

The forward uses the same fitted `PSWFKernel::weights` as the NUFFT module,
followed by a normalized fine-grid FFT, normalized Fourier transfer and division
by the product of the one-dimensional kernel Fourier transforms. For unbiased
matter the result approximates

```
delta_n = sum_p exp[-2 pi i n.x_p] / N_particles,  n != 0.
```

Positions in this expression are fractions of the periodic box. With Lagrangian
tracer weights, the numerator includes those weights but the denominator remains
N_particles, as in the existing LPT tracer convention; it is not the observed
catalog's sum-of-weights normalization. Existing response factors, DC/mode masks
and Nyquist exclusions retain their roles.

The reverse applies the transpose of each discrete operation. The new
`PSWFKernel::weights_and_derivatives` differentiates the fitted Chebyshev recurrence
used for the actual weights, with a differentiated Legendre fallback. Fine-cell
position derivatives are multiplied by `density_mesh` to obtain derivatives
with respect to fractional-box displacements. Weight derivatives and position
derivatives flow through the existing Lagrangian bias, RSD and 1LPT/2LPT reverse.
Fourier cotangents use the existing half-complex multiplicities and normalized
FFT convention. Derivative weights are computed only on the reverse pass.
The kernel and Fourier factors are constructed once per model; no additional
field-sized PSWF grid is allocated beyond the existing LPT workspace. A wider
PSWF support increases spreading work and can enlarge sparse request buffers;
this change has no production memory or speed benchmark.

Compact PSWF truncation introduces small jumps when a particle's stencil changes.
The reverse is the derivative of the fitted discrete operation **within a fixed
stencil**, not a derivative at a jump. Kernel tolerance controls the approximation
but is not a universal bound on force/position-gradient error. The individual
particle test measures both mode and derivative errors independently. Tighten
epsilon and test posterior/gradient stability for the intended cutoffs before
long HMC runs. The globally smooth legacy assignment remains available.

Checkpoint identities for supported distributed LPT targets distinguish PSWF
from smooth assignment and include its epsilon. Existing smooth identities retain
their bytes. A separate [Eulerian checkpoint adapter](EULERIAN_RESTART.md) now supports the
composed model, including compact meshes.

## Verification and scope

[Recorded results](validation/studies/nufft_matter/RESULTS.md) include independent
particle Fourier sums/position derivatives, complete Eulerian field and scalar
finite differences, both evolution orders, Lagrangian bias/RSD, rank-crossing
adjoints, all small pencil orientations and unequal domains. Short generic
MAP/HMC and smooth/checkpoint/catalog regressions also run. These are bounded
numerical checks, not new posterior calibration or a production performance claim.

```sh
make -C FML/FieldLevel/tests eulerian-lpt USE_MPI=false
make -C FML/FieldLevel/tests eulerian-lpt USE_MPI=true NP_LIST='1 2 4'
make -C FML/FieldLevel/tests framework-nufft-grid-mpi USE_MPI=true NP_LIST='1 2 4'
```

Use separate or clean builds when changing MPI/grid/precision flags.
[Compact latent/product meshes](COMPACT_MESHES.md) are now available. Exact
challenge inputs, initial cubic filtering and real-data/free-IC science validation
remain open.
[Checkpoint/restart support](EULERIAN_RESTART.md) is now available.
