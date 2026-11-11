# General non-uniform FFT module

This module is being expanded into a general CPU NUFFT library, independent of
`SimulationCode`, the Einstein--Boltzmann solver, `FFTWGrid`, and the particle
classes. The existing `NUFFT.h` particle/grid routines remain as FML adapters
and keep their current names and normalization. The general transform core will
be implemented separately and use FFTW as its regular-grid FFT backend.

## Transform conventions

The public coordinates use radians and a period of `2*pi` for types 1 and 2.
Coordinates outside the principal interval are wrapped periodically. For a
dimension of `N` modes, the canonical signed indices are
`-N/2, ..., N/2-1` when `N` is even, and `-(N-1)/2, ..., (N-1)/2` when `N`
is odd. Each transform carries an explicit exponent sign (`-1` or `+1`). The
default normalization is the unnormalized finite sum below; type 1 is the
adjoint of type 2 and is not an inverse transform.

For source points `x[j]` and complex strengths `c[j]`, type 1 computes

```text
F[k] = sum_j c[j] exp(sign * i * k dot x[j])
```

Type 2 evaluates a supplied Fourier series at points:

```text
f[j] = sum_k F[k] exp(sign * i * k dot x[j])
```

Type 3 has non-periodic source points `x[j]` and arbitrary target frequencies
`s[k]`:

```text
f[k] = sum_j c[j] exp(sign * i * s[k] dot x[j])
```

`ModeArray<N, Real>` will own contiguous complex coefficients and their mode
shape. Its indexing helpers will map signed integer modes to storage indices;
callers will not need to infer Nyquist or odd-size offsets themselves. The
generic core will support `N = 1, 2, 3` and `Real = float, double`.

## Target C++ interface

The one-shot functions make each transform self-contained:

```cpp
using namespace FML::NUFFT::Core;

ModeArray<3, double> modes = type1<3>(points, strengths, {64, 64, 64}, options);
std::vector<std::complex<double>> values = type2<3>(modes, targets, options);
std::vector<std::complex<double>> irregular = type3<3>(sources, strengths, frequencies, options);
```

`Options` will initially contain the requested relative tolerance, exponent
sign, and upsampling factor. Defaults will be suitable for common double and
single precision use. Unsupported tolerances or invalid dimensions will return
an explicit error (or throw `std::invalid_argument` consistently with the FML
header-only API); they will not silently clamp and claim the requested error.

For repeated transforms, a plan will retain mode geometry and point-dependent
preprocessing:

```cpp
Plan<3, TransformType::Type1, double> plan({64, 64, 64}, options);
plan.set_points(points);
plan.execute(strengths, modes);
plan.execute_many(strength_batches, mode_batches); // same points, new strengths
```

Point ownership and invalidation will be explicit: plans own their point data;
calling `set_points` replaces the cached geometry, while changing only strengths
reuses it. Plans will be movable, non-copyable owners of FFT resources.

## Implementation sequence

1. Freeze the conventions and storage contract above, then implement serial
   complex type 1 and 2 transforms with an exponential-of-semicircle spreading
   kernel and FFTW complex-to-complex transforms.
2. Add reusable plans, batched strengths, `float` support, and tuning options.
3. Implement type 3 with its own coordinate/frequency scaling and accuracy
   checks.
4. Add optional MPI-distributed transforms with a documented output policy:
   replicated coefficients for convenience or explicit distributed mode storage
   for large outputs. The serial core will not depend on MPI.
5. Reimplement the existing real particle/grid operations as adapters over the
   general core where their conventions match, retaining their current public
   API and validating normalization compatibility.

The scope is a forward NUFFT library. Iterative inverse NUFFT solvers, GPU
backends, spherical transforms, and solver-specific cosmology logic are outside
the initial module.

## Current implementation

`NUFFT.h` currently supplies the FML particle-to-Fourier-grid, Fourier-grid-to-
particle, and particle-force operations. It is tied to `FFTWGrid` and FML's
particle/MPI helpers, accepts real strengths, and does not expose generic
full-complex type 1/2 or type 3 transforms. Those routines will remain the
compatibility layer while the independent core is developed.

For repeated particle transforms, `NUFFT.h` also provides a reusable spatial
ordering. It caches positions in spatial order and maps them back to original
particle indices, so the particle array itself is left in place and type 2/force
results are written in the caller's original order:

```cpp
auto order = FML::NUFFT::make_particle_ordering<3>(particles, num_particles);
FML::NUFFT::nufft_type1<3>(particles, num_particles, weights, norm, modes, kernel, &order);
FML::NUFFT::nufft_type2<3>(modes, particles, num_particles, values, kernel, &order);
```

The default ordering groups points into 32 blocks per dimension. Rebuild it
after changing particle positions or the particle array order. Type 1 gathers
new strengths into the cached order for each call. The ordering is most useful
when reused across several transforms; building it has a cost.

## ES and PSWF spreading kernels

The particle/`FFTWGrid` NUFFT supports two kernel choices. `PSWFKernel<N>` is
the fiducial native implementation of the order-zero prolate spheroidal wave
function kernel, with width selection following the PSWF settings introduced in
[FINUFFT 2.5](https://github.com/flatironinstitute/finufft/releases/tag/v2.5.0).
It solves the angular PSWF eigenproblem in an even Legendre basis, fits the
particle weights with Chebyshev polynomials, and evaluates the kernel Fourier
transform by Gauss-Legendre quadrature. It does not depend on FINUFFT.

The default convenience calls use PSWF. To use ES, construct an `ESKernel`
explicitly and pass it to the same transform APIs. Construct and reuse either
kernel across calls:

```cpp
FML::NUFFT::PSWFKernel<3> kernel(1e-6, 2.0); // Fiducial; ES is also supported.
FML::NUFFT::nufft_type1<3>(particles, num_particles, weights, norm, modes, kernel, &order);
FML::NUFFT::nufft_type2<3>(modes, particles, num_particles, values, kernel, &order);
FML::NUFFT::compute_force_nufft_with_kernel<3>(
    particles, num_particles, num_particles_total, nmesh, poisson_norm, force, kernel, nullptr, &order);
```

In a local 3D comparison (300,000 random particles, 64^3 modes, requested
accuracy 1e-6, sigma 2, four OpenMP threads), PSWF reached about 1e-9 relative
max error against FINUFFT 2.5.1 for both transform types. Spatially ordered
PSWF was a few percent faster than spatially ordered ES in repeated runs on
this host, and remained about 10--15% slower than FINUFFT. Timings varied
substantially between runs, so this is an initial comparison rather than a
portable performance claim. PSWF is the default based on its stronger observed
accuracy and small observed speed advantage over ordered ES; ES remains
available as an explicit alternative.

`Core.h` now defines the standalone option, point, shape, and full-complex mode
array types. Its canonical signed-mode mapping is implemented; transform
backends and execution plans are the next implementation stages.

The type definitions follow the standard NUFFT conventions documented by
[FINUFFT](https://finufft.readthedocs.io/en/latest/math.html). FML will implement
its own API and algorithm; it is not a FINUFFT wrapper or a code port.
