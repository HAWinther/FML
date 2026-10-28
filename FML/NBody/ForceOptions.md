# Explicit force discretization

Force routines accept a final `const ForceOptions &` argument. An omitted argument
means continuous Laplacian and gradient kernels with deconvolution disabled.
There are no mutable process-wide force defaults.

```cpp
#include <FML/NBody/NBody.h>
FML::NBODY::ForceOptions options;
options.laplace_kernel = FML::NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT;
options.gradient_kernel = FML::NBODY::DISCRETE_FOUR_POINT_GRADIENT;
options.deconvolve = true;
FML::NBODY::compute_force_from_density_fourier<3>(density, force, "CIC", norm, options);
```

The same argument is accepted by the real-density force wrapper, the Fourier
Poisson-to-potential routine, the finite-difference potential-to-force routine,
and the example KDK/Yoshida steps. Passing different options on alternating calls
does not affect other calculations. A caller must not mutate an options object
while another thread is using it.

For finite-difference forces, the potential calculation uses `laplace_kernel`;
the force calculation uses `deconvolve` and its explicit `stencil_order` argument.
`gradient_kernel` applies to Fourier force differentiation. The potential routine
does not deconvolve: deconvolution is performed by the subsequent force routine,
using the density-assignment/interpolation kernel supplied there.

`laplace_kernel_from_name` and `gradient_kernel_from_name` convert `continuous`,
`discrete_2pt` and `discrete_4pt` to typed enum values. Empty strings and `fiducial`
resolve to continuous kernels. Invalid names and invalid enum values are rejected
before parallel force work.

## Migration

The old `FIDUCIAL_LAPLACE_KERNEL`, `FIDUCIAL_GRADIENT_KERNEL`,
`FIDUCIAL_DECONVOLVE_FORCE` globals and `set_fiducial_*` setters are removed.
Construct a local `ForceOptions` and pass it at each force/potential call instead.
Calls that previously relied on unchanged defaults require no new argument.
Custom COLASolver gravity classes must add the final options argument to their
`compute_force` override and forward it to `compute_force_from_total_density`.

COLASolver owns its resolved options per simulation and passes them explicitly
through every gravity model. Its parameter names, automatic gradient choice and
early-time deconvolution switching are unchanged. In particular, `fiducial` no
longer inherits a kernel that another simulation happened to select earlier.

This changes configuration ownership only: the force kernels and deconvolution
formulas are unchanged. It introduces no grid workspace or force caching.
