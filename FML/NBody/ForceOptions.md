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
`gradient_kernel` applies to Fourier force differentiation. For an even mesh every
gradient kernel is zero for the Nyquist component k_i = k_Ny (the continuous kernel
i k_i too: that mode is its own mirror image, so its derivative is ambiguous), which
keeps the force of a reflected density field the reflected force. The potential routine
does not deconvolve: deconvolution is performed by the subsequent force routine,
using the density-assignment/interpolation kernel supplied there.

`laplace_kernel_from_name` and `gradient_kernel_from_name` convert `continuous`,
`discrete_2pt` and `discrete_4pt` to typed enum values. Empty strings and `fiducial`
resolve to continuous kernels. Invalid names and invalid enum values are rejected
before parallel force work.

## Force grid finer than the particle grid

The defaults (continuous kernels, no deconvolution) are a poor choice when the
force grid is an integer factor finer than the particle grid, e.g. `Nmesh = 2 Npart_1D`,
with CIC. `NBodyInitialConditions` puts the particles on a regular lattice, and that
lattice then sits exactly on grid nodes, where the CIC kernel has a kink. As long as
the displacements are small compared to a force cell the assigned density responds to
the size of the displacement, which gives a signal at the grid Nyquist frequency, and
the continuous gradient (large near Nyquist) turns it into spurious forces.

A linear test (64^3 particles, Nmesh = 128, CIC, amplitude so small that the particles
hardly move, 30 steps; P/P_lin at k = 1..6 times the fundamental mode at the end):

| Setup                                            | P/P_lin                        |
|--------------------------------------------------|--------------------------------|
| continuous gradient (default)                    | 0.54 1.65 2.76 3.42 2.99 3.81  |
| continuous, lattice shifted by half a force cell | 1.05 1.09 1.13 1.18 1.20 1.24  |
| TSC, continuous                                  | 1.02 1.03 1.04 1.04 1.04 1.04  |
| `discrete_2pt` gradient                          | 0.99 0.99 0.99 1.00 1.00 1.01  |
| `discrete_4pt` gradient                          | 0.99 1.00 1.02 1.01 1.05 1.09  |
| shifted, `discrete_4pt` gradient, `discrete_2pt` Laplacian | 0.99 0.99 0.99 0.99 0.99 0.99 |
| Nmesh = 64 (= particle grid), continuous         | 0.98 0.96 0.94 0.97 0.88 0.84  |

With realistic amplitudes the particles leave the nodes early and the effect is smaller:
in the NBody example (COLA, 128^3 particles in 1000 Mpc/h from z = 20 on a 256^3 grid)
the continuous gradient gives 3-6% more power at k < 0.03 h/Mpc than the same run with
TSC, while the two-point gradient agrees with TSC to 1-2%. The effect dominates clean
linear tests, very early starts and low-amplitude boxes. With a finer force grid
than the particle grid, set

```cpp
FML::NBODY::ForceOptions options;
options.gradient_kernel = FML::NBODY::DISCRETE_TWO_POINT_GRADIENT;
```

or use TSC, or (as SimulationCode does by default, see `SimulationCode/EVOLUTION.md`)
offset the lattice by half a force cell and use the four-point gradient.
`NBody/example` uses the two-point gradient.

## Migration

The old `FIDUCIAL_LAPLACE_KERNEL`, `FIDUCIAL_GRADIENT_KERNEL`,
`FIDUCIAL_DECONVOLVE_FORCE` globals and `set_fiducial_*` setters are removed.
Construct a local `ForceOptions` and pass it at each force/potential call instead.
Calls that previously relied on unchanged defaults require no new argument.
Custom SimulationCode gravity classes must add the final options argument to their
`compute_force` override and forward it to `compute_force_from_total_density`.

SimulationCode owns its resolved options per simulation and passes them explicitly
through every gravity model. Its parameter names, automatic gradient choice and
early-time deconvolution switching are unchanged. In particular, `fiducial` no
longer inherits a kernel that another simulation happened to select earlier.

This changes configuration ownership only: the force kernels and deconvolution
formulas are unchanged. It introduces no grid workspace or force caching.
