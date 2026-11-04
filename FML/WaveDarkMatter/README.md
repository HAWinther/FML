# Spectral wave / fuzzy dark matter

`FML::WAVEDM` is a C++17, three-dimensional periodic Schrödinger–Poisson library.
It evolves a full complex wavefunction with FFTW complex-to-complex transforms,
supports serial/MPI slabs and OpenMP loops, and supplies static and prescribed
cosmological expansion. It does not require GSL. This is the initial library
implementation of backlog #49; it does not add a SimulationCode evolution mode.

## Equations, normalization and units

Choose one length unit L0, time unit T0 and mass unit M0. Coordinates range over
`[0,boxsize)` in these code length units; unlike `FFTWGrid` particle coordinates,
they are not automatically divided by the box length. Define `eta=hbar/m` in
L0²/T0 and G in L0³/(M0 T0²). With the rescaled comoving wavefunction psi,

```
rho_comoving = |psi|^2                    [M0/L0^3]
mass = integral |psi|^2 d^3x              [M0]
rho_physical = rho_comoving/a^3
U = a Phi_peculiar                       [L0^2/T0^2]
i dpsi/dt = -eta/(2 a^2) laplacian(psi) + U/(eta a) psi
laplacian(U) = 4 pi G (rho_comoving + rho_external - mean_total_density)
```

This absorbs the homogeneous a^(-3/2) dilution into psi. It describes one
nonrelativistic scalar species with no self-interaction, in a prescribed background.
Time is cosmic time in T0, and a is dimensionless. At a=1 the equations reduce to
the non-expanding Schrödinger–Poisson system. `Physics{1,1}` gives the usual
`i psi_dot=-laplacian(psi)/2+U psi`, `laplacian U=4 pi rho_contrast` convention.
The wavefunction represents mass density, not a particle-number wavefunction; no
extra axion-mass factor belongs in this Poisson source.

```cpp
#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
using namespace FML::WAVEDM;
const auto physics = Physics::from_si(
    1e-22,                             // Boson rest mass in eV/c^2.
    FML::UTILS::SI::kpc,                // One code length in metres.
    FML::UTILS::SI::Gyr,                // One code time in seconds.
    1e7 * FML::UTILS::SI::Msun);        // One code mass in kilograms.
Solver solver(64, 20.0, physics);       // 64^3 points; comoving box side 20 code lengths.
solver.initialize_density_phase(
    [](auto x) { return 1.0 + .01*std::cos(2*M_PI*x[0]/20.0); },
    [](auto) { return 0.0; });
solver.step(.001);                     // Freeze a at its current value (initially 1).
auto d = solver.diagnostics();
```

Conversions are explicit: eta_code=(hbar/m)_SI T0/L0² and
G_code=G_SI M0 T0²/L0³. The module does not infer h, Omega or a density unit from a
power spectrum. Solving a cosmological model also requires a consistent background,
mean density, initial field and physical resolution. The runnable example uses a
labeled analytic toy rather than a calibrated cosmology.

## Complex grid, layout and normalization

`ComplexGrid.h` provides a full N³ complex field. Forward transforms are normalized
by N³ and inverse transforms are unnormalized, so stored coefficients are Fourier
series amplitudes. All positive and negative modes are independent; this field
cannot be placed in an r2c grid or reconstructed with Hermitian symmetry. Kinetic
and Poisson operators use the full k², including even-mesh Nyquist modes. Signed
mode indexing assigns +N/2 to an even-mesh Nyquist component; k² is unambiguous.
The odd derivatives in `gravity_acceleration` zero that component, consistent with
FML's real PM-force convention.

FFTW MPI returns both an owned slab and a possibly larger required allocation.
The class honors the complete allocation, not just `local_nx*N*N`. It uses aligned
storage with reusable forward/inverse plans, with FFTW_ESTIMATE planning and no
transposed-output layout. Only owned elements enter field operations and sums.
Empty slabs, uneven partitions and odd meshes are supported. No FML particle-domain
or r2c-grid partition is assumed: use `wave().local_start()`, `local_nx()`,
`coordinate(i)` and `position(i)` for the actual complex-grid layout.

Float/double/long-double FFT symbols follow FML's precision defines. Double and
float MPI paths are tested; long-double evolution is not in the tested matrix and
checkpoint version 1 deliberately supports float/double only. OpenMP parallelizes
pointwise evolution; FFTW threading follows the existing FML build/runtime setup.
Planning, collectives and method calls must run in the same order on all ranks and
outside parallel regions. A solver owns mutable workspaces and is not safe for
concurrent calls from multiple threads.

`wave()` exposes owned complex samples for custom ICs or inspection. Operations
expect a position-space field; do not replace the grid with another layout.
References/pointers to a wave buffer must be reacquired after a successful advance,
because the candidate and live buffers swap. A solver is noncopyable. Grid objects
must be destroyed before FML finalizes MPI/FFTW, as with existing local FML grids.

## Split steps and expansion

For each advance the code performs

```
psi <- exp[-i U_initial B/(2 eta)] psi
psi_k <- exp[-i eta k^2 A/2] psi_k
U_final <- periodic Poisson solve from the evolved density
psi <- exp[-i U_final B/(2 eta)] psi
A = integral dt/a^2, B = integral dt/a
```

Each potential substep preserves density exactly in exact arithmetic, so the
potential is recomputed after the kinetic substep. The solver does not evolve the
old potential through a full step or incorrectly use the initial density twice.
At fixed a this is the symmetric second-order Strang split; using the integrated
coefficients gives a symmetric second-order step for prescribed smooth expansion.
The error is O(dt³) locally and O(dt²) at fixed final time. A unitary kinetic step
can have a large phase while still being exact for a free wave; this does not
establish accuracy of the interacting split.

`step(dt)` freezes the current a. `step_scale_factor(a_next,H,panels=4)` obtains

```
dt = integral da/(a H(a))
A  = integral da/(a^3 H(a))
B  = integral da/(a^2 H(a))
```

with composite 16-point Gauss–Legendre quadrature; H must be finite, positive and
in inverse T0. Large background intervals need panel/step convergence checks.
`advance(TimeFactors)` accepts externally integrated factors, e.g. from an existing
cosmology. The initial a must match the solver epoch and factors must have consistent
signs. Negative steps support reversal tests. Zero steps preserve the state and
counter. Use the same background callback/parameters on every rank; inconsistent
computed factors fail collectively.

`Solver(mesh,box,physics,time,a)` sets the initial epoch. Physical time, a and the
step counter update only on success. Invalid fields, inconsistent settings and
nonfinite phases raise collective exceptions before committing a candidate. The
live wavefunction remains unchanged after a numerical validation failure; candidate
workspaces/counters may change. Catastrophic allocation or MPI errors have the same
limitations as other FML collective algorithms.

`suggested_timestep(phase_limit=.5)` bounds the instantaneous full-cube kinetic
phase and the maximum potential phase at the current a. It is a conservative
suggestion, not an error estimator or an adaptive integrator. No norm repair,
high-k filtering or implicit renormalization hides accumulated errors. Nonlinear
products and phase multiplications are collocation operations and can alias;
spatial convergence and a resolved de Broglie wavelength remain necessary.

## Initial conditions and density/particle hooks

* `initialize(field)` accepts complex position-space values in the chosen density
  units. Initializer callbacks run outside OpenMP and commit only after validation.
* `initialize_density_phase(rho,theta)` sets `sqrt(rho)*exp(i theta)`. rho must be
  nonnegative and theta finite. Zeros of rho are allowed; phase gradients there
  cannot be interpreted as a unique fluid velocity.
* `initialize_density_velocity_potential(rho,sigma)` uses theta=a*sigma/eta when
  peculiar velocity u=grad(sigma). This is a single-stream, irrotational IC mapping,
  with a periodic wavefunction required at the boundaries. It is not a general
  inversion of a vortical or multistream particle catalogue. Direct complex ICs
  can represent quantized winding/plane-wave states.

A physically consistent FDM transfer/growth solution must be supplied by the
caller. The module does not silently substitute a CDM transfer or approximate an
arbitrary wavefunction with density alone. It does not yet construct self-consistent
cosmological axion ICs directly from a TransferProvider.

`set_external_density(local_rho)` includes an additional absolute **comoving mass
density** in the same code units and complex-grid owned layout. It can come from
CDM particles or a fixed source, but must not already include the wave density.
It remains fixed during one advance; callers must use a consistent joint time
integration when evolving sources. Empty owned slabs supply empty vectors. All
ranks participate in setting or clearing a source.

`potential()` returns zero-mean U=a*Phi in code velocity². It is a reused workspace
view and is overwritten by subsequent computations. `gravity_acceleration()`
returns local real vectors `-grad U/a^3`, the force term in
`x_ddot+2 H x_dot=-grad U/a^3`. This is **comoving coordinate acceleration**, not
FML's existing kick-integrated PM force or physical peculiar acceleration. For
peculiar velocity evolution, `u_dot+H u=-grad U/a^2`. Convert explicitly before
coupling to a particle integrator; particle ownership and interpolation must match
this complex layout or be redistributed. The module supplies gravity hooks, not
a fully advanced two-way wave-plus-particle N-body simulation.

## Diagnostics and cosmological energy balance

`diagnostics(tail_start=.8)` reports mass, maximum comoving density, kinetic energy,
gravitational potential energy and the mass fraction in modes with any component
at or above the specified fraction of the Nyquist wavevector. No Hermitian mode
multiplicity factors are used for this full complex spectrum.

```
K = eta^2/(2 a^2) integral |grad psi|^2 d^3x
W = 1/(2 a) integral rho_total U d^3x
E = K + W
```

K is computed by full-complex Parseval summation. Without external sources these
are the wave mass/kinetic/self-gravitational energies. With a source, W includes
its self-energy and the wave/source cross-energy, but K contains only wave kinetic
energy. This is not a complete energy budget for moving CDM particles. Time-varying
external sources require their kinetic and work terms before interpreting a joint
conservation law.

For the closed wave system, the energy functional satisfies
`i eta psi_dot=delta E/d psi*`. Its explicit scale-factor dependence gives

```
dE/dt = -H (2K+W)
E(a)-E(a0) + integral_(ln a0)^(ln a) (2K+W) dln a = 0.
```

`EnergyBalance::observe(a,K,W)` accumulates this integral by a trapezoidal rule and
returns the residual. For a static background E is conserved to integration error;
for expansion E itself is not constant. The diagnostic integral has its own
second-order sampling error; refine observation intervals as well as the solver.
It is an energy-balance diagnostic, not a stopping/convergence criterion. The
tracker is separate from the solver: retain/serialize its history separately if a
continuous balance across checkpoints is needed. Float FFT rounding can dominate
this residual once temporal errors become smaller than the precision floor.

## Memory and FFT work

Three full complex slabs persist: live psi, candidate psi and a shared Poisson/
diagnostic workspace. A successful step reuses plans and buffers; no complete
field is all-gathered. Two double real arrays hold k² and U; an optional Real array
stores external density. `memory()` reports their allocated capacities, including
FFTW's larger complex allocation when needed. It excludes opaque FFTW plan/scratch
allocations and process overhead; use peak RSS alongside the report.

A self-gravitating advance performs six complex FFTs (two potential solves and
one kinetic pair). G=0 skips the four potential FFTs. Diagnostics and force hooks
perform additional transforms, all counted by `fft_count()`. No per-step full-grid
allocation is required, though FFTW itself may have internal workspace costs.
Global reductions only transmit scalar diagnostics/settings. The example reports
measured max-rank time, local accounted bytes and transform counts. No GPU, pencil
layout, automatic memory budget or production scaling claim is made.

## Versioned checkpoints

```cpp
#include <FML/WaveDarkMatter/Checkpoint.h>
save_checkpoint(solver, "new_prefix");
auto resumed = load_checkpoint("new_prefix");
```

Version 1 writes `prefix.rankR.wave` slabs and publishes `prefix.manifest` last.
The manifest stores N, box size, eta, G, time, a, completed steps, external-source
presence and the original slab partition. Each shard starts with
`FML_WAVE_SHARD 1 N start nx`, followed by position-order records of three
little-endian IEEE binary64 numbers: Re(psi), Im(psi), rho_external. A trailing
64-bit FNV-1a checksum covers payload bytes. Checksums detect accidental corruption;
they are not cryptographic authentication. Both file length and metadata are checked.

Each old shard is verified once by an assigned reader with a bounded byte buffer.
Current ranks then seek/read only the slab intersections they need, so changing
MPI task count neither replicates a global field nor requires a root-owned full
snapshot. Float samples are represented exactly in binary64; loading a double
checkpoint into float is a deliberate precision conversion, not an exact restart.
Long-double checkpoints are rejected. Source values are retained; the background
H(a), source evolution policy and energy-tracker history are caller configuration.

Use a new prefix and a shared filesystem. Existing outputs are rejected, shards
are written/renamed before the manifest, and a failed save can leave orphan files
that must be cleaned before reusing that prefix. This is manifest-last publication,
not a transactional multi-file storage engine. IO/prefixes and numerical calls
are collective. Loading checks version, partition, checksum, sizes and finite,
representable samples before returning a solver.

## Build, tests and limits

```
make -C FML/WaveDarkMatter/tests run
make -C FML/WaveDarkMatter/example
make -C FML/WaveDarkMatter/tests USE_MPI=false USE_OMP=false run
```

The [shared build guide](../BUILDING.md) covers library paths and compiler overrides.
`tests/generate_reference.py` optionally regenerates committed adaptive SciPy ODE
and radial BVP fixtures; Python/SciPy are not needed for normal builds or tests.
Independent direct DFTs, analytic dispersion/interference/Poisson fields, nonlinear
timestep/energy convergence, backward evolution, EdS time integrals, expanding
energy balance, soliton resolution/volume checks and corrupt/repartitioned
checkpoints are covered. See [measured results](tests/RESULTS.md).

Self-interactions, isolated/absorbing boundaries, relativistic axion evolution,
self-consistent live-CDM integration, direct axion transfer-function IC preparation
and production cosmological/halo calibration remain extensions. A stationary
isolated soliton is only approximately stationary in a finite periodic box; volume
and mesh convergence must be distinguished. A norm-conserving calculation can
still be spatially unresolved or have inaccurate phase/energy evolution.

The static equations and split-step reference are described in
[Edwards et al., PyUltraLight (2018)](https://arxiv.org/abs/1807.04037).
The [FFTW distribution contract](https://www.fftw.org/fftw3_doc/Basic-and-advanced-distribution-interfaces.html)
explains why allocation can exceed the owned slab. The comoving factors and energy
identity above follow by rescaling density and varying the stated energy functional;
this implementation is independent and is not native PyUltraLight format compatibility.
