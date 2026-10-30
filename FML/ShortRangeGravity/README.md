# Short-range gravity: P3M and monopole TreePM building blocks

Header-only C++17 module in `FML::SHORT_RANGE_GRAVITY`. Support is three-dimensional Newtonian or Newtonian-plus-Yukawa gravity in a cubic periodic box, positive or zero source masses, one common softening support, and a cutoff smaller than half the box. Newtonian gravity remains the default. This module provides force calculations; it does not change SimulationCode or its integration scheme (backlog item 47).

See [tests/RESULTS.md](tests/RESULTS.md) for measured validation, timings and current limitations.

## Newtonian force split and softening

Use a Gaussian long-range potential filter `exp(-k^2 rs^2)`. The complementary unsoftened pair acceleration is

```
a_short = G m (source-target)/r^3 [erfc(q) + 2 q exp(-q^2)/sqrt(pi)], q=r/(2 rs).
```

The compact cubic-spline density has support `h=softening_support`, with `W=8/(pi h^3) (1-6u^2+6u^3)` for `u=r/h<1/2`, and `W=16/(pi h^3) (1-u)^3` for `1/2<=u<1`. The full force is exactly Newtonian beyond h; **h is the support radius, not the Plummer-equivalent epsilon** (`h=2.8 epsilon`). Inside h we subtract the Gaussian long-range force from the full spline-softened Newtonian force. Multiplying the softened force by the unsoftened short-range window would give a different total force. A small-r series evaluates the Gaussian force without cancellation. Coincident particles have a finite coefficient and zero vector acceleration.

These are the Gaussian split and cubic-spline density conventions described in the [GADGET-2 code paper](https://wwwmpa.mpa-garching.mpg.de/~volker/gadget/gadget2-paper.pdf). The implementation is independent.

`cutoff_tolerance` determines the radius where the short-range pair force, relative to unsoftened Newtonian gravity, falls below the requested tolerance. The force is then truncated, without tapering. The tolerance bounds this individual pair's window; **it does not bound the relative error in a many-particle net force**, particularly near cancellations. Require `h <= cutoff < boxsize/2`. Omitted periodic images have exponentially small but nonzero tails. Check cutoff and split-scale convergence for production use.

## Optional Newtonian-plus-Yukawa model

```cpp
Kernel newtonian(options);                         // Existing model and default.
Kernel also_newtonian(options, Newtonian{});        // Explicit selection.
Kernel modified(options, NewtonYukawa{1.0/3, 10});  // alpha dimensionless; lambda=10 in the coordinate length unit.
```

The point-particle potential outside the softening support is `Phi=-coupling*m/r [1+alpha exp(-r/lambda)]`. The force multiplier is `1+alpha (1+r/lambda) exp(-r/lambda)`. Finite positive lambda and finite alpha are required; negative alpha, including repulsive and cancellation cases, is supported. A zero alpha takes the Newtonian force path exactly, including its original cutoff. The model is selected once in an immutable `Kernel`; search backends and particle ownership are shared.

The long-range Fourier potential kernel is proportional to

```
exp(-k^2 rs^2)/k^2 + alpha exp[-(k^2+mu^2) rs^2]/(k^2+mu^2), mu=1/lambda.
```

`apply_long_range_filter` supplies the corresponding density multiplier before the ordinary continuous-k Newtonian PM solve. The Yukawa piece uses the Helmholtz-aware factor `exp[-(k^2+mu^2)rs^2]`; it is not the common Gaussian filter applied to both full potentials. The analytic complementary short-range Yukawa Green function is

```
[exp(-mu*r) erfc(q-z) + exp(mu*r) erfc(q+z)]/(2r),
q=r/(2rs), z=mu*rs.
```

This is the [Yukawa Ewald decomposition, equations 30–32](https://link.aps.org/accepted/10.1103/PhysRevE.103.043204). Forces are derivatives of the potentials. Scaled erfc evaluation avoids overflowing `exp(mu*r)`. Series and positive heat-kernel integrals resolve small-radius and extreme-range limits. For negative alpha, stable Newtonian-minus-Yukawa differences preserve small forces instead of subtracting two rounded, almost equal accelerations. Near a many-particle cancellation, force accuracy still needs component-based absolute tolerances; a relative error in a nearly zero net force is not controlled by the cutoff or tree opening alone.

Yukawa softening is a numerical **pair-potential prescription**, not the Yukawa field of the spline's extended density (the Newtonian shell theorem does not apply to Yukawa gravity). Let `gN(r)` be the positive spline-softened Newtonian Green function and `s(r)=1/gN(r)`. Define the softened Yukawa term as `exp[-s(r)/lambda]/s(r)`. Its coefficient is `Newtonian_soft_coefficient * (1+s/lambda) exp(-s/lambda)`. This is finite at the centre, matches the exact point Yukawa law outside the support where `s=r`, and tends to the existing Newtonian spline as lambda grows. Both potentials and their forces match smoothly at the support boundary. The short-range force subtracts the matching Ewald long-range force; it does not multiply the softened force by an unsoftened split window.

For nonzero alpha the cutoff conservatively solves `(1+abs(alpha))*Newtonian_short_window=cutoff_tolerance`. The point Yukawa short force is positive and no larger than the Newtonian short force, as follows from its positive heat-kernel integral. Thus this bounds the combined omitted pair force relative to the Newtonian point force outside the softening support, including negative alpha; it does not promise a relative bound against a cancelled total force. The octree additionally resolves lambda in its opening rule. This is deliberately conservative and can make very short-range Yukawa trees slower, even when that contribution is small. Cell-list geometry and ghost routing use the resulting cutoff without changing their algorithms.

`potential(r)` is the full isolated softened pair potential per unit source mass, including coupling; `full_coefficient(r)` is its acceleration coefficient. `long_range_potential` and `long_range_coefficient` expose the Ewald long component (the latter excludes coupling, matching the original Newtonian API). `untruncated_potential` and `untruncated_coefficient` expose the short component with coupling; `coefficient` additionally truncates it. The old `softened_coefficient` remains the Newtonian spline helper. The PM force sets its zero Fourier mode to zero; uniform-potential constants and periodic potential-energy conventions are not supplied by these isolated-pair helpers.

Lambda and all other lengths use the same coordinate unit. The library does not infer a scale factor: for fixed physical range with comoving coordinates, pass `lambda_comoving=lambda_physical/a` at the force epoch. Rebuild the kernel, search structures and ghosts whenever the model, split or softening changes. The large-lambda limit is Newtonian gravity with coupling `(1+alpha)*coupling`; the small-lambda limit recovers Newtonian gravity at fixed nonzero separation and for the softened model. These are linear pair interactions, not an environmental screening prescription.

## API

```cpp
#include <FML/ShortRangeGravity/ShortRangeGravity.h>
#include <FML/ShortRangeGravity/Octree.h>
using namespace FML::SHORT_RANGE_GRAVITY;
Options options;
options.boxsize = 250;             // All lengths in the same coordinate unit, e.g. comoving Mpc/h.
options.split_scale = 2;
options.softening_support = .1;
options.cutoff_tolerance = 1e-6;
options.coupling = 1;              // G, or the explicitly derived simulation prefactor.
Kernel kernel(options);
std::vector<Particle> particles;   // Fill position, mass and globally stable uint64 ID.
auto reference = direct_sum(particles, particles, kernel);
auto p3m = CellList(particles, kernel).compute(particles);
auto treepm = Octree(particles, kernel, TreeOptions{.3, 16, 40}).compute(particles);
```

Results are **attractive accelerations**, with source mass and coupling already included. A target's own mass is not included. Match target/source IDs to omit self interactions. Source IDs must be unique; source positions must lie in `[0,boxsize)` in every dimension. Targets may be a subset or a separate list. Zero masses are allowed; negative masses are rejected. `CellList` and `Octree` reference the source vector: keep it alive and unchanged and rebuild after moving particles, modifying masses, reordering or migration. Each compute allocates its output. OpenMP parallelizes over targets, with no shared force writes; source-pair contributions are evaluated separately for each target. Cell lists preserve pairwise momentum to summation precision; one-sided tree approximation need not do so exactly.

`CellList` uses contiguous offsets/indices, with cell width at least the cutoff. `max_cells` bounds cell metadata by selecting coarser cells; this may increase pair work, but does not omit neighbours. It is not a bound on total memory. Single- and two-cell periodic aliases are handled without duplicate visits. Direct sums and cell lists use the same spatial kernel; independent quadrature and Ewald tests validate the kernel separately.

`Octree` stores flat nodes with mass, centre of mass, bounds and child/particle ranges. It uses monopoles. Acceptance requires a node entirely inside the cutoff, outside the softening region, within a single periodic image, excluding the target's source identity, and sufficiently small relative to both target distance and split scale. The split-scale condition is deliberately conservative: the short-range kernel varies faster than 1/r near the cutoff. Nodes straddling boundaries are opened. `opening_angle=0` gives exact leaf sums; `.3` is a starting value, not a guaranteed relative error. Validate tighter openings against direct sums, including weak/cancelled forces. Coincident sources terminate at `max_depth` even if leaf capacity is exceeded. This is an initial correctness-oriented tree; higher multipoles, relative-error acceptance and tree traversal optimisation remain possible extensions.

`Result::statistics` reports directed nonself candidate and accepted pair counts; `TreeResult` also reports visited/accepted nodes and total tree nodes. Node interactions are not included in pair counters. These counts are local to the calling rank.

## PM coupling and units

`LongRangePM.h` supplies `apply_long_range_filter(fourier,kernel)`. Apply it **once to a dedicated Fourier density or potential copy**, before the ordinary PM gradient calculation. FML wavevectors use unit-box coordinates, so the helper explicitly uses `rs/boxsize`. Initially use the continuous Fourier Laplacian and gradient. Assignment, interpolation, deconvolution, Nyquist handling and finite mesh resolution still affect the combined force; do not assume the analytic split removes mesh anisotropy or aliasing. The standalone tests exercise FML's PM solver for an on-grid source and independently evaluate its filtered Fourier series at off-grid probes. General off-grid deposition/interpolation and production mesh convergence need validation when integrating the caller. Float FFT grids incur additional rounding, amplified in weak net forces: the manufactured Ewald test uses relative thresholds of `5e-5` for float and `2e-7` for double; the pair and tree kernels accumulate in double in either build.

For FML's dimensionless box coordinate `x=r/L`, Poisson equation `laplacian_x Phi=C delta`, and total source mass M,

```
Phi_pair = - C/(4 pi) (m/M) L/|r-rj|
-grad_x Phi_pair = C L^2/(4 pi M) m (rj-r)/|rj-r|^3.
```

Use `fml_acceleration_coupling(C,M,L)` as `coupling`. Here M is the **global total source mass** corresponding to the density field, not local or ghost mass. With unit-box source positions use L=1. In SimulationCode GR, `C=1.5 OmegaM a`; this produces the acceleration that is added with the existing kick integral. FML's mesh API instead returns `+grad_x Phi`, which its kick subtracts. Avoid applying the source normalization, scale factor or kick twice. The helper returns an instantaneous force prefactor; time integration and species conventions remain the caller's responsibility. Screened gravity and relativistic/neutrino sources require separate prescriptions.

## Slab MPI ghosts

```cpp
#include <FML/ShortRangeGravity/SlabGhosts.h>
auto sources = exchange_slab_ghosts(owned, xmin, xmax, kernel);
auto result = CellList(sources.particles, kernel).compute(owned);
// Or Octree(sources.particles, kernel).compute(owned).
```

This is collective on `FML::communicator`. Slabs, in rank order, must form a contiguous partition of `[0,L)`; bounds are in the same length unit as positions. Owned particles must already have migrated to their slabs. IDs must be globally unique; the function checks local source IDs, but does not globally gather IDs to validate that contract. All ranks must use identical box, split, softening, cutoff tolerance, coupling and interaction model (including alpha and lambda). Model mismatches are rejected collectively. Empty particle slabs are supported; zero-width domains are not.

The exchange sends each source once to every remote slab within the cutoff, including periodic and nonadjacent slabs. It does not replicate particle payloads through an all-gather. Only small settings/count arrays are exchanged globally. A rank's result starts with its owned particles, followed by ghosts with base-box positions. Buffers pack fields explicitly and exchange bounded chunks with 64-bit counts, avoiding padding and MPI int-count overflow. Sender/receiver rounds agree collectively. The current implementation scans local particles per destination and uses a ring exchange with global round counts; it favours simplicity over communication overlap. Narrow slabs or a large cutoff can still require nearly all particles as ghosts. Ghost storage is not bounded by `max_cells`. Rebuild ghosts and local search structures after each drift; the function does not modify ownership or velocities. `sent_particles` and `received_particles` expose traffic per rank.

## Tests and timing

```
make -C FML/ShortRangeGravity/tests run USE_MPI=false
make -C FML/ShortRangeGravity/tests run USE_MPI=true NP_LIST="1 2 4"
OMP_NUM_THREADS=4 FML/ShortRangeGravity/tests/test_short_range --benchmark
# After building with MPI:
OMP_NUM_THREADS=2 mpirun -np 4 FML/ShortRangeGravity/tests/test_short_range --benchmark
```

Both `test_short_range` and `test_yukawa` are registered in the main `FML/tests` suite. The Yukawa tests add 100-digit Ewald reference values, independent periodic image sums, softened potential derivatives, range limits, negative-alpha cancellation cases and MPI model-agreement checks. `tests/generate_yukawa_reference.py` regenerates the reference header with mpmath at 100 digits; format that header with `FML/.clang-format` after regeneration. FFTW and GSL are needed by the PM validation (via NBody headers); the pair kernel, cell list and octree themselves have no external dependencies. The ghost helper uses FML Global/MPI. Make clean when changing compiler, precision or MPI options in the same build directory.

Tests include independent spline enclosed-mass quadrature; split and zero-separation limits; tolerance cutoffs; direct/cell/tree agreement for uniform/clustered/coincident/periodic particles; cell aliases and empty catalogues; thread determinism; monopole acceptance and tighter opening; periodic Ewald validation at multiple split/softening scales; FML force sign and normalization; length/mass-unit invariance; ghost completeness, empty/nonadjacent slabs and multi-batch messages; and collective input errors. The benchmark reports maximum-rank build-plus-force and ghost-exchange times, global candidate counts, maximum-rank relative RMS force error and maximum ghost payload bytes for 8000 uniform/clustered particles. It deliberately holds a replicated catalogue for the direct reference in the test harness; the force/ghost implementation does not gather that catalogue. Ghost payload bytes exclude vector capacity, owned particles, tree/cell metadata, communication buffers and allocator overhead; they are not RSS. Timings are machine dependent and are not production scaling claims.

Remaining work under item 47 is SimulationCode force assembly, interlacing/lightcone handling, standard N-body/COLA simulation validation and short-range subcycling. Further production validation should include mesh/interpolation convergence, large clustered MPI workloads, tree force-error percentiles and memory/load-balance measurements.
