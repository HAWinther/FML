# Euler Riemann solvers and finite-volume gas dynamics

C++17, header-only ideal-gas hydrodynamics in `FML::HYDRODYNAMICS`. The module now
provides Rusanov, HLLE and HLLC fluxes, an exact 1D Riemann reference, and a
conservative uniform-Cartesian finite-volume solver in 1D/2D/3D with slab MPI and
OpenMP. It shares `EquationOfState.h` with [SPH](../SPH/README.md). FFTW/GSL are not
required. This is the initial implementation of backlog #51; measured results
and limits are in [tests/RESULTS.md](tests/RESULTS.md).

## Basic use

Initialize FML once. All ranks construct and call the same solver operations,
including ranks owning no cells.

```cpp
#include <FML/Hydrodynamics/FiniteVolume.h>
namespace H = FML::HYDRODYNAMICS;
H::Options<1> options;
options.shape = {256};
options.lengths = {1};
options.gamma = 1.4;
options.boundaries = {{{H::Boundary::Outflow, H::Boundary::Outflow}}};
H::FiniteVolume<1> gas(options, [](const auto& x) {
    return x[0] < .5 ? H::Primitive<1>{1, {0}, 1}
                     : H::Primitive<1>{.125, {0}, .1};
});
while (gas.time() < .2)
    gas.advance(.2 - gas.time());
auto local = gas.primitive_cells();
auto totals = gas.diagnostics();
```

`Primitive<D>` holds density, velocity[D] and pressure. `Conserved<D>` is an array
ordered as density, momentum[D], total energy density, auxiliary internal energy
density. `conserved`, `primitive`, `recover`, `normalize` and `physical_flux`
provide reusable conversions/operations. `IdealGas` supplies pressure, sound
speed squared, pressure-to-specific-energy conversion and entropy proxy with
constant gamma>1. The tested thermodynamic cases use gamma=1.4 and 5/3; general
or nonideal EOS are not supplied by this initial solver.

Use consistent length/mass/time units. Total energy is rho*u+rho*v²/2, with u the
same specific internal energy stored in an SPH particle. Neither the mesh nor
the EOS infers cosmological units, a gravitational constant or source factors.

## Riemann interfaces and exact reference

```cpp
#include <FML/Hydrodynamics/Riemann.h>
#include <FML/Hydrodynamics/ExactRiemann.h>
H::IdealGas eos(1.4);
H::Primitive<1> left{1, {0}, 1}, right{.125, {0}, .1};
auto face = H::riemann(left, right, eos, 0, H::FluxMethod::HLLC);
if (!face.valid) { /* inputs or flux are unrepresentable */ }
H::ExactRiemann exact(left, right, eos);
auto q = exact.sample((x - x0) / time);
```

`FluxResult` includes the flux, interface velocity, maximum signal speed, actual
method and explicit fallback reason. Invalid or unrepresentable states return
`valid=false`. Inputs have strictly positive finite rho/P and representable
sound speed/conserved variables. Only dimensions 1–3 and valid normal axes are
supported. No pressure/density floor is applied.

Rusanov uses the largest end-state |v_n|+c. HLLE uses Roe/Einfeldt bounds widened
to include both physical end-state sound cones; tangential kinetic energy is
included in the Roe enthalpy. Its interface velocity is derived from the HLL
intermediate momentum/density, rather than its diffusive mass flux. HLLC restores
the contact wave and preserves tangential contact states. It falls back to HLLE
for generated vacuum, degenerate waves or nonpositive/unrepresentable star states.
Finite-volume updates can fall back further to first-order Rusanov as described
below. There is no separate Roe flux or Roe entropy-fix switch.

The auxiliary component is a conservative advected internal-energy *density*,
with pressure work handled by the evolution layer. An optional final two-element
array argument to `riemann` supplies its left/right face densities; otherwise they
are initialized from the physical rho/P states. HLLE/Rusanov apply their normal
conservative scalar flux, while HLLC transports it through its contact/star
states. This distinction matters for kinetic-dominated tangential contacts.

`ExactRiemann` solves the ideal-gas pressure matching equation with a bracketed
log-pressure solve, then samples shocks, contacts and rarefaction fans. Positive
initial states can generate exact vacuum; `has_vacuum()` identifies it, and
`sample` returns rho=P=0 inside the vacuum interval (velocity there is a reference
convention). Star quantities are meaningful only without vacuum. This reference
is independent of the approximate fluxes and not an evolution option. The FV
mesh requires positive initial gas and represents generated vacuum by a positive
finite-resolution approximation. Initially empty/zero-density cells and
unrepresentably small exact star pressures are not supported.

References: [HLLC construction](https://doi.org/10.1007/BF01414629),
[Clawpack's Euler Riemann discussion](https://www.clawpack.org/riemann_book/html/Euler_approximate.html).
The implementation is independent; these links describe the methods, not code
compatibility with another solver.

## Reconstruction, evolution and positivity

The finite-volume discretization sums directional Euler flux divergences in one
unsplit method-of-lines RHS. Order 2 uses componentwise primitive linear
reconstruction with generalized minmod/MC slopes (theta in [1,2], default 1.5).
Order 1 uses cell states. Time stepping is SSPRK2 in both cases. Reconstruction
that is nonphysical/unrepresentable is reset to the cell state; it never clips
the evolved density or pressure. Limiters reduce local order at extrema and
shocks; the results document observed refinement orders rather than assuming
universal second-order convergence.

Each forward-Euler trial is checked collectively. A failed high-order trial is
recomputed with first-order Rusanov fluxes over the entire distributed grid,
which retains one common conservative flux at each interface. `advance(max_dt)`
then halves dt and retries the full step if needed, up to `max_retries`. `step(dt)`
uses the caller's fixed dt and throws if fallback still fails. A failed operation
preserves the conserved cells and epoch. Work/fallback counters can increase.
There is no silently imposed temperature, density or pressure floor.

`advance` uses a global unsplit CFL bound based on directional face signal maxima
and physical cell sizes, plus explicit expansion/gravity timestep limits. It
returns the actual dt. A timestep must advance the epoch representably. An
explicit `step` caller remains responsible for stability/CFL. Positive trial
checks and tested fallback behavior are not an all-input stability theorem.

`cells()`, `layout()`, `options()`, `statistics()` and `time()` are local accessors.
`primitive_cells()` returns an owning local vector. `diagnostics()` collectively
integrates mass, momentum, total energy, kinetic energy and pressure-derived
thermal energy. `set_state(local_conserved, epoch)` collectively validates and
normalizes replacement cells before mutation and establishes a new statistics
reference. The provided layout exposes centers, global IDs, shapes, start/width
and cell volume; it does not require gathering a mesh.

## Optional dual energy

Default evolution conserves total Euler energy, recovering pressure from E−K.
For high-Mach problems, set `options.dual.enabled=true`. The auxiliary rho*u
then evolves with its own reconstructed advective flux and pressure-work source.
When (E−K)/E falls below `pressure_threshold` (default 1e-3), pressure uses the
positive auxiliary value. A negative E−K is recoverable only if its magnitude is
also below that relative bound; grossly bad total energy is still rejected.
Reliable total thermal energy synchronizes the auxiliary above
`synchronization_threshold` (default 1e-2), or if the auxiliary is invalid.

When auxiliary pressure is selected, E is reset to K+rho*u_aux. This improves
pressure fidelity but changes stored total energy: it is an explicit,
nonconservative dual-energy correction. `energy_repair` and
`absolute_energy_repair` record signed/absolute accepted-step contributions,
weighted by the RK combination and cell volume. Initialization/replacement
repairs have separate `initial_*` counters. Rejected-trial repairs are discarded.
`dual_cells` counts accepted normalization evaluations, not unique spatial cells.
Compare ledger *changes* when defining an energy budget.

High-Mach advection and an HLLE tangential-contact test verify temperature
retention. This does not establish production high-Mach shock, turbulence or
cosmological calibration. At large bulk kinetic energy, a correction can be tiny
relative to total energy while large relative to the thermal energy; inspect
both pressure accuracy and the repair ledger.

## Boundaries, MPI and workspace

Each dimension independently supports paired periodic, zero-gradient outflow or
reflecting boundaries. Reflection changes only the normal momentum; multiple
reflections in one-cell domains are handled. Outflow means zero gradient and
can admit inflow when its copied velocity points inward; it is not a diode.
Two halo layers support the limited slopes, including mixed-boundary corners.

The mesh uses integer x slabs `[floor(rank*Nx/P), floor((rank+1)*Nx/P))`, with full
transverse planes. Nx need not divide P; one-cell slabs and empty ranks work.
This layout is explicit and independent of FML's particle ownership/domain
object. Gravity adapters must honor this mesh layout rather than assume an
unrelated FFTW or particle decomposition.

Halo plans route the required planes to their actual owners, including beyond
adjacent ranks when slabs are narrow. Cell records use the shared bounded
`Global/ChunkedExchange.h`; no mesh payload is all-gathered. Records assume a
homogeneous MPI ABI. `max_message_bytes` bounds payload/receive scratch, not
outgoing staging or total storage. `max_local_cells` bounds the padded grid;
owned RK/RHS arrays, primitive caches, face fluxes and halo metadata add memory.
Buffers for halo cells, reconstructed primitives and face fluxes retain capacity;
RK trials/RHS vectors are owning allocations. No peak-RSS/scaling claim is made.

OpenMP parallelizes faces and cell rates without concurrent writes to a face or
cell. Seam fluxes are evaluated by both ranks from identical halos. Physics
options and global decisions agree collectively. Fallback/reset counters count
work evaluations, including repeated trials and duplicate seam work, and can
therefore vary with rank count. The shared transport is not a general
allocation-failure/fault-tolerant MPI layer.

## Gravity and prescribed expansion

A collective `Gravity` callback receives `(time, layout, primitive_cells,
extra_acceleration)` and supplies dv_pec/dt in the solver's consistent units.
Momentum receives rho*g and total energy receives m.g, once. The timestep also
includes the supplied acceleration. Callbacks run at actual trial states/epochs,
possibly repeatedly during fallback/retries, including on empty ranks. They must
be deterministic for those inputs and coordinate errors around their own MPI
collectives. Shape/nonfinite output and callback presence are checked. Callback
side effects cannot be rolled back.

For 3D, a pure `Background(time)` can supply consistent a(t)>0 and H=adot/a in
physical time. Grid lengths are then comoving, velocity is physical peculiar
velocity, stored density is rho_c=a^3*rho_phys and pressure is P_c=a^3*P_phys.
Sound speed computed from P_c/rho_c is physical. All spatial flux divergences
carry 1/a. Momentum sources are −H*m+rho_c*g; total-energy sources are
`m.g−2H*K−3H*P_c`. The auxiliary pressure work is
`−P_c*(div_x(v_pec)/a+3H)`. No additional gravity factor of a or FML Poisson/kick
normalization is inferred. Nontrivial expansion is restricted to 3D.

These hooks define explicit initial coupling conventions. The module does not
supply a self-gravity/Poisson solver, evolve collisionless species or enable gas
in SimulationCode automatically. Coupled gravity energy budgets and production
cosmological gas tests remain follow-ups.

## Build and verification

```sh
make -C FML/Hydrodynamics/tests run USE_MPI=false USE_OMP=false
make -C FML/Hydrodynamics/example USE_MPI=false USE_OMP=false
# Clean before changing MPI/OpenMP flags.
make -C FML/Hydrodynamics/tests clean
make -C FML/Hydrodynamics/tests run USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST='1 2 4'
```

The normal module-test runner discovers this directory. Bounded fluid experiments
are separate from the C++ regression:

```sh
python3 FML/Hydrodynamics/tests/run_validation.py \
  --executable /path/to/serial/hydro_example \
  --mpi-executable /path/to/mpi/hydro_example \
  --mpirun 'mpirun' --output /fresh/output/path
```

The runner records commands, statuses and hashes and writes raw mesh tables,
histories and standalone PNG/PDF plots. NumPy/Matplotlib are needed for analysis;
SciPy is not. Use only MPI launcher flags supported by the selected vendor.
See [tests/RESULTS.md](tests/RESULTS.md) for shock/reference comparisons,
multidimensional refinement, conservation, dual-energy and rank/thread checks.

Open extensions include native gravity/SimulationCode coupling, hydrostatic and
self-gravitating/cosmological collapse calibration, broader high-Mach/shock tests,
Sedov calibration, higher-order/characteristic reconstruction, general EOS,
checkpoint/restart and performance studies. AMR, MHD, physical viscosity/cooling
and meshless/Godunov variants are separate additions.
