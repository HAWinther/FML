# Glass particle loads and cosmological ICs

`GlassIC.h` supplies a three-dimensional periodic, equal-mass glass generator and
LPT sampling at irregular particle positions. `LoadIO.h` saves unperturbed loads
in a versioned, portable sharded format and reloads with a different MPI task count.
C++17 and FFTW are required; GSL is unnecessary. Include paths and linked `Global.cpp`
follow [the shared build guide](../BUILDING.md).

```cpp
#include <FML/GlassIC/GlassIC.h>
#include <FML/GlassIC/LoadIO.h>
using namespace FML::GLASSIC;
auto load = poisson_load(32768, 12345, 250.0); // Count, seed, box size in Mpc/h.
Options options;
options.mesh = 96;
options.max_steps = 3000;
auto result = relax(load, options);
if (result.status != Status::Converged) {
    // Inspect history, adjust resolution/softening/timestep or continue relaxation.
}
save_load(load, "new_glass_prefix");          // All ranks; new prefix only.
auto reused = load_load("new_glass_prefix");
```

A converged force residual alone is insufficient evidence for a scientifically
suitable load. Measure discreteness power, directional dependence, close-pair
statistics and resolution/relaxation sensitivity at the intended particle count.
Default tolerances are starting choices, not universal glass-quality thresholds.
The example reports an exhausted run explicitly and demonstrates how to apply ICs;
callers should set their own acceptance criteria before production use.

## Repulsive force and artificial relaxation time

Positions and Lagrangian positions `q` use unit-box coordinates `[0,1)`; every
particle has equal mass `1/N` in the relaxation solve. `boxsize` is physical
comoving Mpc/h metadata and does not alter the dimensionless glass calculation.
The periodic zero-mean Poisson equation is `laplacian Phi = delta`. The attractive
acceleration is `-grad Phi`; the generator reverses it exactly once, for **both**
the long- and short-range components. There is no cosmological background evolution
in this relaxation: artificial time and the relaxation velocities have no physical
units or interpretation as cosmological peculiar velocities.

The long-range solve uses CIC deposition, normalized FFTW transforms, the
[existing Gaussian split](../ShortRangeGravity/README.md), a continuous gradient
with even-mesh odd-derivative Nyquist components zeroed, and compensation for both
CIC assignment and CIC force-interpolation windows. The short-range solve reuses
the softened spline kernel and cell-list backend. Its coupling is `1/(4 pi N)`
for unit source masses and unit-box coordinates. Split scale is `split_cells/mesh`;
compact softening **support** is `softening_spacing/N^(1/3)` (not a Plummer epsilon).
The omitted pair-force tolerance is `cutoff_tolerance`; the implied cutoff must be
less than half the box and at least as large as the softening support. These are
approximated periodic P3M forces, with finite mesh anisotropy and interpolation
errors. Reversing only a mesh force would leave unresolved close pairs unconstrained.

Each step holds the acceleration fixed for an exact damped kick,

```
v_next = exp(-damping*dt) v + (1-exp(-damping*dt))/damping * a_repulsive
x_next = wrap(x + dt*v_next)
```

A single global velocity multiplier bounds the largest displacement by
`max_step_spacing` times the mean spacing. This is a first-order dissipative
relaxation method; reduce the timestep and check for timestep-dependent stalls.
Continue by calling `relax` again on the same unperturbed load. Forces, ghosts and
cells rebuild after each drift. The four PM grids are reused between steps.

`Result` distinguishes `Converged`, `MaximumSteps` and `NonFinite`; it reports
completed drifts, final residuals and a history including the initial state.
RMS and maximum **vector norms** are in unit-box coordinates. Convergence requires
both force RMS and velocity RMS below their tolerance times mean spacing for
`stable_steps` consecutive measured states after `min_steps`; maxima are diagnostic,
not additional stopping thresholds. `max_steps=0` evaluates the unchanged load.
A nonfinite proposal is rejected before modifying positions/velocities. Invalid
inputs/settings throw collectively before the corresponding numerical operation.

## Distributed ownership, memory and reproducibility

Every entry point is collective on `FML::communicator`. Call in the same order on
all ranks. The FFT slabs must be nonempty and match `FML::xmin_domain/xmax_domain`;
use a mesh divisible by the task count with the standard equal-width domains.
One halo on each side is required. Empty **particle** slabs are supported.
Settings and IO prefixes must agree across ranks. Catastrophic allocation failures
are subject to the same MPI failure limitations as existing FML containers.

Poisson positions use Philox streams keyed by seed and global ID, so each ID starts
at exactly the same coordinates regardless of task or thread count. Initial ID
ranges are distributed before migration. Particle receive capacities are computed
from actual destination counts before calling `MPIParticles::communicate_particles`,
avoiding speculative global-sized allocations. Local particles and ghost sources
are ordered by ID. Floating-point deposition/FFT reduction order can still differ;
relaxed coordinates agree to numerical tolerances, not bitwise across all builds.

Storage scales with local mesh, owned particles, cutoff ghosts and local force
outputs; there is no particle-payload all-gather in generation/relaxation/IO/ICs.
Small settings/count arrays are replicated. A large cutoff or narrow slabs may
still require most sources as ghosts. Each rebuild also holds owned short-range
records plus ghost sources and cell metadata. History costs O(number of steps).

Generated IDs are exactly `[0,N)`. Keep them unique when adapting a custom `Load`.
Position mutation through the public particle container requires wrapping and
ownership migration before calling the numerical entry points. The module does
not change existing SimulationCode particle classes or parameters.

## Load format and readback

`save_load` writes `prefix.rankR.glass` shards with
`FML_GLASS_SHARD 1 local_count`, then decimal `ID x y z` records at 17-digit
precision. `prefix.manifest` contains `FML_GLASS 1 shard_count total_count seed
boxsize`. Relaxation velocities are deliberately excluded; reloaded velocities
are zero and `q=x`. This is a reusable unperturbed load, not a relaxation checkpoint
or a perturbed cosmological snapshot. Perturbed loads cannot be saved through this API.

All ranks validate IDs without collecting a full global ID array: a temporary
validation copy routes IDs to deterministic owner ranks, then checks local
uniqueness. Count and range checks require a permutation of `[0,N)`.
Each shard has one reader, which streams bounded `batch_records` chunks to current
owners. Changing the rank count therefore needs no root-owned full catalogue.
Text IO prioritizes portability and correctness over throughput.

Use a new prefix per save. Shards are closed and renamed from temporary files;
the manifest is published only after every shard succeeds. A failed write can
leave orphan shards or temporary files, which must be removed before retrying that
prefix. This protects existing loads but is not a transactional multi-file store.
Readback rejects missing/truncated/extra records, unsupported versions, invalid
coordinates, duplicate IDs and manifest/count disagreements. Files must be visible
through a shared filesystem with identical prefixes on all ranks. Files are load
inputs, never executable configuration or instructions.

## Cosmological perturbations at the glass positions

Two interfaces use the glass as the unperturbed Lagrangian load:

* `apply_fields(load, displacement, velocity)` samples two precombined real-space
  vector fields by CIC at the same original glass coordinates. Displacement is in
  box lengths; velocity has the units supplied by the caller. This supports
  independently prepared 1/2/3LPT fields, scale-dependent growth or transverse
  fields without imposing a new growth prescription.
* `apply_3lpt(load, delta_fourier, dlogDdt, dealiasing)` reuses FML's existing
  EdS-factorized 3LPT construction including its transverse term. Input linear
  density is already at the IC epoch. The returned velocity field is
  `dlogDdt*(Psi1 + 2*Psi2 + 3*Psi3)`. Passing `a H f L` gives physical peculiar
  velocities when H and L have compatible units. EdS is exact for this time
  factorization; other cosmologies require explicitly justified approximations
  or separately prepared fields. Padding follows [the LPT conventions](../LPT/README.md).

Both retain original `q`, preserve IDs, replace artificial relaxation velocities,
wrap displaced positions and migrate once after sampling both fields. Applying
ICs twice, or relaxing an already perturbed load, is rejected. Access the result
through `load.particles` and adapt it to a simulation particle type explicitly.

`density_from_provider(grid, provider, species, a, boxsize, field_seed, fixed)`
bridges the shared [transfer-provider API](../InitialConditions/README.md) to the
existing Philox random-field generator. Provider power uses h/Mpc and (Mpc/h)^3;
the grid uses inverse-box wavevectors, so the adapter queries `k_grid/L` and divides
power by `L^3`. It checks IC gauge, species/time/k coverage and finite nonnegative
power before parallel mode generation; DC and even-mesh Nyquist planes are zero.
A radial lookup is O(mesh^2) temporary storage and queries integer squared radii
up to the retained Fourier corner. Spectrum/provider calls run serially outside
OpenMP. Supply the same provider on all ranks, one shared adiabatic realization,
and the required species/growth prescription. The adapter does not reinterpret
provider Newtonian velocity-divergence transfers as density growth rates.

The example's analytic toy provider is solely a units/API demonstration. Real
inputs may come from `TransferProviders` or `InternalTransferProvider`. No new
regular-grid external-IC inversion assumption is introduced. Multiple-species
quiet-load generation and automatic SimulationCode configuration are follow-ups.

Measured results and limitations are recorded in [tests/RESULTS.md](tests/RESULTS.md).

## Building and validation

```
make -C FML/GlassIC/tests run
make -C FML/GlassIC/example
# Serial:
make -C FML/GlassIC/tests USE_MPI=false USE_OMP=false run
```

Tests independently check periodic Ewald forces, mesh refinement, softened-pair
repulsion, zero self force, deterministic IDs, measured low-k suppression and
force decay, explicit stopping statuses including nonfinite overflow, corrupted IO and duplicate-ID rejection,
nearest-neighbour directional/close-pair diagnostics, fixed-split force-mesh refinement,
glass/lattice recovery of the same imposed linear mode, CIC sampling against a direct
nodal interpolation, exact plane-parallel LPT and
provider normalization/gauge/domain failures. IO can additionally be checked with
`test_glass_ic --write prefix` under one task count and `--read prefix` under another.
The production algorithms never gather all particles; only small tests gather
particles to evaluate independent direct references.

Quality remains resolution- and force-dependent. Small-box suppression checks do
not establish an asymptotic k^4 law or isotropy at production volume, and glass
loads do not eliminate all discreteness artifacts. See
[Wang & White (2007)](https://arxiv.org/abs/astro-ph/0702575) and
[Garrison et al. (2016)](https://doi.org/10.1093/mnras/stw1594).
