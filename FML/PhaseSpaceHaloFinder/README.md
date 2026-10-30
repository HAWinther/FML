# Single-snapshot phase-space halo finder

An independent C++17 implementation in `FML::PHASE_SPACE_HALOS`. It finds spatial FoF parents, resolves phase-space peaks, assigns particles hierarchically, removes unbound particles and returns host/subhalo candidates with stable particle-ID memberships. It supports serial/OpenMP and periodic slab MPI without an all-gather of particle payloads. It does not modify the input particles or integrate into COLASolver automatically.

This is a **first implementation, not a drop-in ROCKSTAR replacement or a calibrated production mass function**. A controlled external Plummer comparison is available; cosmological catalogue comparisons and simulation completeness/purity calibration remain necessary. See [tests/RESULTS.md](tests/RESULTS.md) for the actual validation and performance measurements.

## Algorithm and definitions

The spatial FoF linking radius is `linking_length * boxsize / cbrt(global_particle_count)`. The default linking parameter is .28. Links use minimum-image distances and exact connected components; complete parent groups are processed independently. Equal-resolution sampling is assumed for linking: varying masses enter properties and binding energies, but do not change the number-density linking radius.

Each parent is normalized by its mass-weighted 3D position and peculiar-velocity RMS dispersions. The scalar dispersions preserve rotational invariance; zero dispersion is regularized by the softening/velocity floors. The nearest-neighbour-distance quantile (`retained_fraction=.7`) sets a new six-dimensional FoF radius. Refinement repeats in each retained child until the minimum core size, a degenerate unsplittable group, or the depth limit is reached. The latter is counted in `statistics.depth_limited`; it means refinement stopped at the configured resolution limit.

Leaf groups seed haloes. Core position/velocity Poisson uncertainties, using the mass-weighted effective particle count, identify duplicate seeds: each seed merges into at most one larger seed (the most consistent one, measured with its own uncertainties). A symmetric, transitive union of all consistent pairs let a small, poorly measured host fragment lying between a host and a subhalo merge the subhalo into the host. An additional, implementation-specific kNN density-saddle test suppresses weaker peaks with no resolved dip toward their nearest denser peak; each weaker seed attaches to at most one denser seed, reducing the risk of a low-density bridge merging two genuine peaks transitively. Density uses 6D neighbour radius^-6, at least 20 neighbours when available, and samples along the connecting phase-space line. `minimum_peak_contrast=2` and `saddle_samples=8` are defaults; contrast=1 disables this extra check. These controls need calibration on real simulations.

Particles are assigned on the way back up the hierarchy, using core positions/velocities and a dynamical position scale from the seed circular velocity and the reference density. `assignment_density=0` resolves to 200 times the global mean mass density; alternatively supply the desired mass density in coordinate units. Fixed seed metrics within each assignment pass prevent dependence on particle traversal order. Halo identifiers are minimum seed-core particle IDs; they are deterministic for an input snapshot but are not persistent identities across epochs.

The normalization, adaptive nearest-neighbour linking and hierarchical seed-assignment approach is inspired by [Behroozi, Wechsler & Wu, sections 3.2–3.3](https://arxiv.org/abs/1110.4372). This implementation adds a sampled density-saddle check and uses its own unbinding, boundary definitions and MPI implementation. It does not implement ROCKSTAR's temporal corrections or claim equivalence to that catalogue.

## Gravitational binding and properties

Unbinding uses the **full isolated Newtonian potential**, not a truncated PM/TreePM correction. `BindingPotentialTree` uses a monopole binary tree, excludes self potentials and opens nodes intersecting the softening support. The default opening angle is .7 (`potential_opening_angle`; .3 changed 26 of 5686 objects of a cosmological snapshot by at most 1.7% of their particles at 1.15x the cost, tests/RESULTS.md); opening angle zero gives exact leaf sums. The compact spline Green function is shared with `ShortRangeGravity/Softening.h`; the input length is its support h, not Plummer-equivalent epsilon. The tree can be tested independently against direct sums.

At each iteration the bulk velocity is recomputed. Potentials are initially evaluated with the tree;
subsequent passes subtract the removed particles directly when that costs fewer interactions than the last tree
walk, and otherwise rebuild the tree. A particle remains if

```
0.5 * |v_pec - v_bulk + hubble_velocity_gradient * (x-x_bulk)|^2 + Phi_self < 0.
Phi_self = - (G/position_to_physical) * sum_other mass * spline_green(distance,softening_support).
```

Direct removal corrections applied to approximate monopoles retain the original approximation error.
For extreme unequal-mass remnants this can affect binding classification; use `potential_opening_angle=0`
when exact potentials are required. Rebuilding approximate potentials after every removal was rolled back after
simulation testing found unchanged ROCKSTAR comparison results but about 60% greater runtime (see `tests/RESULTS.md`).

The Hubble term is optional and enters binding only, not the phase-space clustering metric. The potential is that of the assigned halo, with zero at infinity. Host tidal fields, a tidal-radius solver, combined-system major-merger unbinding and modified-gravity binding are not implemented. This is an isolated self-bound convention. Objects below the minimum particle count or minimum retained **mass fraction** are discarded. Failure to reach a stable membership by `max_unbinding_iterations` throws, rather than returning a falsely converged bound object.

Output fields:

- `particle_ids`: sorted, **exclusive** surviving memberships. No particle appears in two objects. An unassigned particle still has an `Assignment` with `halo=no_halo`; its spatial FoF ID is retained.
- `bound_mass`: mass of that exclusive membership. `inclusive_bound_mass`: own bound mass plus all host descendants, counted once.
- `position`: shrinking-core centre chosen to minimize the estimated Poisson position error, wrapped into the box; `velocity`: corresponding core peculiar velocity. `bulk_velocity` and per-axis `velocity_dispersion` use all surviving members.
- `bound_radius`: farthest exclusive member from the final core centre. **It is not R200, a virial radius or an SO radius.**
- `vmax`, `rmax`: maximum of sqrt[G M(<r)/(position_to_physical*r)] and its coordinate radius. Default (`vmax_inclusive = true`, as ROCKSTAR): the bound members of the object and all its descendants, radii below the softening support set to the softening support, and only radii inside which the mean enclosed density exceeds the assignment density. `vmax_inclusive = false`: the exclusive members, radii below the softening support omitted. Equal-radius shells are accumulated together; zero Vmax/Rmax means no resolved radial estimate. On a cosmological snapshot the exclusive estimate was 6% below ROCKSTAR for hosts above 1000 particles, the default 2.4% (tests/RESULTS.md).
- `host`: the nearest normalized-distance, more-massive candidate in the same FoF parent whose **bound extent** contains this centre. Equal masses use the smaller ID. This deterministic containment convention is not an SO/virial subhalo definition and is ambiguous in major mergers. Strict mass/ID ordering prevents cycles.
- `bound_fraction`: surviving mass divided by initially assigned mass. `binding_performed` distinguishes the default bound catalogue from a run with `unbind=false`, whose mass fields are assigned masses without a binding guarantee.

`Catalogue.options`, `resolved_assignment_density` and `spatial_linking_radius` record the calculation settings. Statistics report refinements, seed/saddle merges, depth-limited leaves, energy removals, potential interaction visits and MPI traffic/work. They are rank-local in MPI mode except the label iteration count; reduce explicitly when aggregating.

## Units and input API

All positions, boxsize, softening and radii share one coordinate-length unit. Mass has one caller-supplied unit; velocities have one peculiar-velocity unit. G must have dimensions velocity^2 * physical-length / mass. Use `position_to_physical=1` for physical coordinates and a for comoving coordinates. No canonical-momentum, scale-factor or Hubble conversion is inferred.

```cpp
#include <FML/PhaseSpaceHaloFinder/PhaseSpaceHaloFinder.h>
using namespace FML::PHASE_SPACE_HALOS;
std::vector<Particle> particles; // Fill globally unique uint64 IDs, position, peculiar velocity and positive mass.
Options options;
options.boxsize = 1;
options.softening_support = .001;
options.gravitational_constant = 1; // A dimensionless example, not a cosmological unit conversion.
auto catalogue = find(particles, options);
```

For comoving Mpc/h positions, Msun/h masses and km/s peculiar velocities:

```cpp
#include <FML/Units/Units.h>
namespace SI = FML::UTILS::SI;
options.boxsize = boxsize_mpch;
options.softening_support = softening_support_mpch;
options.position_to_physical = a;
options.gravitational_constant = SI::G * SI::Msun / (SI::Mpc * std::pow(SI::km / SI::s, 2));
// Optional expansion velocity across the halo, if H(a)=100*h*E(a) km/s/Mpc:
options.hubble_velocity_gradient = 100 * a * E_of_a;
```

The factors of h cancel in G for these mass/length units. The phase-space particle import helper in `Adapters.h` accepts FML particles with `get_ndim`, `get_pos`, `get_vel`, `get_id` and optionally `get_mass`, with explicit position/velocity/mass multipliers. Convert COLA residual velocities/canonical momenta to full peculiar velocities before calling the finder.

## Distributed complete-parent analysis

```cpp
#include <FML/PhaseSpaceHaloFinder/Distributed.h>
auto catalogue = find_distributed(owned_particles, options);
```

This is collective on `FML::communicator`, including empty ranks. Input must already occupy FML's periodic x slabs (`xmin_domain`, `xmax_domain` in unit-box coordinates). Positions still use `options.boxsize`. Options must agree across ranks. IDs are globally checked by hashing IDs to validation owners; particle payloads are not gathered for this check.

Owned particles plus link-radius ghosts form local FoF edges. Group minima propagate through subscribed ID/label metadata until a global fixed point; this stitches chains crossing any number of slabs, including cutoffs wider than a slab. The global minimum particle ID identifies the parent. Complete parent members are then routed to a deterministic hash owner for analysis, and final particle assignments return to their original input ranks. Halo catalogues remain on parent owners, which can be different from spatial centre owners. No rank holds a replicated global particle catalogue. The largest parent is still analyzed on one rank, and hash assignment balances groups statistically, not their actual computational work.

`max_local_particles` bounds per-rank input/received working record counts, `max_parent_particles` bounds a complete parent, `max_seeds` limits candidate proliferation and `max_neighbor_visits` limits radius-query visits per spatial search/refinement. These are not total RSS limits: copies, tree metadata, IDs, member indices and staged messages consume additional memory. `max_message_bytes` bounds each transport payload, with checked whole-record chunking. Option/input/duplicate-ID/label-convergence/parent-analysis failures are collective. Memory exhaustion outside reserved receive transport is not a recoverable distributed checkpoint.

A parent must be compact enough to unwrap into less than half a box along every axis. Percolating parents that fail that test throw; they are not silently split or analyzed with a broken metric. This is an initial compact-halo restriction. Label iteration limits also fail explicitly. The caller retains ownership and ordering of its original particles.

## Catalogue adapters

`Adapters.h` provides:

- `so_seeds`: IDs, centres and initial radii in **unit-box coordinates**, with no invented SO mass. `so_seeds_by_slab` collectively routes these guesses from parent owners to the spatial ranks needed by the existing SO routine. Measure SO quantities against the full particle distribution; the existing SO search/buffer limitations still apply.
- `hod_halo(halo, mass, radius, concentration)`: the caller supplies the intended mass/radius definition explicitly, e.g. independently measured M200m/R200m. It returns an HOD halo with core velocity and isotropized dispersion. Particle-profile HOD members must be supplied separately.
- `history_snapshot`: an exclusive particle-ID snapshot for `HaloHistory`, with explicit units, mass-definition and ID-namespace metadata. It validates that module's requirements, including positive radii. A distributed result produces a rank-local snapshot; assembling catalogue histories across different parent-owner assignments remains a separate routing step. Temporal history matching does not automatically stabilize this finder's seed or host choices.

## Build, run and validate

```
make -C FML/PhaseSpaceHaloFinder/tests run USE_MPI=false
make -C FML/PhaseSpaceHaloFinder/tests run USE_MPI=true NP_LIST="1 2 4"
OMP_NUM_THREADS=4 FML/PhaseSpaceHaloFinder/tests/test_phase_space_halos --benchmark
```

Tests are also registered in `FML/tests`. The module requires only Global plus header dependencies; FFTW/GSL are not required for standalone builds. Change compiler/MPI options in separate build directories or clean first.

The optional comparison harness has separate regression checks (NumPy and SciPy required, no ROCKSTAR binary):
`make -C FML/PhaseSpaceHaloFinder/tests run-python`. They exercise uint64 ID round trips and sparse membership
joins, including IDs above 2^53 and 2^63. IDs are never converted to floating point or used as dense array indices.

The text-input example reads `input_prefix.rank.txt` rows `id x y z vx vy vz mass`, already distributed to their owning slabs, and writes parent-owner halo catalogues plus input-owner particle assignments:

```
make -C FML/PhaseSpaceHaloFinder/example USE_MPI=false
FML/PhaseSpaceHaloFinder/example/phase_halos input_prefix output_prefix boxsize softening_support G position_to_physical [hubble_velocity_gradient]
```

For MPI, compile with `USE_MPI=true` and launch the same command with mpirun. Every rank needs its own input file, even if empty. Output headers describe bound rather than SO masses/radii; the no-halo sentinel is the largest uint64 value.

Validation covers analytic pair binding and circular velocity, direct/tree potentials, overlapping velocity populations, a smooth host plus cold subhalo and energetic debris, sampled Plummer spheres at two resolutions, permutation/rotation/boost/unit invariance, periodic and multi-slab MPI stitching, chunked messages, empty ranks and explicit failure paths. Further work includes external ROCKSTAR catalogue comparisons, calibrated completeness/purity over mergers and stripping, alternative host boundaries, temporal consistency, percolating-parent support, large-parent load balance and production memory/performance studies.

### Optional external ROCKSTAR check

Build the serial `rockstar_fixture` target, then run the comparison against an existing external ROCKSTAR binary in a dedicated output directory:

```
make -C FML/PhaseSpaceHaloFinder/tests rockstar_fixture USE_MPI=false
python3 FML/PhaseSpaceHaloFinder/tests/compare_rockstar.py /path/to/rockstar /tmp/new-reference-run --reference-label reference-revision
```

The fixture has 4000 Plummer particles in comoving Mpc/h, Msun/h and km/s, with a=1 and Omega_m=.3. Both runs use b=.28 and an input force/softening scale of .0005. The single-process reference is nonperiodic; this isolated cloud stays away from box boundaries. The harness checks one recovered object, Vmax within 5% and centre separation below .005 Mpc/h. It reports the mass difference without asserting equal definitions: our mass is exclusive isolated-bound membership, while the reference field is bound M200b. Softening, centres and binding conventions also differ. The reference is downloaded/built separately; no ROCKSTAR source is bundled or required by the library/tests.
