# Peano-Hilbert MPI domain decomposition

This folder is a standalone module for mapping 3D periodic positions to
Peano-Hilbert keys, dividing the key range among MPI ranks, and storing a
distributed 3D grid. It is not connected to `MPIParticles`, `SimulationCode`,
or particle migration.

The implementation has two layers:

1. `key_from_position` and `key_from_grid_coordinate` map coordinates to a
   64-bit Hilbert key using at most 21 bits per coordinate.
2. `Decomposition::build` reduces a distributed prefix histogram and chooses
   approximately equal-particle contiguous key ranges. `owner_of_key` and
   `owner_of_position` map a key or position to its rank.
3. `DistributedGrid3D<T>` stores the grid values for cells in the local key
   interval. Its compact local storage follows Hilbert-key order, so a rank's
   cells do not need to form a rectangular block.

## API

```cpp
#include <FML/PeanoHilbertDecomposition/DistributedGrid3D.h>

using namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;
std::vector<Key> local_keys;
// Fill local_keys with key_from_position({x, y, z}, 21) for local particles.
#ifdef USE_MPI
auto domains = Decomposition::build(local_keys, MPI_COMM_WORLD, 21, 16);
#else
auto domains = Decomposition::build(local_keys, 0, 21, 16);
#endif
int owner = domains.owner_of_key(local_keys.front());
Key begin = domains.key_begin(domains.rank());
Key end = domains.key_end(domains.rank()); // half-open [begin, end)
DistributedGrid3D<double> grid(1U << 21, domains, 0.0);
grid.initialize([](const Position & center) { return center[0]; });
// Local-order access: grid[i], grid.local_key(i), grid.local_coordinate(i).
// Global-coordinate access works for local cells or cached remote cells:
double value = grid.at(i, j, k);

grid.build_halo(1, HaloConnectivity::Full, true); // local setup
grid.exchange_halo();                             // collective refresh
auto values = grid.fetch_remote_cells(requested_cells); // collective arbitrary reads
```

`bits` controls the integer grid resolution. `histogram_bits` controls the
prefix histogram size (`2^histogram_bits` 64-bit counters per rank, capped at
20 bits). Cuts align to histogram-bin edges, choosing the nearer edge of each
target quantile bin. Increase `histogram_bits` to improve cut resolution at the
cost of memory and reduction bandwidth. Identical keys always have the same
owner, so a highly clustered set of identical positions cannot be split to
force exact balance. Empty rank intervals are supported.

`DistributedGrid3D<T>` requires `N = 2^bits`, where `bits` is the decomposition
resolution. Then every Hilbert key maps to exactly one grid cell. `local_values`
and `operator[]` access rank-local storage; `local_key` and `local_coordinate`
recover the matching global cell. `owner(i,j,k)` is a non-communicating
ownership query. `initialize` calls its callback only for local cell centers.

`fetch_remote_cells` is collective: all ranks call it, each with its own list
of coordinates. It returns values in query order and caches remote results for
`at(i,j,k)`. `build_halo` creates a unique list of off-rank cells for a cube
stencil (`Full`, including edges and corners) or face-only stencil (`Faces`),
with periodic wrapping on by default. Call `exchange_halo` collectively after
owners update local values to refresh these cached ghosts. Remote writes modify
only the local cache; they do not update the owning rank. MPI payloads currently
must fit in legacy `MPI_Alltoallv` integer counts and displacements; oversized
exchanges fail collectively before data movement.

Positions must be finite and already wrapped into `[0,1)` on all three axes.
The decomposition object records all key boundaries on each rank. The grid is
distributed-memory storage, not an MPI shared-memory allocation. Halo lists
are built from geometric cell coordinates, so they work when a rank's Hilbert
region is non-rectangular.

Build the small example with `make` for MPI or `make USE_MPI=false` for serial.
The caller initializes MPI before calling the collective builder. Running
`mpirun -np 4 ./peano_hilbert_example concentrated` creates empty particle and
grid ranks to exercise zero-count collectives and remote reads from empty ranks.

Run the regression suite with `make -C tests run MPIRUN=mpirun NP_LIST='1 2 4'`.
It checks exhaustive low-resolution key bijection/inversion and curve adjacency,
rank key intervals, arbitrary remote reads, face/full halo geometry at widths
one and two, periodic and nonperiodic boundaries, refreshed values, and empty
ranks. Use `make -C tests run USE_MPI=false` for the serial suite.

## References

- Hilbert key transform: J. Skilling, “Programming the Hilbert curve,” *AIP
  Conference Proceedings* **707**, 381–387 (2004),
  [doi:10.1063/1.1751381](https://doi.org/10.1063/1.1751381).
- Use of Peano-Hilbert ordering for cosmological N-body parallelization:
  V. Springel, “The cosmological simulation code GADGET-2,” *MNRAS* **364**,
  1105–1134 (2005), [doi:10.1111/j.1365-2966.2005.09655.x](https://doi.org/10.1111/j.1365-2966.2005.09655.x).

The prefix-histogram partitioning, grid storage and halo exchange are FML's own
implementation; this citation does not imply integration with particle evolution.
