# AdaptiveMPIGrid

An adaptively refined grid on [0,1)^NDIM stored as a **linear tree**: only the leaves are stored, sorted along a Peano-Hilbert curve. Each MPI task owns one contiguous segment of the curve.

```cpp
#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
using Grid = FML::GRID::AdaptiveMPIGrid<3, MyCellData>;   // MyCellData must be trivially copyable

Grid grid(levelmin, levelmax, {true, true, false});        // periodic per axis
for (IndexInt i = 0; i < grid.get_n_local(); i++) {
    grid.get_data(i) = ...;
    grid.set_flag(i, Grid::REFINE);                        // or COARSEN / KEEP
}
grid.adapt(prolong, restrict);   // 2:1 balance + refine/coarsen + rebuild and fill ghosts
grid.load_balance(weight);       // repartition the curve by weight + migrate + rebuild ghosts
grid.exchange_ghosts();          // after changing data (blocking)

// Overlap communication with computation
auto h = grid.begin_exchange_ghosts();               // values are packed here
for (auto i : grid.get_interior_cells()) ...;        // all neighbours are local
h.wait();                                            // (or poll with h.test())
for (auto i : grid.get_boundary_cells()) ...;        // may use ghosts
// begin_communicate_ghosts(local, ghost) does the same for any per-cell vector; several can be in flight

auto nb = grid.get_face_neighbors(i, dim, +1);   // relation: BOUNDARY / COARSER / SAME / FINER
for (int j = 0; j < nb.n; j++) grid.get_data(nb.cells[j]);   // local or ghost, transparently
grid.find_leaf(std::array<double,3>{x, y, z});   // the leaf containing any point near the domain

// Coarser grids (e.g. for multigrid): leaves finer than L replaced by their ancestor, uniform below levelmin
auto coarse = grid.truncated<double>(L);
AdaptiveMPIGridTransfer<3> transfer(grid, coarse);
transfer.restrict_average(fine_values, coarse_values);   // volume average
transfer.gather_from_coarse(coarse_values, fine_values); // each leaf gets its coarse leaf's value
```

A multigrid solver for (non-linear) PDEs on these grids is in `MultigridSolver/AMRMultiGridSolver.h`.

## Design

- **Cells.** A cell is `(key, anchor, level)`. `anchor` holds the integer corner coordinates (LMAX bits per axis; 21 in 3D, 31 in 2D). `key` is the 64-bit Hilbert key of the corner with the low bits cleared. A cell at level l covers exactly the key range `[key, key + 2^(NDIM*(LMAX-l)))`.
- **Geometry vs. ordering.** All geometry (children, neighbours, periodic wrap) works on integer coordinates. The Hilbert key is used only for sorting, partitioning and lookups. Keys use Skilling's transpose algorithm (`HilbertCurve<NDIM>`, `PeanoHilbertDomain/HilbertCurve.h`: FML's one key implementation, also used by `PeanoHilbertDomain` and `Global/Domain.h`).
- **Lookup by containment.** `find_leaf(point)` computes the key of the point and binary-searches for the leaf whose range contains it, among the local leaves or the ghosts. A neighbour that is coarser or finer is therefore found exactly like a same-level one.
- **Partition.** Every task holds `domain_start[NTasks+1]`, and the owner of a key is found by binary search. In 3D `get_domain()` gives it as an `FML::Domain` (a `PeanoHilbertDomain` with the same keys, 21 bits): particles follow the grid with `particles.set_domain(grid.get_domain())` after a `load_balance` (they are moved to the task of the leaf they are in). The reverse: `AdaptiveMPIGrid<3, T>(levelmin, levelmax, periodic, domain)` starts the grid on a given `PeanoHilbertDomain` (e.g. one built from the particle keys at `levelmin` bits; its boundaries must fall on cells at `levelmin`) instead of equal parts of the curve. Partition cuts never fall inside a sibling family, so any family can always be coarsened, and the leaf set is **independent of the number of tasks**. The tests check this with a global hash.
- **2:1 balance** is enforced across faces, edges and corners. It uses iterative target-level propagation with ghost exchange until a global `Allreduce` reports no change.
- **Ghosts.** One layer: every remote leaf touching a local leaf by face, edge or corner, at its own level. A leaf is sent to every task whose key range intersects one of its "touching boxes" one level finer. Under 2:1 balance this set is exact. A leaf is first tested in O(1) (one key): if the leaf grown by one finer cell on every side lies, without wrapping, inside an aligned box whose key range is inside this task's part of the curve, all its neighbours are local and the 56 lookups are skipped; that is most leaves (all of them on one task). The communication pattern is cached and rebuilt after `adapt`/`load_balance`. The same pass splits the local leaves into *boundary* leaves (touching a remote leaf) and *interior* leaves, so the exchange can overlap the interior work. The exchange handle is move-only and waits in its destructor if you forget to. Neighbours across a periodic boundary that are owned by the same task are local leaves, so there are no self-ghosts.
- **Irregular exchanges** (ghost-pattern setup, load-balance migration) use sparse buffers keyed by task. Each call costs only as much as the tasks it actually talks to. From `nbx_min_tasks` tasks upwards (default 64) the NBX algorithm is used (Hoefler, Siebert & Lumsdaine, PPoPP 2010): synchronous sends, probing for incoming messages, and a non-blocking barrier. That means no O(NTasks) collective. Below the threshold an `MPI_Alltoall` of counts is used instead, because NBX polls and is slower on few or oversubscribed tasks. Set `AdaptiveMPIGrid<NDIM,T>::nbx_min_tasks = 0` to always use NBX. The only O(NTasks) state left is the replicated splitter array (8 bytes per task), which is updated with an `Allgather` after load balancing.
- **Truncated grids.** `truncated<U>(L)` is the grid with every leaf finer than `L` replaced by its ancestor at `L`; for `L < levelmin` it is the uniform grid at `L`, i.e. full grids coarser than the base grid, and for `L` at or above the finest leaf it is a copy of the leaf set (same partition and local indices; the ghost pattern is copied, not rebuilt). Truncation keeps 2:1 balance. Its partition is this one with every cut rounded down to a cell at `L`, so no coarse cell is split; the few leaves between a rounded and the original cut move to the next task. `AdaptiveMPIGridTransfer<NDIM>(fine, coarse)` maps every leaf of a grid to the leaf of a coarsening that contains it (any partitions; the pattern is built once) and restricts (volume average) or gathers values along it. The cuts of a truncated grid can fall inside a sibling family, so do not `adapt()` one if the leaf set must stay independent of the number of tasks.
- **Tags.** The ghost, sparse-exchange and transfer tag counters are shared by all grids (collective calls are made in the same order on all tasks), so exchanges on different grids never share a tag.
- **Coarsening** requires a complete local family, every member flagged, and that the resulting parent does not break 2:1 balance (checked conservatively against neighbour target levels).

## Tests

`example/Test.cpp` covers the following:
- Hilbert bijection, adjacency of consecutive keys, hierarchy, and full-depth round trips.
- Overlapped exchange: the interior is computed while ghosts still hold stale values and must match the blocking result. Two exchanges are in flight at once, finished in reverse order. A deliberately wrong ordering (boundary before `wait`) makes the test fail.
- 2D and 3D grids with periodic, non-periodic and mixed boundaries, driven by a moving shell and by random flags.
- After every `adapt`/`load_balance`: the grid truncated at every level down to 2 (invariants, levels, and a hash of all of them that must not depend on the number of tasks), conservation of the volume integral by `restrict_average`, and `gather_from_coarse` giving every leaf its ancestor's value.
- After every `adapt`/`load_balance`: sorted and contiguous leaves, full coverage, ownership, ghost completeness, 2:1 balance, the boundary/interior split, ghost values, and face-neighbour areas.

Verified identical leaf sets on 1, 2, 3, 4, 5, 7, 8, 13, 16 and 24 tasks (24 > number of initial cells, so some tasks start empty), and clean under AddressSanitizer.

Both exchange paths are tested (`./amrtest dense`, `./amrtest nbx`) and give identical grids. They were also checked on 48 tasks and without MPI (`USE_MPI=false`).

```
cd example && make FML_INCLUDE=/path/to/folder/containing/FML && mpirun -np 4 ./amrtest [nbx|dense]
```

## Known limitations / next steps

- `adapt` changes each leaf by at most one level per call. Call it repeatedly to reach deep levels.
- The 2:1 balance in `adapt` does 4^NDIM - 2^NDIM key lookups per leaf (56 in 3D). The ghost-pattern build skips them for leaves with a local neighbourhood. Both could be restricted to cells near changes later.
- The NBX threshold is a guess. Tune `nbx_min_tasks` on a real cluster.
- No ghost "image shift" is stored. For positions across a periodic boundary, use the minimum-image convention.
- Message counts are `int` elements of `sizeof(T)` bytes per pair of tasks.
- No OpenMP yet.
