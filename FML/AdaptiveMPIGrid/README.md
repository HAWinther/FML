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
grid.exchange_ghosts();          // after changing data

auto nb = grid.get_face_neighbors(i, dim, +1);   // relation: BOUNDARY / COARSER / SAME / FINER
for (int j = 0; j < nb.n; j++) grid.get_data(nb.cells[j]);   // local or ghost, transparently
grid.find_leaf(std::array<double,3>{x, y, z});   // the leaf containing any point near the domain
```

## Design

- **Cells.** A cell is `(key, anchor, level)`. `anchor` holds the integer corner coordinates (LMAX bits per axis; 21 in 3D, 31 in 2D). `key` is the 64-bit Hilbert key of the corner with the low bits cleared. A cell at level l covers exactly the key range `[key, key + 2^(NDIM*(LMAX-l)))`.
- **Geometry vs. ordering.** All geometry (children, neighbours, periodic wrap) works on integer coordinates. The Hilbert key is used only for sorting, partitioning and lookups. Keys use Skilling's transpose algorithm.
- **Lookup by containment.** `find_leaf(point)` computes the key of the point and binary-searches for the leaf whose range contains it, among the local leaves or the ghosts. A neighbour that is coarser or finer is therefore found exactly like a same-level one.
- **Partition.** Every task holds `domain_start[NTasks+1]`, and the owner of a key is found by binary search. Partition cuts never fall inside a sibling family, so any family can always be coarsened, and the leaf set is **independent of the number of tasks**. The tests check this with a global hash.
- **2:1 balance** is enforced across faces, edges and corners. It uses iterative target-level propagation with ghost exchange until a global `Allreduce` reports no change.
- **Ghosts.** One layer: every remote leaf touching a local leaf by face, edge or corner, at its own level. A leaf is sent to every task whose key range intersects one of its "touching boxes" one level finer. Under 2:1 balance this set is exact. The communication pattern is cached and rebuilt after `adapt`/`load_balance`. Neighbours across a periodic boundary that are owned by the same task are local leaves, so there are no self-ghosts.
- **Coarsening** requires a complete local family, every member flagged, and that the resulting parent does not break 2:1 balance (checked conservatively against neighbour target levels).

## Tests

`example/Test.cpp` covers the following:
- Hilbert bijection, adjacency of consecutive keys, hierarchy, and full-depth round trips.
- 2D and 3D grids with periodic, non-periodic and mixed boundaries, driven by a moving shell and by random flags.
- After every `adapt`/`load_balance`: sorted and contiguous leaves, full coverage, ownership, ghost completeness, 2:1 balance, ghost values, and face-neighbour areas.

Verified identical leaf sets on 1, 2, 3, 4, 5, 7, 8, 13, 16 and 24 tasks (24 > number of initial cells, so some tasks start empty), and clean under AddressSanitizer.

```
cd example && make FML_INCLUDE=/path/to/folder/containing/FML && mpirun -np 4 ./amrtest
```

## Known limitations / next steps

- `adapt` changes each leaf by at most one level per call. Call it repeatedly to reach deep levels.
- Ghost data is exchanged with nonblocking point-to-point messages, but the API is blocking. Split it into `begin`/`end` to overlap with interior work.
- Balance and ghost-pattern construction do 4^NDIM - 2^NDIM key lookups per leaf (56 in 3D). This is fine for now; it can be restricted to cells near changes later.
- Ghost-pattern construction uses one `MPI_Alltoall` of counts (O(NTasks)). A sparse/NBX exchange would scale better at very large task counts.
- No ghost "image shift" is stored. For positions across a periodic boundary, use the minimum-image convention.
- Message counts are `int` elements of `sizeof(T)` bytes per pair of tasks.
- No OpenMP yet.
