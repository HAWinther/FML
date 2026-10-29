# Pair-count scheduling and workload estimates

The replicated-catalogue backend in `PairCount.h` assigns occupied cells in descending estimated-cost order, breaking equal-cost ties by ascending cell index. For multiple MPI ranks, a min-heap ordered by `(accumulated load, rank index)` selects the next owner. Equal loads select the lowest rank. Zero loads in ascending rank order form the initial heap; an assigned load only increases, so a downward root update restores it. For a given sorted cost list, this reproduces the previous linear scan's exact assignments and per-rank cell order.

## Neighbor-aware work model

For each occupied source cell, the estimated work is the number of candidate particle-distance tests plus `1 + number_of_offsets`, a small allowance for cell setup and offset traversal. Cross-catalogue candidates are `n_A(cell) * sum(n_B(neighbor))`. Auto candidates use only destination cells with index at least the source index, with `n_A * (n_A - 1) / 2` for the source cell itself. Occupancies are converted to double before multiplication to avoid integer overflow in the estimate.

The estimator and pair kernel share offset generation and neighbor traversal. They therefore use the same distance-pruned cell neighborhood, periodic wrapping, nonperiodic clipping and auto-pair ownership. Periodic neighborhoods covering the whole grid visit each destination cell once, including odd/even grid sizes. The proxy counts candidates before the exact particle-distance cut, and does not model arbitrary callback execution time.

This replaces the same-cell-only estimate, which misses dense neighboring cells of a different catalogue. Rank ownership can consequently change. The serial path retains ascending cell order and does not compute workload estimates. Particle-distance arithmetic, weights, binning and callback interfaces retain their existing definitions; MPI floating-point summation order can change with the schedule.

## Cost and storage

With `C` occupied cells, `K` offsets, `G` total cells and `P` ranks, estimation costs `O(G + C K)` per MPI rank. Sorting costs `O(C log C)` and heap assignment costs `O(P + C log P)`. The estimator visits neighbor metadata without materializing per-cell neighbor lists. Offset/cost/heap storage stays `O(K + C + P)`, in addition to existing catalogue cell lists.

This added setup work is a tradeoff for better load estimates, particularly for clustered or asymmetric cross-catalogues. It is not a guarantee of faster execution for every catalogue, rank count or callback. Single source cells remain indivisible scheduling units. Production timings should include estimation as well as counting.

## Regression coverage

`tests/test_paircount.cpp` compares candidate estimates with exhaustive independent cell-box geometry in 1D/2D/3D, including auto/cross counts, empty catalogues, boundaries and complete periodic neighborhoods. Metadata-only large-occupancy tests exercise products beyond 64-bit integer range.

A constructed disjoint cross-catalogue has all of the second catalogue's particles in cells adjacent to source cells. With two ranks, the old estimate assigns 19,000 candidate tests to the busiest rank; the neighbor estimate reduces that maximum to 10,000. An actual MPI/OpenMP counting fixture confirms all 19,200 pairs are retained and the maximum rank processes 10,000 for two or more ranks. This is a validation fixture rather than a general timing claim.

Existing exact heap-versus-scan assignment tests, brute-force weighted pair counts, angular bins, callback and survey estimators remain in the suite. `test_paircount_odr.cpp` checks multi-translation-unit inclusion. Engineering 3's saved microbenchmark measures heap assignment with a supplied cost list; it excludes this new estimation pass.
