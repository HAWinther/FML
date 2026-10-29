# Deterministic pair-count rank assignment

The replicated-catalogue backend in `PairCount.h` assigns occupied cells in descending estimated-cost order, breaking equal-cost ties by ascending cell index. For multiple MPI ranks it now uses a min-heap ordered by `(accumulated load, rank index)` to select the next owner. Equal loads therefore select the lowest rank, matching the previous `std::min_element` scan. Zero loads in ascending rank order form the initial heap. After an assignment, the selected load can only increase, so a single downward root update restores the heap. Each selected rank receives the same cells in the same order and accumulates its load with the same double additions.

For `C` occupied cells and `P` ranks, the assignment step changes from `O(C P)` to `O(P + C log P)`, including linear heap construction. Heap metadata is `O(P)` and stores one `(double, int)` node per rank. The existing cost list and its `O(C log C)` sorting remain. This is a scheduling optimization; it does not change pair enumeration, the replicated catalogue backend or the cost estimate.

The single-rank path retains ascending cell order for locality. Existing auto/cross counts, periodic/nonperiodic distances, weights and generic pair callbacks use their existing interfaces. The cost estimator still uses same-cell occupancy; neighboring candidate occupancy is the separate Engineering item 4.

`tests/test_paircount.cpp` compares the exact per-rank cell lists against an independent implementation of the old scan, including empty schedules, idle ranks, equal/zero costs, skewed and rounded loads, and deterministic randomized costs. Existing brute-force weighted counts in 1D/2D/3D, angular bins, callbacks and survey estimators cover integration. `test_paircount_odr.cpp` also checks inclusion from multiple translation units.

The heap targets assignment scaling at larger rank counts. At small rank counts its bookkeeping can cost more than a short linear scan; this does not imply an end-to-end speedup for every catalogue or MPI size. Focused local benchmark timings and reproducible source are saved with this change's review artifacts.
