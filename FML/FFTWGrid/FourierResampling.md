# Fourier resampling

`resize_fourier_grid(input, nmesh, nleft=0, nright=0)` retains its existing API and normalization. It copies Fourier coefficients directly between meshes; there is no real-space interpolation or volume factor. A mode survives only when every signed component satisfies `2*abs(k) < min(input_mesh, output_mesh)`. This discards ambiguous even-mesh Nyquist planes while retaining the highest unambiguous odd-mesh modes. All other owned output coefficients are zero.

## Reusing geometry

For repeated fields, allocate the destination and build a plan once:

```cpp
FML::GRID::FFTWGrid<3> output(newmesh, 1, 1);
FML::GRID::FourierResamplingPlan<3> plan(input, output);
plan.apply(input, output);              // default maximum message: 64 MiB
plan.apply(next_field, next_output);    // same meshes and local slab geometry
```

Construction and application are collective on `FML::communicator`; all ranks must call them in the same order. Input and output must be distinct allocated grids with the constructor's meshes, local slab starts and widths. Field values, objects and halo allocations may change between calls. Applications assume input contains Fourier coefficients, set the output's status to Fourier, and leave its halo slices untouched. The existing FFTWGrid dimension/divisibility restrictions still apply; the plan supports the slab layout for dimensions N >= 2, with no transposed distribution.

The plan owns mutable scratch storage and must not be applied concurrently or reused after changing communicator/rank assignment. It stores geometry rather than references to grid objects. The 3LPT padding path reuses one crop plan for the scalar potentials and requested curl components, releasing each padded field after cropping as before.

## Routing and storage

Construction gathers both slab layouts in one exchange. Nonempty slab intervals are sorted and validated as complete partitions; zero-width slabs are ignored for ownership. Routing is determined once per retained x slice, with contiguous runs along the final stored r2c axis. Adjacent runs merge when both input and output indices are contiguous. Source and receiver independently construct matching source-ordered runs, so the wire carries only complex values, without a target index or record padding for each coefficient.

One vector maximum reduction during construction fixes the maximum payload count for every cyclic peer step. Applications reuse this schedule, skip globally empty steps, and use bounded `MPI_Sendrecv` messages. There are no per-peer count messages or per-step global round reductions. Two small per-application reductions still negotiate the message cap and validate reuse collectively. Zero-payload participants send/receive zero bytes when other edges in the step need more rounds.

`plan.apply(input, output, max_message_bytes)` takes the minimum cap over ranks, clamped to `INT_MAX`. The cap must fit at least one `ComplexType`; any fractional value remainder is unused. Counts and ceiling division stay wide, with flattened grid-index products checked against `PTRDIFF_MAX`. A message is always a whole number of coefficients and fits MPI's int byte count. The cap may change between calls without rebuilding geometry.

The plan retains two payload buffers, each no larger than one configured message and the corresponding largest peer payload. Storage is reused on subsequent calls; reducing the cap releases excess retained capacity. This cap describes retained payload storage, excluding allocator/reallocation overhead, grids and routing metadata. Routing metadata uses O(P + retained local/incident rows) space before run merging, rather than one target-index record per retained coefficient. Metadata and the grids themselves are not bounded by the message cap. There is no full outgoing payload catalogue or global grid replication.

Invalid reuse, aliasing or a cap smaller than one coefficient throws collectively before modifying output. Allocation failures and MPI failures do not have transactional rollback. Native complex-value bytes assume the same homogeneous executable/precision on every rank, as the previous record protocol did. Rebuild dependent objects after changing this header.

## Validation

`tests/test_fourier_resampling.cpp` is registered in the standard test build/run recipe. It checks every owned output coefficient against an independent signed-wave oracle, including exact excluded zeros, DC and nonzero Nyquist input. It covers 2D/3D/4D, up/down/same-size maps, odd/even combinations, changed field values through a reused plan, one-value and non-row-aligned messages, rank-dependent caps, input preservation and halo preservation. Invalid-cap, one-rank incompatible-output and aliased-grid cases verify collective rejection before mutation. Standard LPT, random-field and power-spectrum regressions exercise integration and FFT round trips.

Saved timing artifacts compare the prior per-mode record implementation, the unchanged convenience-call API using the new implementation, and an already-built plan. Reused-plan timings exclude plan construction and output allocation. They are local measurements, not a production scaling guarantee.
