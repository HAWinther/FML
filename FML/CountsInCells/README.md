# kNN-CDF and counts-in-cells

`SpatialStatistics.h` adds three-dimensional nearest-neighbor and spherical-cell
summaries using the existing KD-tree. It does not change KD-tree internals or
implement the deferred query-workspace/support-pruning optimizations. FFTW and
GSL are not required.

```cpp
using namespace FML::COUNTS_IN_CELLS;
FML::KDTREE::pointVec tracers; // Complete ordered tracer catalogue on each rank.
double boxsize = 1000.;
SpatialStatistics statistics(tracers, boxsize);
// Alternatively: SpatialStatistics statistics(particles, count, boxsize);

Bounds volume{{0.,0.,0.}, {boxsize,boxsize,boxsize}};
auto queries = uniform_queries(local_query_count, volume, seed, global_first_query_id);
std::vector<double> radii{5.,10.,15.,20.};
auto cdf = statistics.knn_cdf(queries, {1,2,4,8}, radii);
auto counts = statistics.counts_in_cells(queries, radii);
```

All coordinates, radii, bounds and box sizes use one consistent length unit.
Periodic points and queries are wrapped by the KD-tree. `boxsize=0` selects
ordinary nonperiodic Euclidean geometry. Positions must be finite, with absolute
coordinates at most 1e150 so squared-distance arithmetic stays representable.
This first API counts unit-weight tracers; a weighted-density PDF and weighted
factorial estimators would need different noise conventions.

## Query sample and self-exclusion

A `Query` contains its position and an optional `exclude` index (default: no
exclusion). That index is the tracer's original input index. Exactly that tracer
is excluded; coincident other tracers remain. There is no automatic removal of
the nearest or zero-distance point.

Use volume-distributed query points for the usual volume-weighted kNN-CDFs and
counts-in-cells. Queries placed at tracers instead measure a tracer-conditioned
statistic, even with self-exclusion. Results report `excluded_queries` so this
choice is not silently lost. Do not interpret a self-excluded tracer-centered
sample as random volume sampling.

`uniform_queries` uses Philox keyed by global query ID. Allocate disjoint
`[global_first_query_id, global_first_query_id+local_query_count)` ranges on MPI
ranks; changing batching or decomposition preserves the same positions.
Rectangular bounds are half open. Queries are not taken from tracer positions
implicitly, and independent positions do not imply independent cell counts:
overlapping cells and a shared density field correlate the measurements.

## Distances and CDFs

`knn_distances(queries, orders)` is a local batched operation returning arrays
`distances[order_index][local_query_index]`. An unavailable kth neighbor has
infinite distance. Positive distinct orders can be supplied in any order.

`knn_cdf` streams the query results into exact integer hit counts rather than
retaining the full distance matrix. For every requested radius,

```
F_k(r) = number of queries with distance_to_kth_neighbor <= r / number_of_queries .
```

Unavailable neighbors remain in the denominator and contribute no hit. Radii
must be finite, nonnegative and strictly increasing. Boundary inclusion is based
on squared distances consistently with the sphere counts, including zero radius.
`KnnCDF` contains the orders, radii, integer hits, fractions, query count,
self-excluded query count and box size. With zero global queries the fractions
are NaN, distinguishing an undefined measurement from a measured zero.

## Raw sphere counts and physical cell windows

`sphere_counts` returns local integer counts `[radius_index][query_index]`.
It accepts any finite nonnegative radius, counts coincident objects separately,
and returns each periodic tracer at most once. Beyond half a periodic box, this
is the count inside a minimum-image toroidal ball, not an ordinary Euclidean
sphere with volume `4*pi*r^3/3`.

`counts_in_cells` therefore requires positive radii at most half a periodic box,
so its reported spherical volume is valid. It searches once at the largest
radius per query, sorts the returned squared distances, then derives all the
smaller-radius counts. It accumulates summaries without a full query-by-radius
count matrix. Use `sphere_counts` if the individual counts themselves are needed.

For a nonperiodic catalogue, supply a rectangular observation window:

```cpp
Options options;
options.window = Bounds{{xmin,ymin,zmin}, {xmax,ymax,zmax}};
// Use centers at least max(radii) from every boundary.
auto counts = statistics.counts_in_cells(queries, radii, options);
```

All cells must lie completely inside that window and all tracers must be inside
it. This preserves a common query sample across radii. It does not infer an
observation mask from the minimum/maximum tracer positions. Arbitrary survey
masks/holes need a separate full-cell selection function provided by the caller;
a rectangular bounding box alone is not a mask correction.

`allow_truncated_cells=true` explicitly permits uncorrected clipped cells and is
recorded as `uncorrected_clipping`; no boundary-volume correction is then applied.
An explicit number density or window is still needed for normalization. Raw kNN
CDFs also make no nonperiodic selection/boundary correction.

## PDFs, moments and shot-noise convention

Each `CellStatistics` contains:

- The radius, volume, query count and reference number density.
- Integer occupancy histogram and discrete probabilities for N=0 through
  `histogram_limit`, plus a final **N>limit** overflow bin. The default limit is
  128. Tail objects are retained in all moments; the histogram never clips counts.
- Population mean and variance of N, accumulated with stable Welford/Chan
  updates. Variance divides by the number of queries, consistently with PDF
  moments; it is not an N-1 uncertainty estimator.
- Falling factorial moments `F_q = <N(N-1)...(N-q+1)>` and `F_q/<N>^q`, including
  order zero. The default maximum order is four; supported orders are zero to 16.
- `poisson_subtracted_variance = Var(N)-<N>` and
  `normalized_connected_variance = (Var(N)-<N>)/<N>^2`.

Under a Poisson-sampling model, factorial moments remove self-counting terms and
estimate moments of the integrated continuous intensity. This does not prove
that an HOD galaxy process has universal Poisson noise. The finite-sample ratios
are not advertised as unbiased estimators, and a noisy shot-subtracted variance
can be negative. No deconvolution of Poisson sampling is applied to the observed
count PDF.

Density contrast is `delta = N/(nbar*cell_volume)-1`. Set
`Options::number_density` explicitly, or use the default full catalogue count
divided by periodic-box/observation-window volume. `delta_edges` controls its
histogram. `density_probability` is probability mass per bin;
`density_pdf` divides by the bin width. Underflow/overflow counts are separate
and normalization includes all query cells. Zero reference expected count gives
an undefined density normalization and is counted explicitly in
`density_undefined`. Zero mean makes normalized positive-order factorial moments
and connected variance undefined (NaN). An empty tracer catalogue still has
P(N=0)=1 when query cells exist. Zero query samples instead give undefined means,
PDFs and moments.

The point-process identity is also useful for cross-checks:
`P(N=0)=1-F_1`, and `P(N=j)=F_j-F_(j+1)` for j>=1, using the same queries,
self-exclusion and inclusive-radius convention.

## MPI and OpenMP ownership

The KD-tree is local and the **complete ordered tracer catalogue is replicated**
on every participating rank. Queries, not tracers, are distributed. There are no
implicit tracer gathers or distributed-neighbor searches. A 64-bit consistency
checksum detects common accidental catalogue/schema mismatches before array
reductions; callers must still honor the replicated-catalogue contract. This is
not the future distributed-catalogue backend.

CDF/count summaries default to `Reduction::Global`: all ranks must call with the
same radii/options, even ranks without queries. Query ranges must not overlap
unless repeated samples are deliberately intended. Results are returned on all
ranks. `Reduction::Local` avoids MPI collectives and allows independent local
catalogues. Raw distances/counts and query generation are always local.

Run summary calls outside OpenMP regions. Query loops are parallelized internally
with thread-local histograms/moments. Integer hit/PDF counts are exact across
thread/rank reductions; floating moment merges can differ by roundoff. Global
counts and histogram reductions use wide unsigned counters; vector reductions
are chunked to MPI's int-count limit.

Memory consists of the existing KD-tree, the caller's queries, per-thread largest
neighbor lists, and per-thread small summary arrays. CDFs do not retain all
query distances; count summaries do not retain all individual counts. The raw
batch APIs deliberately allocate their requested output matrices.

## Tests and example

```
make -C FML/CountsInCells/tests run
make -C FML/HOD/example
mpirun -np 4 FML/HOD/example/mock_statistics
```

Tests compare against brute force, verify duplicate/self-exclusion behavior,
nonperiodic windows, empty data/query samples, CDF-to-count identities, density
PDF tails, stable variance and factorial moments, analytic uniform-binomial
limits, and rank/thread/schema behavior. The HOD example explicitly replicates
its small mock catalogue before constructing statistics; that replication is
visible in the example rather than hidden inside a statistic call.

Reference: [Banerjee & Abel, nearest-neighbor distributions](https://arxiv.org/abs/2007.13342).
