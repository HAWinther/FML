# Density-split and marked clustering

`EnvironmentClustering.h` adds a first periodic-box implementation of backlog item 31. It reuses `CountsInCells::SpatialStatistics` for spherical top-hat counts and the existing cell-list pair counter for radial correlations. It requires C++17 and `Global.cpp`, without FFTW/GSL. All positions and radii are in caller-defined physical length units in a cubic box `[0,L)^3`.

## Input and parallel contract

```cpp
#include <FML/Environment/EnvironmentClustering.h>
using namespace FML::ENVIRONMENT;
std::vector<Sample> tracers, centers; // Sample{std::array<double,3> position, uint64_t id}
```

IDs must be unique within each catalogue. `top_hat_environment`, `density_split_correlation` and `marked_correlation` are collective over `FML::communicator`. **Full catalogues and their order, density/mark vectors and options must be identical on every rank.** This follows the existing replicated pair-count backend; it is not a distributed-catalogue implementation. Schema checks reject incompatible inputs before pair work. Do not pass only the particles owned by each MPI slab. No catalogue gather is hidden in these functions.

Density queries are divided by index across ranks and only a compact count array is reduced. Pair work uses the existing MPI/OpenMP task assignment. Every rank receives the same results (floating-point weighted sums can differ at roundoff level between decompositions). Call collectives outside OpenMP regions. Local pure functions `density_quantiles` and `inverse_density_marks` do not communicate and throw locally.

## Environments

```cpp
auto env = top_hat_environment(tracers, centers, L, smoothing_radius);
// env.counts[i], env.delta[i]; delta = N_inside / expected_count - 1
```

The spherical radius is positive and at most `L/2`; sphere boundaries are inclusive. Counts are unweighted; expected count is `N_tracer * (4*pi/3)*(R/L)^3`. The supplied tracer number density is estimated from this catalogue. Repeated identical-coordinate tracers with distinct IDs count separately.

For environments at the tracers themselves:

```cpp
auto own_env = top_hat_environment(tracers, tracers, L, smoothing_radius, true);
```

`exclude_same_id=true` removes only the matched tracer index and uses `(N_tracer-1)` in its expected count. Matching IDs must have exactly matching positions; at least one tracer must remain. Without a matching ID there is no exclusion and the reference stays `N_tracer`. Independent centers can use a separate ID namespace when exclusion is false. No Poisson-noise subtraction or deconvolution is performed.

## Density quantiles and cross-correlations

```cpp
auto ds = density_split_correlation(centers, env.delta, tracers, 5,
                                    L, rmin, rmax, number_of_bins);
// ds.quantiles.label[i], .counts[q], .minimum[q], .maximum[q]
// ds.cross[q].{radius,pairs,xi}; ds.parent.{radius,pairs,xi}
```

Use uniform volume samples for `centers` to obtain volume-weighted environments; using tracer locations defines a different, tracer-weighted selection. A helper example uses the existing ID-keyed `CountsInCells::uniform_queries`. Arbitrary externally measured/smoothed environment values may replace `env.delta`, provided they are finite, aligned with the centers and consistent across ranks. No field interpolation or smoothing is applied to such values.

Sorting uses `(environment, ID)`. The first `N % Q` quantiles receive one extra center; all quantiles are nonempty and contain either `floor(N/Q)` or `ceil(N/Q)` samples. **Equal densities can be split between quantiles.** Reported minimum/maximum ranges may consequently overlap and are not unique threshold cuts for assigning new samples. IDs resolve ties deterministically and preserve assignments under catalogue reordering. IDs should be independent of position within tied environments; spatially ordered IDs can bias the spatial selection among ties. Uniform random query IDs satisfy that requirement. This matters particularly for discrete counts and empty spheres. `density_quantiles(values,ids,Q)` exposes the same ranking as a local pure operation.

Cross-correlations use `xi_q = D_q T / (N_q N_t f_shell) - 1`. One query–tracer pair traversal accumulates the parent and every quantile through the existing extra-quantity callback, avoiding repeated catalogue/cell construction per quantile. The identity is

```
sum_q (N_q / N_center) * xi_q(r) = xi_parent(r)
```

The finite set of uniform query points has sampling fluctuations: its parent cross-correlation need not vanish exactly. Do not force the quantile sum to zero or assume exactly equal quantile fractions when the sizes differ.

## Marked correlations and normalization

```cpp
auto marks = inverse_density_marks(own_env.delta, /*delta_s=*/1, /*power=*/1);
auto result = marked_correlation(tracers, marks, L, rmin, rmax, number_of_bins);
```

Marks are nonnegative finite scalars with positive sample mean; the current API uses unit baseline tracer weights. Marks must align with the tracer order. The inverse-density recipe is `((1+delta_s)/(1+delta_s+delta))^power` with `delta_s>0`, `power>=0`, and `delta>=-1`. It includes the caller's environment-estimation noise. Raw marks above `1e100` are rejected; overall mark scaling does not change the result.

One pair traversal accumulates unordered distinct-tracer `DD` and `WW=sum_ij (m_i/mean_m)(m_j/mean_m)` in every radial bin. With `A=N(N-1)/2`:

```
unmarked.xi = DD / (A f_shell) - 1
weighted_xi = WW / (A f_shell) - 1
mark_ratio = WW / DD = (1 + weighted_xi) / (1 + unmarked.xi)
```

`mean_mark` records the measured sample mean and `marked_pairs` stores `WW`. Self pairs are absent. An empty `DD` bin has `xi=-1` and `mark_ratio=NaN`, rather than an artificial zero/one mark ratio. Constant marks give unity wherever pairs exist.

Sample-mean normalization has a finite-population distinction. Let `g_i=m_i/mean_m`. The all-pair baseline is

```
finite_pair_baseline = sum_(i<j) g_i g_j / A
                    = (N^2 - sum_i g_i^2) / (N (N-1))
pair_normalized_ratio = mark_ratio / finite_pair_baseline
```

The first expression is evaluated with positive prefix products to avoid subtractive cancellation. The second ratio corresponds to separately normalizing marked and unmarked pair counts by their own all-pair totals; it is reported separately, not silently substituted for the sample-mean mark convention. For nonconstant marks, random shuffling over fixed positions has expected `mark_ratio=finite_pair_baseline`, and expected `pair_normalized_ratio=1` in nonempty bins. A single nonzero mark yields zero marked pair normalization, so the pair-normalized ratio is NaN. Baseline normalization does not remove bias from estimating marks/environments with noisy tracers.

## Bins, noise and current scope

Radial bins are linear with physical centers. Require `0 < rmin < rmax <= L/2`; the pair engine uses half-open `[rmin,rmax)` selection, including each lower edge. This avoids zero-separation/overlap ambiguity and permits the analytic spherical shell fraction `f_shell=(4*pi/3)*(r_hi^3-r_lo^3)/L^3`. Beyond half a box, that spherical volume is not the minimum-image shell volume. Top-hat counting separately uses an inclusive boundary. Finite double-precision coordinate/bin rounding follows the existing pair engine.

The measurements include the discreteness noise of the input tracer process, the finite query sampling and the shared-tracer dependence of estimated environments/marks. Positive-separation pair exclusion is not a general shot-noise correction. There is no blanket `1/nbar` subtraction for these nonlinear statistics. Validate query density, smoothing radius and tracer-density convergence; use independent noise realizations/thinning or a noise-aware model when interpreting cosmological signals. Uniform Poisson data can have density-selected correlations inside a smoothing sphere, even though the unsplit process is unclustered. Uncertainties require independent realizations or a justified resampling scheme; no error bars are estimated here.

This first version supplies periodic radial monopoles. Survey randoms/masks, arbitrary selection weights, density-split autocorrelations, angular multipoles and smooth-grid environment interpolation remain extensions; the existing lower-level pair and grid APIs remain available. No KD-tree internals were changed.

## Storage and work

Density ranking needs an O(N_center) index/label workspace and O(N_center log N_center) sorting. Top-hat measurements use the existing replicated KD-tree and per-query neighbor buffers; their count output is O(N_center), reduced in checked MPI chunks. Pair measurements convert each catalogue once to unit-box coordinates. Marked counts use two accumulators per radial bin, and density-split counts use `Q+1`, per thread, in one geometry traversal. Existing cell-list storage scales with the replicated catalogues; no new all-particle communication is introduced. Output storage is O(Q * number_of_bins). The code intentionally retains the existing KD-tree allocation behavior; deferred tree workspace optimizations remain separate.

## Build, example and tests

From `example/` or `tests/`:

```
make USE_MPI=false USE_OMP=false
make run USE_MPI=false USE_OMP=false # tests directory
# Clean between configurations, then:
make run MPIRUN=mpirun NP_LIST="1 2 4"
```

`example/Main.cpp` constructs reproducible uniform Poisson tracers and independent volume queries, calculates top-hat quintiles and their tracer cross-correlations, then constructs self-excluded tracer-density marks and prints both mark normalizations. It is a usage demonstration, not a cosmological forecast.

Tests use independent direct minimum-image enumeration for counts and marked/density-split pairs, plus hand-computed quantile and normalization cases. They cover inclusive top-hat radii, self-excluded `N-1` normalization, deterministic ties, input-order and mark-scale invariance, constant marks, empty bins, distinct finite-population conventions, recombined parent correlations, empty query ranks and collective schema/input failures. MPI/OpenMP and sanitizer runs are recorded in the accompanying validation report.

References: [Pinon et al., density-split clustering](https://arxiv.org/abs/2501.14638); [Skibba et al., marked correlations](https://arxiv.org/abs/astro-ph/0512463); [Armijo et al., inverse-density marks and conventional marked statistic](https://academic.oup.com/mnras/article/479/4/4824/5050389). These motivate the statistics; the explicit estimators and finite-sample conventions implemented here are stated above.
