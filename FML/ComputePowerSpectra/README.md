# Fourier shell polyspectra

`PolyspectrumBinning::dealias` defaults to `true`. The estimator counts genuinely
closed integer-wavevector configurations rather than closure modulo the FFT mesh.
When the configured shell range could wrap an ORDER-fold product into its zero
mode, the shell fields and geometry masks are evaluated on a padded mesh. Their
normalization uses the work mesh volume. Low-frequency ranges that cannot wrap
keep the original FFT width.

For nonlinear orders (ORDER > 2), even-mesh Nyquist planes are excluded from both
field and geometry selections: their signed Fourier wavevector is ambiguous.
Strictly sub-Nyquist modes are retained, including all represented modes of odd
meshes. Ordinary power spectra keep their full even/odd r2c multiplicities.

Set `bins.dealias = false` explicitly to reproduce the previous modulo-grid
estimator, including its Nyquist convention. This affects physical results at
high wavenumbers. The fourth-order function remains a raw shell-averaged fourth
moment; disconnected Gaussian contributions are not subtracted.

Padding is expensive: at full radial Nyquist range, each 3D shell grid grows by
approximately 3.375 times for a bispectrum and 8 times for a fourth moment. Grids
are rounded to an even width divisible by the MPI task count. Shell work grids
have no particle-deposition halos.

Geometry counts are cached automatically after successful computation. Internal
cache identities include the source mesh width, exact shell edges and the
closure policy. Changing any of these, or calling `reset`, invalidates reuse.
For external counts use `set_bincount(counts, source_mesh)` to record the mesh
identity. The legacy `set_bincount(counts)` overload trusts the caller and binds
the counts to the mesh of their first use; it cannot verify which mesh originally
produced an unlabelled vector. Imported counts must use the same bin edges and
closure convention as the receiving object.

## Memory and MPI execution options

```cpp
bins.options.max_cached_shells = 6;              // 0: all required shells (default)
bins.options.shell_memory_budget_bytes = 256ULL * 1024 * 1024; // per rank; 0: unlimited
bins.options.reduction_batch_size = 4096;        // 0: one batch, up to MPI int-count limits
compute_bispectrum(delta_fourier, bins);
```

The two cache limits are combined. The cache must hold the largest number of
**distinct** shells in a requested configuration (up to ORDER). Too small a
budget throws before shell FFTs. Repeated indices share one grid, so a purely
diagonal selection can use a one-grid cache even for a fourth moment.

The byte budget covers resident zero-halo **work-resolution shell grids**, using
FFTW's local allocation size. It excludes the caller's input, one source-resolution
grid used while padding a cache miss, MPI remapping buffers, FFT workspace, allocator overhead, and dense
result arrays. It is not a hard process-RSS limit. Across MPI ranks the smallest
permitted capacity is used, so every rank executes the same FFT/eviction schedule.

A deterministic least-recently-used cache protects all shells of the current
configuration while loading misses. Evicted shells are rebuilt on demand, so
less memory can mean more FFTs. All-needed-shell caching remains the default.
The geometry and density phases are sequential and each has its own cache.

Configuration sums are reduced in bounded arrays, followed by symmetry expansion.
Auxiliary shell powers and mean k use one Fourier traversal and one array
reduction. Statistics in `bins.statistics` report per-phase shell builds and
reduction batches, maximum resident shell count, and the maximum local shell-grid
bytes across ranks. They describe the latest call; reused geometry has zero builds.
These counters do not include FFT/remapping or cache-setup collectives.

## Selected configurations

```cpp
bins.set_configurations({{1, 2, 3}, {4, 4, 4}});
compute_bispectrum(delta_fourier, bins);
bins.set_configurations({}); // restore all configurations
```

Bin indices are zero-based, sorted and deduplicated. Only valid canonical
configurations are integrated, and all their permutations are populated.
Unrequested results/counts are zero; auxiliary P(k) and mean k still cover all
shells. Only requested shells enter the cache. Padding width is chosen from the
requested range, so unused high-frequency shells do not force a larger FFT.
Selections must match on every MPI rank. Changing the selection invalidates
geometry; reordering or repeating an equivalent selection does not.

The default dense result vectors and accessors are retained for compatibility.
For compact output, use the sparse types below.

## Sparse canonical results

```cpp
SparseBispectrumBinning<3> bins(100, nmesh);
bins.set_configurations({{1, 2, 3}, {4, 4, 4}});
bins.options.max_cached_shells = 3;
compute_bispectrum(delta_fourier, bins);

double value = bins.get_spectrum(3, 1, 2); // permutation-aware lookup
for (size_t i = 0; i < bins.P123.size(); ++i) {
    auto tuple = bins.get_coord_from_index(i); // sorted canonical bin indices
    double spectrum = bins.P123[i], geometry = bins.N123[i];
}
```

`SparsePolyspectrumBinning<N, ORDER>` and the aliases
`SparseMonospectrumBinning`, `SparseBispectrumBinning`, and
`SparseTrispectrumBinning` use the same estimators, normalization, shot-noise
rules, padding and shell caches as dense results. The particle overloads also
accept these types. Fourth-order results remain raw fourth moments.

Construction allocates shell metadata, not `nbins^ORDER` results. On estimation
(or an explicit `prepare_storage()` call), `P123` and `N123` get one element per
valid canonical configuration. `stored_configurations()` gives their tuple
ordering. Selected tuples are sorted and deduplicated; empty selection restores
all canonical configurations. An impossible selection has no stored slots.
Large bin counts therefore need explicit selections if all canonical tuples
would themselves be too numerous.

Lookup sorts tuple indices and uses binary search, without expanding permutations
or building a dense index table. Valid unrequested tuples return zero from
`get_spectrum`/`get_bincount`; out-of-range bins throw `std::out_of_range`.
`get_index_from_coord` throws when a tuple has no stored slot. Raw indices are
compact offsets, not the dense base-`nbins` encoding. Reduced spectra retain the
existing denominator convention, including its tuple ordering at fourth order.

Changing selection releases old compact results; equivalent selections retain
geometry reuse. Cache identity still includes mesh, exact edges, selection and
closure policy. To import geometry use `set_bincount(compact_counts, source_mesh)`
with the same canonical ordering, edges and selection. `reset()` clears values
and invalidates geometry; `free()` releases results and the stored canonical
tuple list, retaining shell metadata and the selection for lazy rebuilding.
Scaling and combining estimates work
as for dense bins; combining requires the same storage type and selection.

Dense allocation is explicit:

```cpp
auto dense = bins.to_dense(); // allocates nbins^ORDER values AND counts
```

This expands every stored permutation and carries auxiliary powers, mean k,
settings and geometry identity. Unrequested values remain zero. Dense size
products use checked `size_t` arithmetic; addressable arrays can still be much
larger than available RAM. No estimator calls `to_dense()` internally.

For example, 100 fourth-order bins formerly require two arrays of 100 million
doubles (1.49 GiB per rank). With one selected tuple, the sparse result has one
power and one count (16 bytes), one canonical tuple, and O(nbins) metadata.
Workspace options bound shell grids separately; they are not total-RSS limits.

## Total evaluation-storage report

The shell-cache budget above remains a budget for resident shell payloads only.
`bins.statistics.memory` now reports the larger evaluation footprint separately:

```cpp
// Optional caller estimate of FFT plans/opaque transform workspace per rank.
// This is an estimate you supply, not a measured FFTW allocation or a limit.
bins.options.fft_workspace_allowance_bytes = 64ULL * 1024 * 1024;
compute_bispectrum(delta_fourier, bins);
const auto &memory = bins.statistics.memory;
std::cout << "Accounted evaluation storage: " << memory.accounted_peak_bytes << " bytes\n";
if (memory.estimated_peak_bytes)
    std::cout << "Including supplied FFT allowance: " << *memory.estimated_peak_bytes << " bytes\n";
```

The report includes these byte components:

| Component | Included storage |
| --- | --- |
| `input_bytes` | Caller's retained grid capacity, including halos, real/r2c padding and FFTW excess allocation; zero for a standalone geometry-only call |
| `result_bytes` | Both dense or compact P123/N123 vector capacities |
| `bin_metadata_bytes` | Binning object and shell/selection/geometry-cache/canonical-tuple vector capacities |
| `configuration_bytes` | Evaluation's canonical configuration list, separate from compact result indexing |
| `cache_metadata_bytes` | Cache object and entry/control storage, separate from shell payloads |
| `shell_bytes` | Maximum simultaneously resident shell payload capacities during the latest call |
| `padding_source_bytes` | Largest source-resolution shell retained while constructing a padded shell |
| `remapping_bytes` | Routing geometry plus the larger of construction scratch or retained communication buffers for Fourier remapping |
| `metadata_work_bytes` | Auxiliary shell-power/mean-k/count reduction vector |
| `reduction_bytes` | Largest configuration-sum batch vector |
| `fft_workspace_allowance_bytes` | Largest caller-supplied FFT allowance over ranks |

Component maxima are taken over the latest geometry/field evaluation phases and
all MPI ranks. `accounted_peak_bytes` sums the first ten components; this is a
conservative sum of tracked component maxima, which can occur at different times
or on different ranks. `estimated_peak_bytes` adds the supplied FFT allowance.
It is **unset unless every rank supplies an allowance**. Explicit zero is an
asserted zero allowance, distinct from leaving it unspecified. Totals use checked
wide arithmetic and fail collectively if they cannot fit size_t.

These are evaluation-storage estimates after result preparation, not measured
process peak RSS or hard upper bounds on every allocation. They exclude allocator
and container-growth transients, output construction/reconfiguration transients,
small temporary stack/callback/label bookkeeping, opaque MPI/runtime storage and
other application objects (including particle catalogues). FFTW plan caches and
internal workspaces are opaque; the caller allowance must cover the relevant
retained plans and transforms for the application. Supplying an allowance does
not enforce it or establish a rigorous memory bound.

The report resets with the existing per-call statistics. Reused geometry has no
new geometry-shell builds, but its retained result/metadata storage still counts.
Unpadded execution reports zero padding/remapping bytes; standalone geometry
counts report no external input grid. `to_dense()`, `free()` or later manual
vector changes do not rewrite a historical estimator report: run the estimator
again for a new snapshot. Cross-rank memory maxima add bounded end-of-phase
reporting collectives, excluded from the configuration-reduction batch counters.

`FFTWGrid::allocated_storage_bytes()` exposes retained payload capacity rather
than just logical mesh size; copy assignment may retain a larger buffer.
`FourierResamplingPlan::workspace_storage_bytes()` includes routing, construction
scratch and retained payload buffers, without counting input/output grids again.
The report does not change cache limits, configuration order, FFT counts,
normalization or result storage. Cache-aware scheduling in backlog item 12 is
still waiting.
