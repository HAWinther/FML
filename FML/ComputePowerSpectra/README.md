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

The existing dense result vectors and accessors are retained for compatibility.
Selected configurations save integration work and shell grids, but do not yet
make the n^ORDER output tensors sparse.
