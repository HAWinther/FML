# Connected four-point statistics

This C++17 module implements a bounded periodic-box scope for backlog item 34:

- Joint fourth moments of four **known zero-mean** variables, with the three Wick contractions reported separately from the raw moment.
- Shell-filtered Fourier fourth moments, integrated over ordered closing wavevector quartets, and their quartet-normalized connected shell averages.
- Independent-box power-spectrum covariance minus an explicit Gaussian covariance, with a volume-scaled collapsed trispectrum adapter.

It leaves the existing `compute_trispectrum` raw-fourth-moment API unchanged. It does not resolve the quadrilateral diagonal or angular shape, implement a survey four-point estimator, or automatically remove particle shot noise.

## Joint moments and the Wick reference

Include `ConnectedFourPoint.h` for the dependency-light core (only FML Global; no FFTW/GSL).

```cpp
using namespace FML::FOURPOINT;
auto moments = zero_mean_four_point(local_count, [&](size_t i) {
    return std::array<double, 4>{a[i], b[i], c[i], d[i]};
});
```

Each rank supplies disjoint rows. All ranks receive the same result. With `M_ab = average(x_a x_b)`, the result is

```
fourth       = average(x0 x1 x2 x3)
disconnected = C01 C23 + C02 C13 + C03 C12
connected    = fourth - disconnected
```

By default `C=M` from these same rows: this is a **finite-sample plug-in statistic**, not an unbiased ensemble cumulant estimator. Means are recorded for diagnostics; they are not estimated and subtracted. Correlated grid cells are not independent realizations, so an IID k-statistic correction based on the number of cells would be inappropriate.

An optional `std::array<double,16>` reference supplies a row-major covariance for the four fields. It must be finite, symmetric and positive semidefinite (within roundoff). Supplying the exact ensemble covariance gives an unbiased fourth-cumulant estimate for known zero means. A noisy independent covariance reference avoids reusing the measured field, but products of noisy covariance entries can still be biased; its uncertainty and product bias need separate treatment. The API cannot establish independence or the scientific accuracy of supplied references.

## Fourier shell estimator

Include `Fourier.h`; FFTW is required, with FFTW-MPI when `USE_MPI` is enabled.

```cpp
FourierGrid delta(nmesh); // fill real density contrast, then delta.fftw_r2c()
ShellOptions options{boxsize, k_edges};
auto results = connected_shell_four_point(delta, options,
                                         {{0,0,0,0}, {0,0,1,1}});
// Optional fourth argument: one covariance matrix per tuple.
```

`boxsize=L` is a positive physical box length. `k_edges` are strictly increasing, nonnegative **physical inverse-length edges**, unlike the bin-center arguments of some older FML power APIs. Shells are half-open `[low,high)`. Input is a physically valid real field's FML-normalized Fourier density contrast:

```
delta_k = (1/N^3) sum_x delta(x) exp(-i k.x)
f_b(x)  = sum_{k in shell b} delta_k exp(i k.x)
```

The caller is responsible for Hermitian symmetry, assignment corrections and density normalization. DC is always removed. Ambiguous even-mesh Nyquist planes are excluded; odd meshes retain all unambiguous modes. Selection and padding reuse the existing Fourier polyspectrum/resampling rules. Padding prevents closure modulo the mesh from contaminating genuine integer-wavevector closure. The input is never transformed or modified.

For requested tuple `(b0,b1,b2,b3)`, `moment.fourth` is `average(f_b0 f_b1 f_b2 f_b3)`, the **dimensionless shell-integrated closing-quartet sum**. Its `second_moment` and optional covariance reference are also dimensionless. For a homogeneous isotropic Gaussian model with power `P(k)` in volume units,

```
C_ab = 0                            if a != b (disjoint shells)
C_aa = (1/V) sum_{full signed k in a} P(k),  V=L^3
```

Repeated tuple legs referring to the same shell therefore share that shell covariance; e.g. all 16 entries are the same for `(a,a,a,a)`.

`mode_counts` include both members of each Hermitian pair. `closing_quartets` counts all **ordered** signed-wavevector tuples satisfying `k0+k1+k2+k3=0`, including repeated modes; it is not an independent-sample count. Geometry is measured with unit Fourier masks. The normalized outputs are

```
raw          = V^3 * moment.fourth       / closing_quartets
disconnected = V^3 * moment.disconnected / closing_quartets
connected    = V^3 * moment.connected    / closing_quartets
```

These have length^9 units. With the exact ensemble Wick reference, the expectation of `connected` is the connected trispectrum averaged over these quartets. This averages over every allowed diagonal/shape within the selected shells. It is distinct from the collapsed power-covariance statistic below. Empty shells or no closing quartets return `valid=false` and NaN normalized values; the raw integrated moments remain available. Zero power in a populated shell is valid. Negative connected estimates are retained.

All grid operations are collective over `FML::communicator`. Use the existing FFTWGrid slab layout, whose mesh width must be divisible by the MPI rank count. Grid/option/reference failures, including mismatches between ranks and nonfinite field values, are rejected before transforms. Core sample-callback exceptions are coordinated before reductions. Calls must be made on every rank outside parallel regions; this is not a fault-tolerant allocator/FFT wrapper.

For each tuple, at most four distinct real shell grids are retained, first for geometry and then for density, plus one source-resolution scratch grid during padding and FFT/resampling workspace. No particle or full mesh all-gather is used. Shells are rebuilt for each tuple rather than cached across all requests. At high k the work mesh can approach twice the input width (eight times the 3D payload per shell). `work_mesh` records the actual width; `moment.samples` counts work-grid cells, not independent physical samples. Many-bin production workflows may benefit from a bounded reusable cache later.

## Collapsed power covariance

```cpp
FML::ENSEMBLE::RunningCovariance powers(number_of_bins);
// Add each independent box's physical P(k) vector once, on one rank only.
auto ensemble = powers.global().summary();
auto gaussian = gaussian_covariance_disjoint_shells(sum_model_P_squared,
                                                     full_signed_mode_counts);
auto result = connected_power_covariance(ensemble, gaussian, boxsize);
```

`ensemble.covariance` must use `N_realizations-1` sample normalization for independent boxes, as `RunningCovariance` does. It is the covariance of **single-box powers**, not the covariance of their ensemble mean. The pure postprocessing functions are local and perform no collectives.

For disjoint, equally weighted periodic shells, the Gaussian helper returns

```
G_ij = delta_ij * 2 * sum_{full signed k in i} P_total_model(k)^2 / N_i^2
```

Use an external power model or separately calibrated ensemble, not `sum |delta_k|^4` from the same realization. `2 * mean(P)^2 / N` is only equivalent when P is constant within the shell. When P itself is estimated, its square also needs a bias/uncertainty treatment. Counts must match those of the measured estimator, including any DC/Nyquist exclusions. The helper does not support overlapping/weighted bins, power multipoles or survey windows; these require a separately supplied Gaussian matrix.

The adapter returns `total`, `gaussian`, `connected=total-gaussian`, and `collapsed_trispectrum=V*connected`. For matter density powers, units are length^6 for covariance and length^9 for collapsed T. A connected covariance may be negative or indefinite and is not itself a covariance suitable for a Gaussian likelihood. No division by the ensemble size or positivity clipping is applied. Gaussian-reference uncertainty is not propagated. Periodic-box assumptions exclude super-sample covariance and survey-window terms. For discrete tracers, subtracting Gaussian covariance alone does not remove all connected Poisson terms; report a discrete-field statistic or supply separately validated noise corrections.

The periodic covariance/trispectrum convention follows [Scoccimarro, Zaldarriaga & Hui, *Power Spectrum Correlations Induced by Non-Linear Clustering*](https://arxiv.org/abs/astro-ph/9901099). The implementation and tests use the explicit finite-mode formulas above.

## Building and validation

From `tests/`:

```
make run USE_MPI=false
make clean
make run MPIRUN=mpirun NP_LIST="1 2 4"
make clean
make run USE_MPI=false USE_FFTW=false   # core only
```

From `example/`, `make USE_MPI=false` builds a small sinusoidal-field and toy independent-power-vector example. It is an API demonstration, not a cosmological calibration. Clean between compiler/precision/MPI configurations. The root test suite registers both tests, with the Fourier test conditional on FFTW; module CI discovers `tests/Makefile` automatically.

See [tests/RESULTS.md](tests/RESULTS.md) for actual validation and [TODO.md](TODO.md) for remaining science/performance work.
