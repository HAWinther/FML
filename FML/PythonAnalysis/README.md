# FML Python analysis interface

`fml-analysis` exposes a deliberately small, serial 3D catalogue API backed by the existing [C++ spatial-statistics implementation](../CountsInCells/README.md). Public API schema 1 consists of `Catalogue`, its methods below, and `uniform_queries`. It does not wrap MPI containers, grid templates, simulation evolution or weighted tracers. Legacy [SWIG demonstrations](../MakePythonWrapper) and the specialized [halo-model wrapper](../HaloModel/PythonWrapper) remain separate.

## Build and run

Python >=3.9, NumPy >=1.23, a C++17 compiler, setuptools, wheel and pybind11 >=2.11,<4 are required. The package build declares those requirements; FFTW/GSL/MPI/SWIG are not used. Only Python 3.11 was exercised in this implementation pass; other supported interpreter/platform combinations need CI coverage.

From the repository root:

```sh
python3 -m pip install ./FML/PythonAnalysis
python3 FML/PythonAnalysis/examples/catalogue_statistics.py
python3 -m unittest discover -s FML/PythonAnalysis/tests -v
```

Use a virtual environment. For an offline environment that already has build requirements, append `--no-build-isolation` to `pip install`. Select a compiler with `CXX`, for example `CXX=c++`. Build isolation may download build dependencies when absent. Do not inject FML MPI/FFTW/OpenMP flags: this extension enforces a serial build and disables FML's automatic runtime setup. Import does not initialize or finalize MPI/FFTW, even inside a Python process that separately uses mpi4py. Each process operates on its own complete catalogue; results are local, with no hidden collectives/gathers or communicator ownership.

A wheel contains the native extension and public package. `python3 -m build FML/PythonAnalysis` also produces a source archive carrying the transitive FML headers and `Global.cpp` needed to rebuild outside the checkout. Rebuild/install the extension after modifying its C++ dependencies. This is a native CPython package, not a Python stable-ABI/abi3 wheel.

## Owned catalogue and array contract

```python
import numpy as np
from fml_analysis import Catalogue, uniform_queries

positions = np.array([[0.1, 0.2, 0.3], [0.9, 0.2, 0.3]])
cat = Catalogue(positions, boxsize=1.0)
queries = uniform_queries(100, [[0, 0, 0], [1, 1, 1]], seed=17)
distances = cat.knn_distances(queries, [1, 2])
cdf = cat.knn_cdf(queries, [1, 2], [0.05, 0.1, 0.2])
counts = cat.sphere_counts(queries, [0.05, 0.1, 0.2])
summary = cat.counts_in_cells(queries, [0.1, 0.2])
```

Positions and queries have shape `(n,3)`; explicitly use `np.empty((0,3))` for an empty sample. Real integer/floating arrays and corresponding nested lists are accepted. Noncontiguous, reversed, read-only and float32 inputs are supported via conversion to double. Boolean, complex, string and object arrays are rejected. All positions must be finite with absolute coordinate <=1e150. Units of positions, radii, box size and window bounds must match; no cosmological unit conversion is applied.

Construction copies coordinates into C++-owned KD-tree storage; changing/deleting the source array cannot change the catalogue. No raw pointer or mutable view into the catalogue is exposed. `len(cat)` and the read-only `cat.boxsize` report its size and geometry. `boxsize>0` selects a cubic periodic box with minimum-image distances and coordinate wrapping; `boxsize=0` selects nonperiodic Euclidean distances. Negative or nonfinite box sizes fail.

Each call copies query coordinates and exclusion/order/radius arrays before releasing the GIL for C++ work. Results are independent, writable NumPy-owned arrays (float64 values, uint64 counts; `orders` uses native `size_t`/NumPy uintp). Results remain valid after deleting the catalogue or other result objects. Do not mutate an input concurrently while its initial conversion/copy is in progress. There are no background jobs or OpenMP threads in the extension. Copying inputs and outputs is intentional; this version has no zero-copy view API.

## Methods and result schema 1

All method options after the required query/order/radius arguments are keyword-only. Orders must be a nonempty one-dimensional array of distinct positive integers; their supplied order is preserved. Radii must be a nonempty one-dimensional array of finite nonnegative strictly increasing values. `counts_in_cells` requires strictly positive radii. Float/bool order and exclusion arrays are rejected rather than truncated.

- `knn_distances(queries, orders, *, exclude=None)` returns shape `(n_orders,n_queries)` in length units. An unavailable neighbor has distance `inf`.
- `sphere_counts(queries, radii, *, exclude=None)` returns uint64 shape `(n_radii,n_queries)`, including each tracer within the radius once. Boundaries are inclusive, duplicates count separately, and zero radius counts coincident tracers. Beyond half a periodic box it counts a toroidal minimum-image ball, with no spherical-volume interpretation.
- `knn_cdf(queries, orders, radii, *, exclude=None)` returns a dict with `orders`, `radii`, integer `hits` and float `cdf` of shape `(n_orders,n_radii)`, scalar `queries`, `excluded_queries` and `boxsize`. Fractions are hits/query count; unavailable neighbors remain in the denominator. With no queries, fractions are NaN.
- `counts_in_cells(queries, radii, *, exclude=None, histogram_limit=128, factorial_order=4, delta_edges=None, number_density=None, window=None, allow_truncated_cells=False)` returns the summary dict described below. `delta_edges=None` uses C++ defaults `[-1,-.5,0,.5,1,2,4,8]`; an empty array disables that histogram. Factorial order is 0–16.

`exclude` is a one-dimensional integer array with one entry per query. Each entry is the original zero-based catalogue index to exclude, or **-1** for no exclusion. There is no automatic nearest/zero-distance removal; coincident other tracers stay. Invalid indices fail. Tracer-centered queries measure a tracer-conditioned statistic, not a volume sample, even with exclusion.

The counts summary has keys `cells` (one dict per radius), `delta_edges`, `histogram_limit`, `boxsize`, `excluded_queries` and `uncorrected_clipping`. Each cell dict contains:

| Fields | Convention |
| --- | --- |
| `radius`, `volume`, `reference_number_density`, `queries` | Physical spherical volume and reference number density |
| `histogram`, `probability` | Counts/probability for occupancies 0..limit, followed by one N>limit bin; length limit+2 |
| `mean`, `variance` | Population count moments; variance divides by query count |
| `factorial_moments`, `normalized_factorial_moments` | Falling factorial moments including order zero; normalization divides by mean^order |
| `poisson_subtracted_variance`, `normalized_connected_variance` | Var(N)-mean and that value/mean^2 |
| `mean_delta`, `density_histogram`, `density_probability`, `density_pdf` | Delta=N/(nbar*volume)-1; histogram probability mass and mass/bin width |
| `density_underflow`, `density_overflow`, `density_undefined` | Counts outside delta edges or without a positive reference expected count |

Periodic summary radii cannot exceed L/2. A nonperiodic summary normally needs `window=[[xmin,ymin,zmin],[xmax,ymax,zmax]]`, containing the tracers and every complete cell. There is no inferred mask or boundary correction. Explicit `allow_truncated_cells=True` permits uncorrected clipping and records it; an explicit `number_density` or window is still required. Default density is catalogue size divided by box/window volume. Empty tracers with nonempty queries have P(N=0)=1; no queries give undefined moments/PDFs (NaN). Undefined positive-order normalized moments at zero mean are also NaN. Factorial/Poisson subtraction assumes a Poisson-sampling interpretation and is not a universal galaxy shot-noise correction. See the C++ contract for estimator limits and density-PDF tail conventions.

`uniform_queries(number, bounds, *, seed, first_id=0, stream=0)` returns float64 `(number,3)` volume points in half-open rectangular bounds. Philox keys each point by global query ID: splitting a range into batches preserves positions exactly. Nonoverlapping ID ranges avoid accidental repeated samples. Overlapping cells remain statistically correlated; deterministic query generation does not imply independent counts.

Invalid array dtype gives `TypeError`; invalid shape, geometry, radius/order or estimator domain gives `ValueError`. Standard allocation failure maps to `MemoryError`. Binding-level unsigned scalar conversion (negative/out-of-range seed, count or histogram arguments) raises `TypeError`/`OverflowError` rather than wrapping. C++ exceptions are translated after reacquiring the GIL; the process is not aborted for ordinary invalid inputs. Errors leave the immutable catalogue reusable.

## Tests

The unittest suite compares against independent NumPy brute-force minimum-image distances/counts and moments. It checks duplicate/excluded tracers, histogram tails, nonperiodic windows, empty samples, invalid inputs, result/source lifetimes, writable independent output storage, deterministic query batching and post-error reuse. A source-archive rebuild and installed-wheel example are also part of the implementation validation. These are unit/packaging checks, not production survey validation or a distributed Python backend.
