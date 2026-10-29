# Excursion-set voxel morphology

`VoxelMorphology.h` computes volume, surface area, integrated mean curvature and Euler characteristic for a union of **closed cubic voxels**. It accepts a compact `BinarySlab` or thresholds a real-field grid, including `FFTWGrid<3>`. MPI exchanges one adjacent x slice and reduces four integer counts; it never replicates the full field.

## Geometry and conventions

Let `C,F,E,V` count occupied cubes and all unique faces, edges and vertices in their closed cell complex; interior entities are included. For cell side `h=L/N`:

```
volume = C h^3
surface_area = (2 F - 6 C) h^2
integrated_mean_curvature = pi (3 C - 2 F + E) h
Euler = V - E + F - C
```

Mean curvature uses `(k1+k2)/2`, so a cube has integrated curvature `3 pi h`. Cavities contribute negative curvature. Edge- or corner-touching occupied cubes are connected as closed sets; Euler is not the component count from a six-neighbor flood fill. Handles and cavities also contribute. `Boundary::Periodic` identifies opposite faces of the box (a full periodic box has Euler zero). `Boundary::Open` takes empty space beyond the box and includes exposed box faces (a full open box has Euler one). A masked-out region set to zero introduces real voxel boundaries; no survey-mask correction is supplied.

`Measures::minkowski_density` contains the exact cubical geometric densities:

```
{ volume/L^3, area/(6 L^3), curvature/(3 pi L^3), Euler/L^3 }
```

`Measures::crofton_density` separately returns the conventional isotropic lattice Crofton estimates:

```
{ C h^3/L^3, (2/9)(F-3C) h^2/L^3,
  (2/9)(E-2F+3C) h/L^3, (V-E+F-C)/L^3 }
```

These two normalizations of area and curvature **differ**. The first describes the actual axis-aligned voxel boundary; the second is an isotropic estimator for a sampled underlying field. Staircase voxel area is orientation dependent and is not an unbiased smooth-surface area. No marching cubes, interpolated isosurface or derivative-based continuum estimator is implemented. Specify smoothing, resolution, threshold and estimator when comparing scientific results; test convergence for the intended field and estimator.

## Usage

```cpp
#include <FML/Morphology/VoxelMorphology.h>
using namespace FML::MORPHOLOGY;
// smoothed_density must be a real-space grid; values and thresholds use caller units.
auto results = excursions(smoothed_density, {-0.5, 0.0, 1.0}, boxsize,
                         Boundary::Periodic, ThresholdRule::AtLeast);
for (const auto& r : results) {
    // r.threshold, r.rule, r.measures.{volume,surface_area,
    // integrated_mean_curvature,euler_characteristic,counts,...}
}
```

The default is `value >= threshold`; `Greater` gives strict `>`. Threshold order and duplicates are preserved. All physical field values and thresholds must be finite. Choose density, overdensity or normalized field thresholds explicitly; no mean/variance normalization or smoothing is performed. Only physical cells are read, excluding FFT padding and ghosts. Fields with a status getter must be in real space. The minimal const grid protocol is `get_nmesh()`, `get_local_nx()`, `get_local_x_start()`, `get_real(std::array<int,3>{local_x,y,z})`. Accessors used by threaded loops must support concurrent reads.

For an existing binary mask:

```cpp
BinarySlab mask;
mask.nmesh = N;
mask.x_start = global_first_x;
mask.local_nx = owned_x_slices;
mask.cells.resize(mask.local_nx * size_t(N) * N);
// cells[(local_x*N + y)*N + z]: zero empty, nonzero occupied
// Fill the locally owned cells, then collectively:
auto result = measure(mask, boxsize, Boundary::Open);
```

Every rank must participate with the same `N,L,boundary,thresholds,rule` and a rank-ordered contiguous partition covering the box. Unequal and empty slabs are supported. Binary storage must exclude padding and halos. Returned results are global and identical on every rank. Validation failures are collective for these APIs; MPI calls occur outside OpenMP parallel regions.

## Memory and cost

One field threshold uses one byte per owned physical cell, one `N*N` byte boundary slice on nonempty ranks, O(number of ranks) layout metadata and O(number of threads) counters. `measure` uses the caller's binary storage directly. `excursions` allocates its byte mask once and reuses it for every threshold. Each threshold performs O(number of local cells) work; thresholds are processed sequentially instead of retaining a mask per threshold. Counts use checked 64-bit integer geometry bounds. The cell complex is counted by vertex-anchored incident-voxel patterns, so each entity is owned exactly once, including MPI and periodic seams. Counting is exact across rank/thread decompositions; derived floating-point measures follow from those integer totals.

## Build and tests

Core code links only `Global.cpp`; no FFTW/GSL library is required for a custom field or binary mask. `FFTWGrid` inputs use their usual FFTW build options. From `example/` or `tests/`:

```
make USE_MPI=false USE_OMP=false
make run USE_MPI=false USE_OMP=false  # tests only
make run MPIRUN=mpirun NP_LIST="1 2 4" # MPI build, clean first
```

Clean objects between configurations. The example thresholds a deterministic periodic cosine field and prints physical geometric measures.

Tests compare the incident-pattern implementation with an independent explicit set-union enumeration of vertices/edges/faces. Analytic cases cover empty/full boxes, isolated cubes, face/edge/corner contacts, disconnected components, a cavity, a periodic solid ring and cross-seam contacts. Checks also cover physical length scaling, the separate Crofton normalization, a deterministic irregular mask, uneven/empty MPI slabs (`N=1,2,7`), threshold ties/order, nonfinite fields, incompatible MPI schemas, Fourier-state rejection and ignored NaN padding. The ensemble module's optional FFTW test checks the real-grid adapter.

Gaussian random fields (checked 29 October 2026): for a white-noise field on 128^3 cells smoothed with a Gaussian of 4 cells, the Crofton densities averaged over 4 realizations agree with the continuum Gaussian-field predictions (Schmalzing & Buchert 1997, with sigma_0, sigma_1 of the realizations) to ~1% for V0 and V1, 1-4% for V2 and 2-7% for V3 at nu = -2..2 (statistical errors 1-4%). The exact-cube area density is 3/2 of the Crofton value, as expected for a staircase surface. Validate smoothing/resolution effects for the intended field before production inference.

References: [Schmalzing & Buchert (1997)](https://arxiv.org/abs/astro-ph/9508154), [Schmalzing et al. (1999), lattice counting appendix](https://cds.cern.ch/record/385977/files/arXiv%3Aastro-ph_9904384.pdf). Formulae are documented explicitly above so boundary/normalization conventions remain reviewable.
