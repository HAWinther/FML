# Native Delaunay triangulation and periodic Voronoi tessellation (experimental)

Experimental: kept in `FML/Experimental/` while it is developed (MPI, moving mesh); the interface may change and it
is not yet used by other modules.

Header-only C++17 module in `FML::DELAUNAY` (2D and 3D), with no external dependencies: the building block for a
moving-mesh (AREPO-style) hydro solver and an alternative to the CGAL-based [Triangulation](../../Triangulation) module.
Serial, with OpenMP for the Voronoi geometry; MPI (ghost points from neighbouring domains) is a planned next step.

| Header | Contents |
|---|---|
| `Predicates.h` | Exact `orient` and `insphere` predicates (floating-point filter + exact expansion arithmetic) |
| `Delaunay.h` | `Delaunay<D>`: incremental Bowyer-Watson triangulation with a symbolic perturbation for degenerate input |
| `PeriodicVoronoi.h` | `PeriodicVoronoi<D>`: Voronoi cells (volume, centroid) and faces (area, centroid, normal, neighbour, periodic shift) of points in a periodic box |

## Usage

```cpp
#include <FML/Experimental/Delaunay/PeriodicVoronoi.h>
using namespace FML::DELAUNAY;

std::vector<std::array<double, 3>> points = ...; // in [0, boxsize)^3 (other positions are wrapped)
PeriodicVoronoi<3> voronoi(points, boxsize);
for (size_t i = 0; i < voronoi.ncells(); i++) {
    const auto & cell = voronoi.cells()[i];     // cell.volume, cell.centroid
    auto [begin, end] = voronoi.cell_faces(i);  // the faces of cell i
    for (auto it = begin; it != end; ++it) {
        const int64_t code = *it;
        const auto & face = voronoi.faces()[code >= 0 ? code : -code - 1];
        // face.area, face.centroid, face.normal (from face.i to the image of face.j), face.distance, face.shift;
        // code >= 0: this cell is face.i, code < 0: this cell is face.j (the normal points into it)
    }
}
```

Every face is stored once, between cell `i` and the image of cell `j` at `x_j + shift * boxsize` (`j > i`, or
`j == i` with a lexicographically positive shift: with very few points a cell can neighbour its own images). Face
centroids are given near point `i`; cell centroids are in the coordinates of the point, so they can be slightly
outside the box. Faces of zero area (from degenerate input, e.g. the diagonal neighbours on a lattice) are dropped
by default (`Options::drop_zero_area_faces`, area below `zero_area_tolerance` = 1e-14 times distance^(D-1); set
`drop_zero_area_faces = false` to keep every Delaunay edge). `Options::initial_buffer` (in mean interparticle
spacings, default 2) and `verbose` control the periodic images; `statistics()` gives the number of images, the final
buffer and timings.

`Delaunay<D>` can also be used directly for points in a bounding box: `Delaunay<3> dt(lo, hi); dt.insert(points);`
then `get_simplices()` (vertices `v[0..D]`, positively oriented, and neighbours `n[k]` across the facet opposite
`v[k]`), `incident_simplices(v, ...)`, `circumsphere(s, ...)` and `check()`. The first D+1 vertices are the corners
of an enclosing super simplex.

## Algorithms

- **Predicates.** `orient(p0..pD)` = sign det[p1 - p0, ..., pD - p0] and `insphere(p0..pD, q)` (+1 inside the
  circumsphere of a positively oriented simplex), CGAL's conventions. The determinant is evaluated in floating point
  with Shewchuk's (1997) a priori error bound (times a safety factor 4); if the sign is uncertain it is computed
  exactly with floating-point expansions from the input coordinates. When all coordinate differences are exact
  doubles (lattices, or the quantized coordinates of `PeriodicVoronoi`) a fixed-size stack version is used, otherwise
  a general one. Exact for any double input barring overflow/underflow.
- **Triangulation.** Points are inserted in Peano-Hilbert order (`FML::GRID::HilbertCurve`): a stochastic visibility
  walk from the last new simplex locates the point, the cavity of simplices whose circumspheres contain it is
  found by a breadth-first search, and it is replaced by the simplices joining its boundary facets to the point
  (neighbours matched with a small hash table). Cospherical points are resolved by the symbolic perturbation of
  Devillers & Teillaud (2011), as in CGAL: the result is the Delaunay triangulation of points perturbed in the order
  of a per-vertex key, a valid Delaunay triangulation of the input (unique for every input). Duplicates are not
  inserted (`insert` returns the existing vertex).
- **Periodic Voronoi.** Coordinates are mapped to the unit box and rounded to multiples of 2^-40 (relative change
  ~1e-12), so periodic images are exact translates and the predicates see identical configurations around a point
  and around its images; the perturbation key is (point index, image shift), which is translation invariant. The
  points and their images in a buffer around the box are triangulated; the buffer grows (inserting the new shell's
  images into the same triangulation) until every simplex touching a point in the box has its circumsphere inside
  the triangulated region, which makes those simplices final. Faces are the polygons of the circumcentres around each
  Delaunay edge (2D: segments); cell volumes and centroids are sums over the cones from the point to its faces.

## Validation

`make -C Experimental/Delaunay/tests run` (41 checks, about 15 s; also clean under AddressSanitizer/UBSan):

- Predicates equal exact 128-bit integer determinants for 2x200000 random small-integer cases (thousands of exactly
  degenerate and cospherical ones), and the filter agrees with the exact evaluation on 100000 nearly degenerate
  double configurations.
- Triangulations of random, clustered, lattice (fully cospherical) and duplicated points are valid, positively
  oriented and locally Delaunay across every facet; brute force: no vertex strictly inside any circumsphere.
- Periodic Voronoi (2D and 3D): volumes sum to the box to 1e-15, closed cells (|sum A n| / sum A < 1e-13), volumes
  from the divergence theorem with the face centroids agree to 1e-12; lattices with offsets 0, 0.5, 0.37 give
  cubes/squares with 2D faces (to the 1e-11 rounding of the coordinates); translation invariance; 1, 2, 3 and 7
  points (cells neighbouring their own images); a clustered set across the box corner; identical results with 1 and
  2 threads; duplicates and invalid input rejected.

`make -C Experimental/Delaunay/tests compare_cgal CGAL_INCLUDE=...` compares with CGAL 5.6's periodic triangulations: the Delaunay
edges (with periodic offsets) are identical for random and clustered sets of 1e5 points and random sets of 1e6
points in 2D and 3D. Timings (one core of a cloud VM, GCC 13 -O2):

| Case | FML triangulation | FML geometry | CGAL periodic triangulation |
|---|---|---|---|
| 2D, 1e5 random | 0.20 s | 0.14 s | 0.60 s |
| 3D, 1e5 random | 1.15 s | 0.91 s | 1.73 s |
| 3D, 1e5 clustered | 1.04 s | 0.82 s | 1.70 s |
| 2D, 1e6 random (geometry on 2 threads) | 2.2 s | 1.5 s | 8.9 s |
| 3D, 1e6 random (geometry on 2 threads) | 11.9 s | 6.5 s | 22.0 s |

Exactly degenerate input is slower (many exact evaluations and larger cavities): a periodic 3D lattice takes 25-30
us per point, a random set about 8 us per point (including the images).

## Limitations and next steps

- Serial triangulation; MPI with ghost points from neighbouring domains (as `MPIPeriodicDelaunay` does for CGAL) is
  the next step, then use in a moving-mesh solver (see the plan in the project notes).
- The geometry step (circumcentres, face polygons) is not yet optimized; points cannot be moved or removed (the
  triangulation is rebuilt).
- Coordinates are rounded to 2^-40 of the box; positions closer than that are duplicates and rejected.
- References: J. R. Shewchuk, Discrete Comput. Geom. 18, 305 (1997); O. Devillers & M. Teillaud, Comput. Geom. 44
  (2011); V. Springel, MNRAS 401, 791 (2010) for the use of such tessellations in AREPO.
