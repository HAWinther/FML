# Native Delaunay triangulation and periodic Voronoi tessellation (experimental)

Experimental: kept in `FML/Experimental/` while it is developed (MPI, moving mesh); the interface may change and it
is not yet used by other modules.

Header-only C++17 module in `FML::DELAUNAY` (2D and 3D), with no external dependencies: the building block for a
moving-mesh (AREPO-style) hydro solver and an alternative to the CGAL-based [Triangulation](../../Triangulation) module.
MPI parallel (any distribution of the points over the tasks), with OpenMP for the Voronoi geometry; the results are
bitwise identical for any number of tasks and any distribution of the points.

| Header | Contents |
|---|---|
| `Predicates.h` | Exact `orient` and `insphere` predicates (floating-point filter + exact expansion arithmetic) |
| `Delaunay.h` | `Delaunay<D>`: incremental Bowyer-Watson triangulation with a symbolic perturbation for degenerate input |
| `PeriodicVoronoi.h` | `PeriodicVoronoi<D>`: Voronoi cells (volume, centroid) and faces (area, centroid, normal, neighbour, periodic shift, tasks) of points in a periodic box, with MPI |

## Usage

```cpp
#include <FML/Experimental/Delaunay/PeriodicVoronoi.h>
using namespace FML::DELAUNAY;

std::vector<std::array<double, 3>> points = ...; // this task's points, in [0, boxsize)^3 (others are wrapped)
PeriodicVoronoi<3> voronoi(points, boxsize);       // collective; optional Options and global ids
for (size_t i = 0; i < voronoi.ncells(); i++) {
    const auto & cell = voronoi.cells()[i];     // cell.id (global id), cell.volume, cell.centroid
    auto [begin, end] = voronoi.cell_faces(i);  // the faces of local cell i
    for (auto it = begin; it != end; ++it) {
        const int64_t code = *it;
        const auto & face = voronoi.faces()[code >= 0 ? code : -code - 1];
        // face.area, face.centroid, face.normal (from face.i to the image of face.j), face.distance, face.shift;
        // code >= 0: this cell is face.i, code < 0: this cell is face.j (the normal points into it)
    }
}
```

Each task gets the cells of its own points (in input order) and every face touching them, once. A face is between
point `i` and the image of point `j` at `x_j + shift * boxsize`, with global ids (`j > i`, or `j == i` with a
lexicographically positive shift: with very few points a cell can neighbour its own images). Global ids are the ones
given to the constructor, or by default the position in the concatenation of the tasks' points in rank order (the
input index in a serial run). A face also has `task_i`, `task_j` (the tasks holding the two points), `index_i`,
`index_j` (their indices on those tasks) and `local_i`, `local_j` (the local cell on this task, or `npos`): a face
between two tasks is on both, bitwise identical, so a code can compute a flux once (e.g. on `task_i`) or on both
sides consistently. Faces are sorted by (i, j, shift). Face centroids are given near point `i`, in its box
coordinates; cell centroids are in the coordinates of the point, so they can be slightly outside the box. Faces of
zero area (from degenerate input, e.g. the diagonal neighbours on a lattice) are dropped by default
(`Options::drop_zero_area_faces`, area below `zero_area_tolerance` = 1e-14 times distance^(D-1); set
`drop_zero_area_faces = false` to keep every Delaunay edge). `statistics()` gives the number of ghosts, rounds and
searches and timings.

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
- **Periodic Voronoi with MPI.** Each task triangulates its points together with ghosts: copies of other tasks'
  points and periodic images. A first set comes from a buffer of `initial_buffer` (default 2) mean interparticle
  spacings around the bounding box of the task's points. Then every simplex touching a local point is *certified*:
  its circumsphere is searched on every task (and periodic image) whose bounding box it meets, the points inside are
  sent back and inserted, and a simplex that was searched and survives is final (no other point can be inside its
  circumsphere). Spheres that meet only the task's own points need no search; super-simplex neighbours trigger a
  larger buffer. Rounds (typically 3) continue until no task has uncertified simplices. This needs no assumption on
  the distribution of the points (slabs, pencils, Peano-Hilbert ranges, random, or empty tasks), and is exact.
  Searches and replies use `FML::COMMUNICATION::exchange_records_chunked` (bounded messages).
- **Determinism.** Coordinates are mapped to the unit box and rounded to multiples of 2^-40 (relative change
  ~1e-12): a point, its images and its copies on other tasks have exactly translated coordinates and every
  coordinate difference is exact. The perturbation key is (global id, image shift), translation invariant. The
  geometry is computed from exact differences only: circumcentres from the key-ordered vertices of a simplex
  (relative to its lowest-key vertex), the polygon of the circumcentres around a Delaunay edge started at the
  smallest simplex and walked in a fixed direction, positions relative to the face's point `i`, cell sums in the
  face order. So the cells and faces depend only on the configuration: identical on any number of tasks and on both
  sides of a face. Faces are the polygons of the circumcentres around each Delaunay edge (2D: segments); cell
  volumes and centroids are sums over the cones from the point to its faces.

## Validation

`make -C Experimental/Delaunay/tests run` runs `test_delaunay` (41 checks, about 15 s; also clean under
AddressSanitizer/UBSan) and `test_delaunay_mpi` on 1, 2, 3 and 4 tasks (30 checks each):

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
- MPI: random, clustered and lattice points (2D: 20000, 20000, 4900; 3D: 20000, 20000, 4096) distributed in x slabs,
  x-y pencils, Peano-Hilbert ranges, at random and all on the last task give cells and faces bitwise identical to
  the run with all points on task 0, every face present on the task(s) of its two points (on both if they differ)
  and identical there, on 1, 2, 3 and 4 tasks.

`make -C Experimental/Delaunay/tests compare_cgal CGAL_INCLUDE=...` compares with CGAL 5.6's periodic triangulations: the Delaunay
edges (with periodic offsets) are identical for random and clustered sets of 1e5 points and random sets of 1e6
points in 2D and 3D. Timings (one core of a cloud VM, GCC 13 -O2; FML includes the ghosts and the certification):

| Case | FML triangulation | FML geometry | CGAL periodic triangulation |
|---|---|---|---|
| 2D, 1e5 random | 0.14 s | 0.17 s | 0.63 s |
| 3D, 1e5 random | 0.76 s | 0.60 s | 1.70 s |
| 3D, 1e5 clustered | 0.69 s | 0.65 s | 1.76 s |
| 2D, 1e6 random | 1.4 s | 1.7 s | 6.9 s |
| 3D, 1e6 random | 7.3 s | 7.5 s | 18.0 s |

MPI (3D, 4e5 random points in x slabs, 2 cores): 5.7 s on one task, 3.2 s on two (1.8x). Distributions that give
tasks large surfaces (random assignment of points to tasks) need many more ghosts and searches. Exactly degenerate
input is slower (many exact evaluations and larger cavities): a periodic 3D lattice takes 25-30 us per point.

## Limitations and next steps

- The triangulation itself is serial within a task (MPI across tasks; OpenMP only in the geometry). Points cannot
  be moved or removed (the triangulation is rebuilt). No load balancing: each task triangulates the points it is
  given.
- Circumspheres are searched with a relative margin of 1e-6: a sliver with a circumcentre wrong by more than that
  (condition number above ~1e10) could in principle be certified incorrectly.
- Coordinates are rounded to 2^-40 of the box; positions closer than that are duplicates and rejected.
- References: J. R. Shewchuk, Discrete Comput. Geom. 18, 305 (1997); O. Devillers & M. Teillaud, Comput. Geom. 44
  (2011); V. Springel, MNRAS 401, 791 (2010) for the use of such tessellations in AREPO.
