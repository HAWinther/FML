# Triangulation

Periodic Delaunay/Voronoi tessellations of particles in the unit box and what is built on them:

| Header | What |
|---|---|
| `VoronoiTessellation.h` | `PeriodicVoronoiTessellation<NDIM>`: FML's own tessellation (no CGAL; Experimental/Delaunay). Voronoi volumes, neighbours, values of the neighbours on other tasks. The particles can be on any task; no buffer to choose; the result does not depend on the number of tasks |
| `PeriodicDelaunay.h` | `MPIPeriodicDelaunay`: CGAL's tessellation on the slabs (or a domain) with a buffer from the neighbour tasks and guard points (`USE_CGAL=true`) |
| `Watershed.h` | The watershed on the neighbour graph of either tessellation (`WatershedGeneral`, `WatershedDensity`, `WatershedDensityNative`) |
| `DTFE.h` | The DTFE density, fields and velocity derivatives on a grid, on either tessellation (`DTFETessellation`) |

The users (DTFE, ZOBOV in VoidFinders, ORIGAMIHalos in CosmicWeb, Analysis and SnapshotAnalyzer) choose the
tessellation with an option; CGAL's is the default in `USE_CGAL=true` builds, FML's own otherwise. The two give the
same results (the native tessellation rounds the positions to 2^-48 of the box, so values agree to about 1e-8 or
better); see `tests/test_watershed_native.cpp`, `tests/test_origami_halos.cpp`, `tests/test_dtfe_native.cpp` and
`Experimental/Delaunay/tests/compare_library.cpp`.

For points in degenerate positions (e.g. an exact lattice) the Delaunay triangulation is not unique and the two may
pick different tetrahedra: Voronoi volumes and neighbours across faces of nonzero area agree, but the DTFE (which
depends on the tetrahedra) can differ.
