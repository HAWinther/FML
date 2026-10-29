# Void finders

Four finders, all on particles with wrapped positions in `[0,1)^3` (owned by their local x slab), returning a
catalogue of `Void` (Void.h) that is the same on every rank:

- `SphericalVoidFinder` grows underdense spheres around density-grid candidates (radius: the largest sphere with
  mean density below (1 + Delta) times the mean, Padilla et al. 2005), optionally recenters them, and removes
  overlaps in descending radius order (`max_overlap` in [0, 1)).
- `VoxelVoidFinder` (Revolver): a smoothed CIC density grid and a watershed (`VoxelWatershed`, which also works on
  any density grid, e.g. from the DTFE).
- `ZobovVoidFinder` (Neyrinck 2008): the watershed of the Voronoi densities (needs CGAL, FML/Triangulation).
- Zones are merged by `ZoneMerging`: `None` (every zone is a void, Revolver), `Threshold` (merge through links below
  `link_density_threshold` times the mean, 0.2 as in VIDE, Sutter et al. 2015) or `Full` (the ZOBOV hierarchy;
  `parent` gives the void a void merges into). With `find_clusters` the same is done on 1/density; clusters with
  `Threshold` need `link_density_threshold > 1` (merge through links above it).

The spherical and voxel catalogues depend on the grid. The default grid size (`Nmesh = 0`, as in Revolver) is
rounded down to a multiple of the number of MPI tasks, so with the default the catalogue changes with the number of
tasks (a warning is printed): set `Nmesh` (a multiple of the number of tasks) for results that do not. With a given
grid the catalogues are identical for any number of tasks; ZOBOV's does not depend on a grid.

## Spherical-void particle exchange

Each rank receives only remote particles within the required periodic x buffer
of its domain. The buffer includes the maximum search radius, allowed center
drift, and one candidate-grid cell. It may span any number of slabs. Crossing a
single slab's width no longer triggers all-particle replication. The cell list
covers the buffered interval; it covers the full x range when that interval
actually spans the box.

The shared `FML/Global/PeriodicSlabExchange.h` helper appends ghosts from all
intersecting slabs, including across the periodic seam. A source record is sent
at most once to each destination; wrapped coordinates are preserved, with no
duplicate periodic images. Counts use `unsigned long long` and payloads use the
whole-record chunked transport with an 8 MiB default message/receive-scratch
limit. Send records are staged per destination and the final owned-plus-ghost
array is reserved before receipt; the message limit is not a total memory cap.
Nonuniform/empty slabs and rank-dependent buffers are supported by the helper.
The finder still requires the density-grid slabs to fit the particle domains.

For a roughly uniform catalogue, retained particle storage scales with the
buffered fraction `min(1, slab_width + 2*buffer)` rather than the whole catalogue.
Broad searches can still require the whole box. Candidate/sphere catalogue
gathering and the final replicated void catalogue are unchanged.

## Tests

`tests/test_spherical.cpp` checks exact ghost IDs with tiny forced MPI chunks,
periodic and slab boundaries, buffers wider than a slab, box-wide buffers,
empty sources, nonuniform/empty slabs, and different per-rank radii. Its small
brute-force catalogue reference uses all particles and checks radii, centers,
masses, member counts, and overlap removal, with and without recentering (it uses the same candidate grid and the
same first-crossing rule, so it checks the MPI exchange and bookkeeping rather than the algorithm). The voxel,
ZOBOV and merging code has no unit test; it was checked against an independent watershed/merging implementation
(Oct 2026 review).

Build with `make -C tests test_spherical` and run with MPI at 1, 2, 4, and 8 ranks;
four or more ranks exercise the former thin-slab replication fallback. Use
`test_spherical exchange` to run only the transport tests. MPI/FFTW paths and
compiler flags come from the normal FML Makefile configuration.
