# Void finders

`SphericalVoidFinder` grows underdense spheres around density-grid candidates,
optionally recenters them, and removes overlaps in descending radius order.
Particles must be owned by their local x slab and have wrapped positions in
`[0,1)^3`. The returned catalogue is the same on every rank.

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
masses, member counts, and overlap removal, with and without recentering.

Build with `make -C tests test_spherical` and run with MPI at 1, 2, 4, and 8 ranks;
four or more ranks exercise the former thin-slab replication fallback. Use
`test_spherical exchange` to run only the transport tests. MPI/FFTW paths and
compiler flags come from the normal FML Makefile configuration.
