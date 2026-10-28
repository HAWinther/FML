# Spherical-overdensity measurements

`SphericalOverdensity<T, NDIM>` in `SphericalOverdensity.h` measures halos around locally supplied seed centers using the distributed particle catalogue. The routine is collective over the FML communicator, including ranks without particles or halos. It requires the same search radius, density thresholds and global total mass on every rank. Halo counts and initial centering radii may differ between ranks.

Particle coordinates must lie in the periodic unit box, with x owned by the rank's half-open slab `[xmin_domain, xmax_domain)`. Slabs must form a contiguous, rank-ordered partition of `[0,1]`; nonuniform and empty slabs are supported. Empty slabs must have no particles or halo centers. Particle masses (or the unit-mass fallback), thresholds and global total mass must be finite and positive. Seed centers can be unwrapped finite coordinates and are normalized to `[0,1)`. Initial radii must be finite and nonnegative; the search radius must be finite and positive.

## Multi-slab coverage and centering

Every slab intersecting a rank's periodic x buffer supplies its required particles, including nonadjacent slabs. Each remote source particle is received once, even when the expanded periodic intervals overlap. The SO cell index uses wrapped coordinates and minimum-image separations in every dimension. A wide search radius has no slab-width restriction.

The initial buffer covers the growth sphere around every local seed. During shrinking-sphere centering, the code checks coverage before each query. If a center moves far enough to require additional particles, all ranks participate in a selective buffer refresh and the paused centering iteration resumes. The final growth sphere receives the same coverage check. `SO_number_of_buffer_refreshes` counts these collective refreshes in the last call. The existing 10 percent shrinking rule, 200-iteration cap, mass-weighted centers, outermost density crossing, velocity moments and truncation convention are retained. Floating-point summation order can differ with the MPI decomposition.

A coverage radius of half the box spans the complete periodic x axis. In that case every remote particle is needed by the slab buffer and is received once. There is no particle all-gather or special replication fallback triggered by crossing a slab width.

## Serialization and message bounds

The optional final SO argument `max_exchange_bytes` defaults to 8 MiB. The collective minimum of the rank limits, capped at `INT_MAX`, bounds each serialized MPI message. Each complete record contains an unsigned-long-long byte-length header followed by its particle payload; a single framed record must fit the limit. Oversized records fail collectively instead of being split through a user serializer.

Fixed-size trivially copyable particles use FML's default byte serialization. Other types supply `get_particle_byte_size()`, `append_to_buffer(char*)` and `assign_from_buffer(char*)` through the particle reflection interface. Sizes may vary between particles, but the size reported for a particle must remain stable during packing and match the reconstructed record. Custom methods must write/read their declared payload exactly. Exceptions from packing or decoding become collective errors before the next communication step. Record lengths, aggregate receive counts and MPI byte counts are checked before narrowing or allocation.

Transport scratch consists of local serialized-size metadata, one destination's index list, rank metadata and bounded message buffers. It does not stage a complete serialized peer catalogue. The received particle vector is reserved from checked counts. Old ghosts and cell pointers are released before a centering refresh. Memory still scales with the particles actually received, the local cell index and the largest queried sphere; the message limit is not a total-memory limit.

The helper remains callable as:

```cpp
auto ghosts = FML::FOF::fetch_boundary_particles(part, count, width);
```

Its default `unwrap_periodic_images=true` selects the x image closest to the receiving slab, preserving the legacy unwrapped-coordinate convention. The SO routine passes `false` and uses wrapped minimum-image queries. The helper accepts rank-dependent widths and message limits; owned input coordinates are left unchanged. It returns no ghosts in serial.

## Regression coverage

`tests/test_so.cpp` retains the analytic isothermal and cored-NFW tests. `tests/test_so_multislab.cpp` checks exact ghost membership, unique periodic records, variable-size nontrivial serialization, bounded multi-message exchanges, empty catalogues, nonuniform/empty slabs and collectively rejected inputs or serializer failures. A direct full-catalogue reference checks centers, radii, masses, counts, velocity means/dispersion and truncation, including centers that leave their owner's slab. Native float positions, unit-mass/absent-velocity fallbacks and unwrapped seed centers are also covered. Both executables are registered in `tests/Makefile`.
