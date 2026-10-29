# Particle migration routing

`MPIParticles<T>::communicate_particles(max_message_bytes, force_general)` moves each particle to the rank owning its x coordinate. Existing calls without arguments remain valid. The optional message limit retains Engineering 5's default `INT_MAX` and collective-minimum behavior. `force_general` defaults to false; if any rank sets it true, every rank uses general routing for that call. It is useful for comparisons or explicitly disabling the optimized route.

The container's domains must form a finite, contiguous rank-ordered partition of `[0,1]`. Nonuniform and empty slabs are allowed. Particles must have finite x in `[0,1]`; the existing exact-boundary convention remains: with multiple ranks, x=1 moves to the preceding representable coordinate and x=0 to the smallest positive normal coordinate of the particle's position type. Ordinary callers remain responsible for periodic wrapping.

## Ownership and route selection

Particles already in the local half-open slab remain local. Departures check the two cyclic neighbor intervals first. Other destinations use `upper_bound` on slab upper boundaries, reducing worst-case owner lookup from linear to logarithmic in rank count. Repeated boundaries are skipped, so zero-width slabs never own particles. Position values retain their native comparison precision, including long double.

All ranks select a common route after counting outgoing particles:

- `NoTraffic`: nobody has an outgoing particle; skip count exchange, buffer construction and payload communication.
- `Neighbors`: all outgoing destinations are cyclic rank neighbors. Exchange particle/byte-count pairs with those neighbors, then exchange payloads on the same edges. Two ranks have only one distinct neighbor, so it is contacted once. With three ranks every other rank is a neighbor.
- `General`: at least one destination is not adjacent, or a rank requested `force_general`. Retain the general all-to-all count exchange and cyclic all-peer payload route.

Eligibility concerns destination ranks, not a measured displacement. A periodic seam crossing can use the neighbor route. Skipping an empty slab may require general routing when the destination is not adjacent in rank order. Idle and initially empty ranks participate in the collective decision and receive normally.

`get_last_migration_route()` reports `FML::PARTICLE::MigrationRoute::{Serial,NoTraffic,Neighbors,General}`. Serial calls report `Serial`; before any call the default is also `Serial`. The getter records the last selected route and is intended for diagnostics; it is not a solve/failure status.

## Preserved contracts and limits

Native/custom particle serialization, checked 64-bit counters, receive-capacity checks and message chunking retain their existing rules. Owned-particle compaction and per-source receive order are the same in automatic and forced-general routing. Messages can split variable-size records because complete peer buffers are reassembled before deserialization.

The optimization reduces contacted payload peers from `P-1` to at most two for neighbor moves. Existing per-rank count/offset arrays still occupy `O(P)` metadata, and complete local outgoing/incoming byte buffers are still staged. This does not add automatic capacity growth, transactional failure recovery, nearest-nonempty-rank graph construction or communication/computation overlap. A globally selected general route remains necessary for arbitrary jumps. Rebuild dependent objects after changing this header/API.

## Regression coverage

The `migration` group in `tests/test_particles.cpp` compares automatic routing with general routing forced by only rank 0. It validates exact IDs, all fields, per-rank particle order and the selected route against an independent linear ownership scan. Cases include no traffic, two-sided neighbors, periodic seams, a single-rank long jump, arbitrary destinations, exact/adjacent boundaries, initially empty ranks, nonuniform/empty slabs and variable-size records split into small messages. A long-double boundary fixture guards against narrowing position values during lookup.

The full particle suite also retains existing lattice shifts, batched creation, checkpoints, custom serialization and container/reflection tests. Focused timing artifacts compare automatic and forced-general protocols with identical workloads; both versions in those timings use the new ownership lookup, so they isolate route selection rather than measuring the previous implementation end to end.
