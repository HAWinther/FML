# Eulerian checkpoint and restart validation

Recorded 18 November 2026. Bounded integration checks pass.
[Machine-readable summary](summary.json); [API and output contract](../../../EULERIAN_RESTART.md).

The native suite covers shared N8 and compact N4/N8 field/product grids with N16
deposition, smooth and PSWF assignment, joint field/scalar and fixed-scalar field
HMC. It splits during warmup, continues into sampling, then resumes in a separate
process. Encoded bytes of final positions/metrics/statistics and retained row and
potential suffixes are compared to uninterrupted references, including sink draw
indices and cumulative completed counts. The strict byte comparison preserves
signed-zero distinctions rather than relying only on floating-point equality.

Serial prepare/resume passes 15/43 checks. Each 1/2/4-rank MPI prepare/resume pair
passes 30/86 checks, using slabs plus default pencils (1x1, 2x1 and 2x2). Tests
inject a one-rank shard-write failure and verify CURRENT stays on a complete,
readable prior generation. Corruption of one shard and changes to callback version,
HMC controls, data, fixed noise, field prior, meshes, assignment/tolerance,
evolution, filters and parameter priors are rejected collectively without
publishing a replacement checkpoint.

The executable example runs 12 warmup trajectories and 16 toy draws in serial,
on two slab ranks and on four pencil ranks. An uninterrupted run matches a
three-launch run stopped after two warmup trajectories, after two retained draws,
and at completion. All 16/32/64 per-rank draw JSON files are byte-identical in
those respective cases. An orphan draw file created beyond the committed prefix
is replayed correctly. Starting over in a nonempty directory is rejected. Separate
CLI checks reject changed rank count, pencil process grid and thread count while
leaving CURRENT unchanged. The plotted alpha and field-component traces are a
restart illustration, not evidence of posterior mixing or amplitude recovery.

Legacy field/survey checkpoint and PM checkpoint suites pass at 1/2/4 ranks after
the default configuration-codec refactor. A retained executable built before the
refactor writes real two-rank linear/1LPT/2LPT Fourier/survey checkpoints (21
prepare checks); the current executable resumes them exactly (36 resume checks).
The old executable hash is pinned in summary.json. Existing configuration payload
order, shard and generation formats are preserved.

Builds use Clang 17, double precision, bounds checks and FFTW; MPI runs use Open
MPI and two OpenMP threads per rank. All exact commands and logs, example draw
records, source hashes and restart-comparison.png are outside the library under
workspace `outputs/fieldlevel-eulerian-restart/`. The figure was visually inspected.

Restart supports the same rank count/domain/process grid, thread count and
compatible build; it does not repartition chains. User callbacks require a
versioned problem_id. The specialized fixed-IC amplitude/bias Metropolis sampler
and the existing Cartesian survey adapter for the Eulerian model are not added
by this increment. No production-scale posterior, new catalog, long-chain mixing,
large-rank/float or power-loss/fsync durability validation is claimed. Catalog
inputs, initial cubic filtering, convergence/resource checks and the scientific
free-IC application remain open.
