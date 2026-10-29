# Reproducible performance cases

This suite measures representative FML operations with explicit workload, build
and measurement provenance. It is separate from correctness tests and does not
set timing/RSS thresholds in CI. The initial cases are deliberately small enough
to run locally; larger/clustered/application-specific cases can extend the suite.
It does not benchmark every module or replace production scaling studies.

## Build and run

Requires C++17, Python >=3.9 and FFTW. MPI/OpenMP are enabled by default; GSL/Lua
are not required. Use the shared dependency paths in `Makefile.local` or command
line overrides. The build enables `FML_ENABLE_PERFORMANCE_COUNTERS` consistently
in its own object directory. Never mix counter-enabled and ordinary objects.
Normal FML builds leave instrumentation disabled and retain their original grid
storage type. `make clean` removes executables/objects, not saved report folders.

```sh
make -C FML benchmark -j2
make -C FML benchmark-check
python3 FML/Performance/run.py --exe FML/Performance/fml_benchmark \
    --output /tmp/fml-performance-run --mesh 32 --ranks 1 2 4 --threads 1 2 \
    --repeat 5 --warmup 1
```

The output directory must be new: prior results are never overwritten. Each
case/rank/thread combination runs in a fresh process/launcher invocation. The
runner retains a JSON report, raw stdout/stderr log and exact argument vector for
each case, an aggregate `results.jsonl`, and `manifest.json` with host/platform,
CPU count/driver affinity, selected OpenMP settings and executable SHA-256.
`completed=false` remains if a run fails partway through. A timeout terminates
the process group started for that case and preserves its log.

Use `--launcher "mpirun --oversubscribe"` if appropriate for the MPI installation.
Arguments are passed as an array without a shell. Set process/thread placement
explicitly for serious comparisons, e.g. OMP_PROC_BIND/OMP_PLACES and launcher
binding options; inspect raw logs for MPI binding/oversubscription warnings.
The driver sets OMP_DYNAMIC=false and OMP_WAIT_POLICY=passive, and varies
OMP_NUM_THREADS. Actual initialized OpenMP and FFTW thread counts are recorded
separately. FFTW threading requires the matching optional thread library and
`USE_FFTW_THREADS=true`; OpenMP alone does not enable threaded FFTs.

For serial or float builds, clean first or use separate source/build directories:

```sh
make -C FML/Performance clean
make -C FML/Performance USE_MPI=false USE_OMP=false
python3 FML/Performance/run.py --exe FML/Performance/fml_benchmark \
    --output /tmp/fml-performance-serial --ranks 1 --threads 1

# MPI float example; supply your compiler/dependency paths as needed:
make -C FML/Performance clean
make -C FML/Performance CXXFLAGS="-O3 -DSINGLE_PRECISION_FFTW" \
    FFTW_LINK=-lfftw3f FFTW_MPI_LINK=-lfftw3f_mpi
```

The runner probes the executable first. It rejects multiple ranks for a serial
build and thread counts above one for a non-OpenMP build. The source mesh must
be >=8 and divisible by every requested rank count. `--message-bytes` defaults
to 1 MiB; small values such as 37 exercise forced payload chunking.
`--cache-shells` is zero (all needed shells) or at least three for the selected
polyspectrum configurations. Input seed, repeat/warmup counts, message cap and
cache setting are recorded even when irrelevant to a particular case.

A single invocation can be run directly:

```sh
OMP_NUM_THREADS=2 mpirun -np 4 FML/Performance/fml_benchmark \
    --case resample --mesh 32 --repeat 5 --warmup 1 \
    --message-bytes 1048576 --output /tmp/resample.json
```

Direct executable output paths are caller-controlled and can be overwritten;
the Python runner provides the new-directory protection and run manifest.

## Workload definitions

| Case | Timed work | Validation outside timing |
| --- | --- | --- |
| `fft_roundtrip` | Normalized forward/inverse FFT pair on one preallocated analytic 3D field | Analytic field recovery with precision/repeat-scaled tolerance; exact wrapper FFT-call counts |
| `grid_copy` | Construct a fresh full grid copy and replace the prior destination | Exact coefficients, allocation/copy-byte counts |
| `grid_clone` | Construct a fresh `clone_empty()` and replace the prior destination | Owned coefficients zero; one allocation per repeat and no payload copies for this zero-halo Fourier input |
| `resample` | Apply one reused 3D routing plan from mesh M to 2M, with bounded messages | Independent signed-mode/ownership oracle, exact coefficients/zeros and submitted payload bytes |
| `migration` | Migrate a balanced lattice-ID catalogue after a prepared left/stay/right rank hop | Exact local ID catalogue/positions and global serialized payload-byte oracle |
| `polyspectrum` | Four selected canonical bispectrum configurations with a bounded shell cache | All-shell-cache numerical reference and agreement of shell-build statistics with actual inverse FFT calls |

Preparations (including migration position updates) are outside timing. Migration
has M^3 global particles and capacity local_count+3 per rank; each ID's fixed
left/stay/right class keeps local active counts within that capacity. It is a
balanced protocol fixture, not a realistic clustered cosmological drift. The
polyspectrum input and selected tuples are deterministic by global coordinates;
its padded work mesh is recorded. With warmup, geometry counts and frequently used FFT plans are
primed; new alignment-dependent plan creation is reported explicitly; `--warmup 0` instead includes cold setup performed by timed APIs.

Correctness failure produces a nonzero exit; a success report is written only
after numerical/counter checks. Particle validation reconstructs the expected
local ID set; it does not replicate the global catalogue. Validation is outside
timing and after the RSS sample. This permits timing/cache changes while still
rejecting lost records, incorrect modes or inconsistent FFT/payload counters.

## Timing, RSS and counter semantics

A barrier precedes each timed call. Local elapsed steady-clock time ends when
the operation returns; the maximum over ranks is reduced outside that timing.
The report saves every maximum-rank sample, its median (mean of the two middle
values for an even sample count) and the largest sample. Work repetitions are
not averaged across ranks. Warmup and correctness/reference work are excluded.
Instrumentation itself is enabled during timing, so compare like builds.

OS peak RSS is sampled after measured repetitions and **before final validation**.
It is a process-lifetime high-water mark including loader, MPI/FFTW, setup,
warmup, input and measured work. It cannot be reset or interpreted as an
incremental per-call allocation. `max_rank_peak_rss_bytes` is the largest rank's
peak; `sum_rank_peak_rss_bytes` sums rank high-water marks that need not occur
simultaneously. Unsupported RSS readings produce null. These measurements are
separate from [polyspectrum evaluation-storage estimates](../ComputePowerSpectra/README.md).

Counters aggregate **all measured repetitions**, excluding warmup, setup and
validation. JSON contains both rank sums and rank maxima. One distributed FFT
invoked on P ranks contributes P to the rank sum and one to each rank's count;
it is not P different global transforms. Counters are process-local atomics,
reset/sampled at quiescent boundaries. Local overflow saturates and is flagged;
the benchmark rejects overflow and conservatively checks rank-sum upper bounds.

| Counter | Coverage |
| --- | --- |
| `fftw_plan_creations` | New FFTWGrid cached-plan creations; distinguishes cold/alignment effects after warmup |
| `forward_ffts`, `inverse_ffts` | FFTWGrid wrapper execution calls, excluding FFTW's internal planning operations |
| `grid_storage_allocations`, `grid_storage_allocated_bytes` | Grid payload vector allocations (construct/copy/load), planning backups and explicit FFT halo-backup arrays; cumulative capacity bytes requested, not live/peak bytes |
| `grid_copy_operations`, `grid_copied_bytes` | Grid storage copy construction/assignment and explicit planner/halo/empty-clone bookkeeping copies; empty-clone auxiliary spans count as one logical event |
| `transport_buffer_allocations`, `transport_buffer_allocated_bytes` | Resampling and migration payload-buffer allocations/capacities; excludes routing/count metadata |
| `payload_sendrecv_calls` | Instrumented resampling/migration payload calls, including zero-byte participation |
| `payload_send_messages` | Those payload calls with a nonzero send |
| `payload_sent_bytes`, `payload_received_bytes` | Submitted application payload byte arguments, without control headers or network protocol overhead |
| `transport_packed_bytes`, `transport_unpacked_bytes` | Serialized particle or complex-value packing/unpacking through those payload paths |

This is scoped instrumentation, **not a global allocator/memcpy/MPI profiler**.
Zero initialization and arithmetic stores are not counted as copies; fewer
logical copied bytes need not mean lower runtime or RSS. It omits arbitrary
kernel copies, result/metadata allocations, other modules'
transport, control collectives and FFTW's internal communication/allocation.
In particular, zero traced payload bytes in the FFT case does not mean the FFT
communicated no data. Counter scope is also embedded in each JSON report. RSS
includes more than the allocation counters; neither is interchangeable with
allocator capacity or the supplied FFT-workspace allowance.

## Provenance and comparison

Reports embed compiler version/command/options/include/library flags, precision,
C++ standard, MPI and FFTW library versions, planner flag, actual rank/thread
configuration and rank hostnames. Generated `BuildInfo.h` records repository
revision and tracked dirty state plus a SHA-256 of transitive FML C++ sources and
benchmark build/driver/test sources. A copied build directory and an in-tree
build use canonical source labels in that digest. The runner separately hashes
the executable actually launched; keep reports with their manifest and raw logs.
No full environment dump is saved.

Compare the same source/binary/configuration except the intended change, and
repeat runs with controlled placement and machine load. Fixed inputs and
provenance are reproducible; wall time/RSS are observations, not deterministic
results. There are no built-in speedup claims or timing pass/fail thresholds.
Peak RSS is available on the supported Linux/macOS runtime paths; additional
platforms and benchmark cases need their own validation.

`TestCounters.cpp` checks enabled/disabled behavior, copy/move/clone accounting,
thread safety, saturation/reset and FFT calls. `test_reports.py` checks all six
cases, cold runs, metadata/JSON/median correctness, argument failures and runner
output protection. CI runs these functional checks without performance gates.
The existing grid, particle and Fourier-resampling regressions should also pass
with instrumentation disabled and enabled. Memory logging can be combined with
these counters; it remains a separate facility.
