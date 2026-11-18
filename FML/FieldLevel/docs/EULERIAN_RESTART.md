# Eulerian LPT checkpoint and restart

Include `EulerianCheckpoint.h` to use the existing checkpointed HMC driver with
`DistributedEulerianLPTModelT<Grid>`. Both shared and compact/product configurations,
smooth and PSWF assignment, and slab/pencil layouts are supported. The posterior
must use the whitened-field prior; the periodic fixed-noise Gaussian target is
validated here.

```cpp
#include <FML/FieldLevel/EulerianCheckpoint.h>
// model/data are built with the same versioned physical inputs on every launch.
auto target = FF::fixed_noise_posterior(model, observed_modes, noise_power);
FF::CheckpointOptions io;
io.directory = "/external/run/checkpoint";
io.problem_id = "analysis-v1:spectrum-source-v2:epoch-and-input-contract-v1";
io.every = 10;             // Save every ten completed trajectories, including warmup.
io.max_iterations = 40;    // Budget for this invocation, not a new sampling schedule.
io.resume = restarting;
auto result = restarting
    ? FF::sample_checkpointed(target, {}, {}, options, io)
    : FF::sample_checkpointed(target, model.white_noise(seed),
                              model.initial_parameters(), options, io);
```

Keep the same original HMC options when resuming. The saved position, gradient,
energy, adaptation state, RNG progress, statistics and frozen metric parameters
are restored. `max_iterations` may change between calls. Restart begins at the
last saved **completed trajectory**, including in warmup; a partially executed
trajectory is rerun. Saved checkpoints allow stopping after a bounded budget.
They do not provide an in-trajectory continuation.

The result's rows and potentials contain this invocation's suffix, while
acceptance/divergence statistics and completed counts are cumulative. Field sinks
receive the global retained-draw index. Passing `vary_parameters=false` samples
the field with fixed scalar parameters. The specialized fixed-IC amplitude/bias
Metropolis samplers are unchanged and do not gain checkpointing through this API.

## Target identity and compatibility

The adapter records compact, explicit/resolved product and deposition mesh sizes,
assignment/kernel/tolerance, evolution order, boxsize, initial cutoff, matter and
likelihood band geometry/cutoff, operator/transfer versions, grid/domain/rank/local
field and mode layout, scalar priors, Gaussian noise/field-prior policy and observed
Fourier data. Changes reject collectively before continuation. A versioned
`problem_id` is required: arbitrary spectrum callbacks and external source physics
cannot be fingerprinted automatically. Include their versions and conventions in
that string, and update it when they change.

Use a checkpoint directory visible to every rank. Restart requires the same rank count, process grid/domain, thread count, grid
precision and compatible compiler/library build. Repartitioning saved chains is
not supported. The configuration adapter encodes its own contract without fake
legacy configuration fields or an artificial Fourier response. It does not enable
the existing Cartesian survey likelihood for the Eulerian model.

The shared codec and generation format are unchanged. The [FL-7 metric correction](EULERIAN_METRIC.md)
adds a numerical identity tag: pre-FL-7 **Eulerian** checkpoints require their original
executable or a fresh chain. Legacy backend identities retain their bytes. Default configuration
encoding retains the old byte order for existing linear/LPT/PM/survey targets.
[Validation](validation/studies/eulerian_restart/RESULTS.md) includes real restart
files produced by a pre-refactor executable and read by the new executable.

## Atomic generations and output records

Each checkpoint generation has one checksummed rank shard. `CURRENT` is published
only after all ranks have written and closed their complete shards. An interrupted
or failed write leaves the previous complete generation available. A corrupt or
missing shard fails collectively. This is process-failure/atomic-publication
protection; no machine-power-loss/fsync durability test is claimed. Keep generation
files while a reader may still reference them.

The generic driver does not transactionally manage application chain files.
`example/EulerianCheckpoint.cpp` demonstrates indexed output records: each draw is
written per rank to a temporary JSON file and atomically renamed, using its global
draw index as a stable key. Replay after an unsaved interval replaces orphan files
with the same deterministic records. Each write is collective before the sampler
can publish a checkpoint containing that draw.

The example publishes `summary.json` after a completed driver invocation. Its
`completed_samples` and generation identify the committed prefix; read only draw
indices below that count, not every file in `draws/`. A crash before summary
publication can leave a conservative older prefix until the next successful
resume. Additional files beyond the committed count are ignored/replayed. Binary
checkpoint shards, not the JSON records, restore the sampler.

```sh
make -C FML/FieldLevel/example eulerian_checkpoint USE_MPI=false
/path/to/eulerian_checkpoint /external/demo start 2
/path/to/eulerian_checkpoint /external/demo resume 12
/path/to/eulerian_checkpoint /external/demo resume 100
# Use a clean/separate build for a different backend:
make -C FML/FieldLevel/example eulerian_checkpoint USE_MPI=true USE_PENCIL_GRID=true
mpirun -np 4 /path/to/eulerian_checkpoint /external/pencil-demo start 8
mpirun -np 4 /path/to/eulerian_checkpoint /external/pencil-demo resume 100
```

Start requires a new/empty output directory. This small example uses a Gaussian
mock, N4 compact field, N8 products and N16 deposition, with 12 warmup trajectories
and 16 draws. Its run is a restart demonstration, not posterior validation or a
production resource recommendation. Set MPI/compiler/FFTW variables for the local
installation, and retain the same settings between launches.

```sh
make -C FML/FieldLevel/tests eulerian-checkpoint USE_MPI=false
make -C FML/FieldLevel/tests eulerian-checkpoint USE_MPI=true NP_LIST='1 2 4'
```

The tests cover fresh-process joint and field-only continuation, mid-warmup and
mid-sampling splits, exact suffixes/final states/metrics/statistics/sink indices,
collective write failure, corruption and target mismatches. Real catalog inputs,
initial cubic filtering, catalog-specific convergence, larger-run resource/mixing
validation and the scientific free-IC result remain open.
