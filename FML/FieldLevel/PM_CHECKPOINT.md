# Same-layout PM inference checkpoint/restart

Include [ParticleMeshCheckpoint.h](ParticleMeshCheckpoint.h) to use the existing
`sample_checkpointed()` API with `PeriodicPMModelT<Grid>`. This explicit adapter
supports the periodic Fourier-Gaussian PM posterior on FFTW slabs or MPI pencils.
[The example](example/ParticleMeshCheckpoint.cpp) supplies a complete small mock.
[Runtime results](tests/PM_CHECKPOINT_RESULTS.md) record validation and limits.

## Use

```cpp
#include <FML/FieldLevel/ParticleMeshCheckpoint.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// Construct the same PM model, posterior/data and full HMC schedule in every job.
FF::CheckpointOptions io;
io.directory = "pm-chain";
io.problem_id = "my-pm-target-v1:spectrum-v2:mock91-noise92:code-build-v1";
io.every = 25;                 // Complete HMC trajectories, including warmup.
io.max_iterations = 40;       // This invocation's budget, including warmup.
auto prefix = FF::sample_checkpointed(posterior, initial_white, parameters,
                                       hmc, io, true, sink);
// In a later process, rebuild the identical model/data and reuse the same hmc.
io.resume = true;
auto suffix = FF::sample_checkpointed(posterior, {}, {}, hmc, io, true, sink);
```

Use `vary_parameters=false` on every invocation for fixed-parameter inference.
Resume requires empty initial field/parameter arguments and restores both from disk.
Keep the **full** `hmc.warmup`, `hmc.samples`, leapfrog/adaptation controls and RNG
seed unchanged; `max_iterations` limits a job without restarting warmup or changing
the target schedule. `every=0` still saves the final trajectory boundary. A zero
budget can initialize/save or validate/re-save a completed checkpoint.

Returned rows and potentials contain only this invocation's retained suffix.
Acceptance, divergences and completed-iteration/sample counts are cumulative.
Sink draw indices continue from the original chain; each retained draw, including
a repeated state after rejection, is delivered once across successful invocations.
External sink files are caller-managed: checkpoint publication and sink writes are
not one transaction, so crash recovery must reconcile any output after the last
published checkpoint to avoid duplicated rows.

## What is saved

The shared checkpoint codec stores rank-local white coordinates, joint scalar
parameters, current potential/gradient, RNG/distribution states, step-size
adaptation, scalar mass-adaptation windows/accumulators and frozen metric parameters.
PM still uses an identity white metric. Scalar parameters remain replicated while
the white field uses its IC layout, independently of the observation mesh.

This checkpoints **inference at complete HMC trajectory boundaries**. It does not
save an unfinished leapfrog trajectory or a PM time-evolution workspace/particle
state. PM workspaces are recreated; the saved potential/gradient are checked against
a fresh evaluation before continuation, and retained for the exact numerical path.
The internal adjoint midpoint checkpoint budget is a separate concept.

## PM target identity

The PM adapter extends the existing local per-rank target fingerprint with:

- Both IC and force/observation meshes and ownership, backend, copied domain edges,
  process grid, rank count/rank, actual thread count and precision; MPI/OpenMP build
  mode flags are also encoded for PM.
- Flat matter/Lambda background Omega_m and the complete ordered scale-factor list.
- Versioned ZA/lattice, equal-mass CIC deposition/gather, continuous spectral force,
  DC/component-Nyquist handling, no-deconvolution, KDK/background-integration,
  discrete-adjoint/replay and identity-metric conventions.
- The adjoint replay checkpoint budget, conservatively treated as numerical
  compatibility metadata.
- Existing configuration cutoffs/box settings, all scalar prior metadata, Fourier
  modes/weights, normalized likelihood volume/data, compiler/standard-library tag
  and the caller's versioned problem ID.

Identity encoding performs **no collectives** inside its local metadata phase.
`CHECKPOINT::ForwardIdentity<Forward>` is the customization point; the default
adapter preserves existing non-PM fingerprint bytes. The shared grid/domain
metadata helper is also local-only. PM's Cartesian survey factory remains deleted:
checkpoint support does not enable an observation adapter that confuses N and M.

Opaque spectrum callbacks cannot be fingerprinted automatically. **Version their
physics and external inputs, and relevant source/build changes, in `problem_id`.**
The saved-point force/potential check is an additional guard; agreement at one
state cannot establish that two arbitrary callbacks define the same target.
Future changes to fixed PM numerical conventions must bump the adapter's PM law tag.

## Compatibility and files

Restart requires the same rank count, backend, process-grid/domain edges, thread
configuration, precision, target, HMC schedule and compatible build. Changed-rank
or changed-layout redistribution is not implemented. The existing manifest/shard
format is reused; existing linear/LPT/survey fingerprints and file schemas are
unchanged. There were no supported PM checkpoints before this adapter.

Checksummed rank shards are written to a new generation. `CURRENT` is published
only after all shards finish; a failed write leaves the previous generation usable.
Older and failed/orphaned generations are retained; there is no automatic garbage
collection. Normal shared-filesystem and existing checkpoint durability assumptions
apply. This does not add PM survey/tracer/RSD likelihoods or cosmological derivatives.

## Build and run

```sh
make -C FML/FieldLevel/example particle_mesh_checkpoint USE_MPI=false
./FML/FieldLevel/example/particle_mesh_checkpoint /path/to/chain start 40
./FML/FieldLevel/example/particle_mesh_checkpoint /path/to/chain resume 40
```

For pencils use a clean/separate build with `USE_MPI=true USE_PENCIL_GRID=true`,
an explicit compatible `MPIRUN`, and the same `FML_PENCILS`/thread configuration on
all invocations. The example uses N=4/M=8; choose a process grid that fits both.

The registered regression target runs prepare/resume as separate processes:

```sh
make -C FML/FieldLevel/tests particle-mesh-checkpoint USE_MPI=true NP_LIST="1 2 4"
```

MPI builds test both slabs and pencils, including unequal pencil boxes. The bounded
fixture checks full retained fields and scalar rows, adaptation, cumulative results,
nonzero seed high bits and sink indices, rather than merely successful file reads.
