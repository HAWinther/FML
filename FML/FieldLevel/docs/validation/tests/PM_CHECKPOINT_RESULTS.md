# PM checkpoint/restart validation

[ParticleMeshCheckpoint.h](../../../ParticleMeshCheckpoint.h) enables the existing
`sample_checkpointed()` API for periodic PM Gaussian inference. The guide is
[PM_CHECKPOINT.md](../../PM_CHECKPOINT.md). The binary shard/manifest codec and HMC
integrator/state representation are unchanged.

## Continuation reference

`test_particle_mesh_checkpoint.cpp` uses N=4 IC cells/particles, M=8 force and
observation cells, box length 1, P=0.02*exp(log_amplitude), noise power 0.001,
cutoffs 7/15 and scale-factor nodes 0.2,0.4,0.6. Both spectrum amplitude and noise
have bounded Gaussian scalar priors. Mock seeds are 91/92; initial field seed is
300. The sampler seed has nonzero high bits, `(1<<40)+37`.

The joint run has 120 warmup and six retained draws, two maximum leapfrog steps,
step-size and scalar mass adaptation. Split budgets 27,33,55,7 cross the adaptation
windows and end after two retained draws; a fresh process produces the final four.
The fixed-parameter run has six warmup/four retained draws with adaptation disabled;
it stops after three warmup trajectories and completes in a fresh process. These
short chains validate restart mechanics, not posterior mixing or calibration.

Every retained local white field, scalar row, potential and sink index agrees
**bitwise** with uninterrupted public `sample()` execution. Final field/scalar state,
masses, step size, cumulative acceptance/divergences and completion counters also
agree exactly. Saved frozen parameters and mass-adaptation window state are checked.
A completed zero-budget continuation emits no further rows.

| Backend/runtime | Prepare checks | Resume checks | Result |
| --- | ---: | ---: | --- |
| Serial FFTW slabs | 5 | 50 | Pass |
| MPI FFTW slabs, 1/2/4 ranks | 5 each | 50 each | Pass |
| MPI pencils, 1/2/4 ranks | 5 each | 50 each | Pass |
| Serial ASan/UBSan + grid bounds checks | 5 | 50 | Pass |

Pencils use cell-aligned unequal boxes (interior edge 0.25 for two-way axes), so
rank-local IC and force layouts differ in both mesh and ownership. MPI runs use
two OpenMP threads. Resume equivalence is tested within each supported layout,
not as changed-layout continuation.

## Rejection and failure cases

- Background Omega_m, an interior time node at fixed endpoints, time-step count,
  adjoint replay budget, spectrum/noise priors, IC mesh, force/observation mesh,
  box length, initial cutoff and likelihood cutoff changes are rejected.
- The zero-white/zero-data fixture has identical saved potential/gradients after
  changing Omega_m. Its resume is still rejected by explicit PM identity, proving
  the new check does not rely solely on saved-point force validation.
- One-rank data changes propagate collectively. Changed problem/callback version,
  HMC schedule and fixed/joint mode are rejected. An unversioned P=0.022 callback
  change is caught by the saved-force check at this fixture's nonzero state; general
  callback equivalence still requires the caller's explicit problem version.
- Separate invocations reject changed MPI rank count, slab-to-pencil backend,
  pencil domain edges, 2x2-to-4x1 process grid and one-versus-two thread settings.
  Both fixed and joint checkpoints are checked.
- One-rank stored-force corruption, truncated shards and missing shards reject
  collectively. An injected one-rank write error leaves `CURRENT` unchanged;
  restored/previous checkpoints remain usable after the failure fixtures.
- A compile-negative probe confirms PM's one-grid Cartesian survey factory stays
  deleted even when checkpoint support is included.

## Existing checkpoint compatibility and public examples

Checkpoints written by a frozen copy of the original `Checkpoint.h`/`ForwardGrid.h`
serializer are resumed by the new default identity adapter. The existing linear,
ZA and 2LPT suite passes all 42 continuation/failure checks. This tests old-to-new
file/fingerprint compatibility rather than only comparing new hash implementations.
The fixture is regenerated with the original binary before that destructive resume
suite, since the resume/failure tests advance and rewrite its checkpoint history.

The existing serial survey Gaussian/Poisson checkpoint suites pass all 13 prepare
and 60 resume checks across linear/ZA/2LPT models. The actual registered PM Make
recipe passes in serial, and with MPI at two ranks for both slab/pencil branches;
the separate native matrix covers ranks 1/2/4. The new public example builds and
starts/resumes in serial and on four-rank pencils, reaching the expected cumulative
10/420 trajectories after budgets seven and three.

Builds use Clang 17, serial FFTW or matching Open MPI/FFTW-MPI, no GSL, grid bounds
checks and explicit launcher arguments. Sanitizers use Clang 20, ASan/UBSan with
leak detection disabled. Tests include fresh processes and injected I/O failures;
no power-loss/fsync durability, scaling or memory improvement is claimed. Full
library CI was not rerun; targeted checkpoint and survey regression targets passed.

Local command logs, build-source snapshots, frozen original headers/binary,
checkpoint generations/reference fields, negative-compile evidence and the complete
review diff are in `outputs/fml-fieldlevel-pm-checkpoint/`. No statistical chains or
plot results from the earlier PM validation milestones are replaced.

## Limits

Same rank count/layout/backend/thread configuration and compatible build are
required. Changed-rank redistribution, mid-HMC-leapfrog or mid-PM-evolution restart,
automatic generation pruning and PM survey/tracer/RSD likelihoods remain outside
this increment. Arbitrary callbacks and relevant source/build changes must be
versioned in `problem_id`; the compiler tag is not an executable content hash.
