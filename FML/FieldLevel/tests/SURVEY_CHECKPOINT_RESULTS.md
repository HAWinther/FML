# Survey checkpoint/restart validation

Recorded 13 November 2026. This extends the same-rank periodic checkpoint API to
explicitly registered likelihoods; [../SURVEY.md](../SURVEY.md#same-rank-survey-checkpointrestart)
describes the API and identity-schema compatibility boundary.

## Cases and reference comparisons

`test_survey_checkpoint_mpi.cpp` exercises Cartesian Gaussian and positive-link
Poisson likelihoods with distributed linear, ZA and EdS 2LPT forwards. Each of
these six combinations has:

- joint field/scalar sampling with 120 warmup and 20 sampling trajectories,
  three leapfrog steps and target acceptance 0.9;
- disk restarts after per-call budgets 27, 33, 55 and 10, crossing mass-adaptation
  windows and ending at iteration 125 (five retained draws);
- a fresh-process restart to iteration 140, comparing the complete local final
  state, remaining rows and potentials, scalar mass, step size, cumulative
  acceptance/divergences and global field-sink indices **bitwise** with an
  uninterrupted reference in the same build/rank configuration;
- separate fixed-scalar 30-warmup/12-sample chains, stopped at warmup iteration
  10 and resumed in another process, also matching uninterrupted results bitwise.

Fresh-process rejection checks cover one-rank mask, selection and active data,
relative variance, count or exposure changes, replicated geometry relabeling,
Poisson link-offset changes and a different likelihood family. Masked inputs are
canonicalized: ignored data/variance/count/exposure placeholders can change
without changing the target identity or the resumed state. Model/caller-code
identity, rank/layout, controls, corruption, generation and cooperative-write
safety remain covered by the periodic checkpoint regression.

A prior-family fixture changes Gaussian to uniform with otherwise identical
metadata and starts at the shared prior center. Cached potentials agree there;
the prior forces also agree by construction. The new identity rejects the
changed family rather than relying only on the cached force/potential check.
A bounded streaming-hash test matches the standard codec checksum across
multiple 4 KiB boundaries, including a long string, vectors and uint64 values.

## Recorded final profiles

| Profile | Prepare checks | Fresh-process resume checks |
| --- | --- | --- |
| Apple Clang serial | 13 | 60 |
| GCC 14 serial | 13 | 60 |
| Clang ASan + UBSan | 13 | 60 |
| MPI one rank | 13 | 60 |
| MPI two ranks | 26 | 120 |
| MPI four ranks | 52 | 240 |
| MPI two ranks, two OpenMP threads, NO_AUTO_FML_SETUP | 26 | 120 |

A two-rank attempt to load the one-rank checkpoints is rejected (14 checks).
Both Gaussian and Poisson example chains stop after 40 warmup trajectories and
resume in another two-rank process to finish the full 60-warmup/80-sample
schedule. Both report 140 completed trajectories, 80 retained draws and zero
sampling divergences. The example is a restart demonstration with immutable
analytic observations, not posterior calibration.

The default periodic checkpoint regression passes on two ranks (24 prepare and
84 resume checks) under identity schema 2. Survey likelihood/adjoint/MAP/HMC
regressions, including the large-input cases below, pass 55 checks per rank under
Clang, GCC and ASan/UBSan and at 1/2/4 MPI ranks. These profiles use FFTW double
precision. The isolated Makefile target runs prepare and resume as separate
processes; multi-node users must set TMPDIR to a shared filesystem.

## Legacy identities and explicit registration

An actual identity-schema-1 checkpoint is written with the committed
`c5f86be` version of `Checkpoint.h` and the same compiler/stdlib/other code, then
read using the new target identity. The reader rejects it specifically for a
changed target identity. This checks fail-closed compatibility, not just a
compiler/build-tag mismatch. The shard/manifest/HMC state framing is unchanged;
older chains remain usable with their original code/build. No automatic exact
trajectory migration is provided.

A compiler-negative test confirms that an unsupported custom likelihood cannot
silently use the periodic Gaussian fingerprint: its primary `LikelihoodIdentity`
registration fails with the explicit specialization-required diagnostic. The
nonzero compiler exit is expected and recorded separately from runtime failures.

## Large-proposal validation issue found by warmup

GCC's normal draws/warmup exposed an existing observation-validation issue:
round-tripping a large valid Hermitian signal through FFTs can leak roundoff into
nominally zero coefficients. Scaling each coefficient's tolerance independently
could misclassify a valid proposal as malformed input and abort HMC. The survey
observation now scales that round-trip comparison by the communicator-wide input
norm; nonfinite FFT results are numerical proposal failures. A 10^12-amplified
analytic wave must synthesize correctly, while a large deliberately broken
conjugate pair must still fail collectively. The new tests pass in all recorded
compiler/MPI profiles; normal adjoint/likelihood gradient checks are retained.
No probability-law or sampler formula changed.

Development also caught a test reference-reader assumption: fixed-scalar HMC has
no scalar mass block even though its recorder returns the fixed scalar values.
The reader now validates that zero-length block correctly. The final profiles
above supersede these development failures.

## Evidence and limits

Final command records, reference files, logs and example checkpoints are under
`outputs/fml-survey-checkpoint/verified/` in the Codex workspace. `validate.py`
reproduces the serial/compiler/sanitizer and MPI/example/regression profiles.
`development-validation/` and root development logs are superseded.

The API preserves the existing same-rank/layout/build/thread, immutable full
schedule and one-writer/shared-atomic-rename filesystem contracts. It does not
add rank repartitioning, sample-schedule extension, automatic pruning, fsync
power-loss durability or transactional reconciliation of external output sinks.
Likelihood input hashing is bounded by small chunks; existing model-metadata
and sampler-state encoding buffers still scale with local modes/cells. No
checkpoint I/O scaling or full-memory reduction benchmark is claimed. Arbitrary
callback physics/code/external inputs still require caller-versioned problem_id.
This is restart engineering validation, not production survey posterior coverage.
