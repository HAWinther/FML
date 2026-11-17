# Slab/pencil surveys and checkpointing — step 3 results

14 November 2026. Cartesian observations now follow the linear/LPT model's grid
backend and copied domain; window callbacks use all global coordinate components.
Checkpoint target schema 3 records the backend, complete real-space partition and
Fourier storage mapping. See [SURVEY.md](../../SURVEY.md) and
[the framework guide](../../FRAMEWORK.md#checkpointrestart-stage-4e-13-november-2026).

## Survey observations and inference

Clang 17/Open MPI/OpenMP (2 threads), double precision and FFTWGrid bounds checks
pass the new cross-layout suite:

| MPI ranks | Reported checks passed | N=8 pencil process grids |
| --- | --- | --- |
| 1 | 88 | 1x1 |
| 2 | 206 | 1x2, 2x1 |
| 4 | 439 | 1x4, 2x2, 4x1 |
| 6 | 556 | 2x3, 3x2, 6x1 |
| 8 | 739 | 2x4, 4x2, 8x1 |

Counts mix collective checks and summed rank-local checks from the existing
independent survey-reference suite; they are not directly comparable across ranks.
Each layout checks analytic synthesis, an independent direct-DFT adjoint,
full-Hermitian dot products, normalized masked/heteroscedastic Gaussian and Poisson
formulas, selection/exposure rules, large Hermitian signals, invalid inputs,
empty footprints, zero rates, underflow and numerical-overflow rejection.
The existing reference routines were extracted into `SurveyTestHelpers.h` and
parameterized on the forward model; the original slab executable retains its tests.

Linear, Zeldovich and 2LPT targets additionally compare observed means/rates,
posterior values, field/scalar gradients and short joint-HMC endpoints. Comparisons
use canonical global cell IDs; full arrays are assembled **only in tests**.
The maximum slab/pencil comparison error is **6.67e-16**, with
`abs(a-b)/max(1,abs(a),abs(b))` scaling. Where slabs cannot use the rank count
(6 ranks on N=8), the first pencil layout is the reference.

Full posterior finite differences include sampled spectrum, bias and noise
parameters: maximum tested field-direction error is **5.87e-9** and maximum scalar
error **6.19e-8**, below the retained 2e-6 acceptance bound. MAP lowers the objective;
short HMC runs establish integration/parity, not scientific mixing/calibration.
Unequal N=10 / density N=20 domains pass both survey likelihoods on 2/4/6/8 ranks.
Observation construction after a global-domain change retains the model domain.
Real-cell packing excludes FFT padding and is checked against all local axes.

## Separate-process checkpoint continuation

`test_checkpoint_grid_mpi` tests all nine combinations of linear/Zeldovich/2LPT
with periodic Gaussian, survey Gaussian and survey Poisson likelihoods. Every case
checks joint and fixed-parameter sampling against an uninterrupted reference, with
splits during warmup and sampling. One periodic linear case uses 120 warmup steps
and mass adaptation; other cases use bounded warmup schedules.

| Layout / MPI ranks | Prepare checks | Fresh-process resume checks |
| --- | --- | --- |
| 1x1 / 1 | 19 | 34 |
| 2x1 / 2 | 21 | 36 |
| 2x2 / 4 | 21 | 36 |
| Unequal 2x2 / 4 | 21 | 36 |

Resumed final fields/scalars, recorded rows/potentials, adapted diagonal masses,
step size, cumulative acceptance/divergences and global sink indices match the
uninterrupted reference **bit for bit**. Fixed-parameter trajectories also match.
Changed fixed inputs reject collectively, while ignored masked values remain
compatible with the same target.

Separate rejection runs (12 checks each) cover changed process-grid shape, changed
y partition edges, pencil→slab backend changes and changed rank count. These test
rejection, not migration. Exact continuation requires the same executable,
numerical environment, backend, rank count, partition and original HMC schedule.

The unequal-partition case exposed an existing assumption in replica validation:
`Codec::state(..., fields=false)` included `n_custom`, the rank-local field length.
It now canonicalizes that value for **replicated metadata only**, while still
checking scalar tails and sampler controls. Persisted shards use `fields=true`,
retain their actual local lengths, and validate those lengths on read. Targeted
positive/negative tests verify all three properties of the replica check; uneven
fresh-process continuation exercises actual shard dimensions.

Identity schema **3** adds all three local starts/counts, explicit backend type,
complete slab/pencil partition edges/process-grid shape and Fourier storage
indices. Descriptor construction contains no MPI collectives. Schema-2 checkpoint
identities require their original code and are rejected under schema 3; shard
framing and full sampler-state encoding remain unchanged. Version callback/code
physics through `problem_id`; compiler tags do not hash the executable.

## Regression and instrumentation

- ASan/UBSan survey tests pass at 1/2 ranks (88/206 checks), and checkpoint
  prepare/resume passes at 1 rank (19/34 checks). External MPI/FFTW libraries are
  uninstrumented; MPI runtime leak detection is disabled.
- Existing survey tests pass with Clang serial and GCC 14 serial (57 checks each),
  and on MPI 1/2/4 ranks (57/114/228 reported checks).
- Existing periodic field-checkpoint prepare/resume tests pass on 1/2 ranks.
- The standalone HMC checkpoint regression passes all 39 checks, including
  corruption, truncation, dimension/option errors and restart equivalence.
- Both registered `survey-grid-mpi` and `checkpoint-grid-mpi` Make targets rebuild
  and pass on 4 ranks.

Use matching MPI/FFTW compiler, headers and libraries; FFTW threading is disabled.
Steps 1–2 inference kernels are preserved. No performance, strong-scaling, new
scientific calibration or cross-layout checkpoint-migration claim is made.

## Reproduction and remaining work

```sh
make -C FML/FieldLevel/tests clean
make -C FML/FieldLevel/tests USE_MPI=true USE_OMP=true survey-grid-mpi NP_LIST='1 2 4 6 8'
make -C FML/FieldLevel/tests USE_MPI=true USE_OMP=true checkpoint-grid-mpi NP_LIST='1 2 4'
```

Set `CXX_MPI`, `FFTW_INCLUDE`, `FFTW_LIB` and `MPIRUN` for matching installations.
The checkpoint target runs prepare/resume in separate processes and creates a
fresh temporary directory per rank count; use a shared `TMPDIR` on multiple nodes.
The next step is frontend/backend selection and the survey calibration driver.
Original serial BAO APIs, general checkpoint repartitioning and production
scientific/performance calibration remain separate work.
