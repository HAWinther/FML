# Slab/pencil linear inference — step 1 results

14 November 2026. This extends the periodic distributed linear model; it does not
port distributed LPT, survey observations, checkpoint layouts or the serial BAO
backend. See [the API guide](../../FRAMEWORK.md#distributed-linear-inference-stage-4a-12-november-2026).

## Cross-layout verification

`test_framework_linear_grid_mpi.cpp` instantiates both the default slab alias and
`DistributedLinearModelT<PencilGrid<3>>`. Production models retain only local cells
and retained local modes. Small full-field arrays are assembled **only in tests**
to compare values by canonical global cell/wavevector, independent of storage order.

| MPI ranks | Collective checks passed | Pencil process grids tested on N=8 |
| --- | --- | --- |
| 1 | 42 | 1x1 |
| 2 | 61 | 1x2, 2x1 |
| 4 | 77 | 1x4, 2x2, 4x1 |
| 6 | 76 | 2x3, 3x2, 6x1 |
| 8 | 91 | 2x4, 4x2, 8x1 |

Each layout is checked against an independent direct DFT, field/scalar central
finite differences, a Fourier-weighted adjoint dot product and the exact linear
conditional MAP. Nonidentity Fourier responses are tested against physical modes
and their adjoint finite differences. Counter-based IC and HMC momentum draws are
checked against their specified global-ID streams, including nonzero high iteration
bits. A short nonadapted field HMC chain exercises the generic Execution hooks.
One-rank state errors and cross-owner Hermitian violations fail collectively.

Comparisons cover seeded fields, momenta, predictions, noisy mock modes, posterior
potential, field/scalar gradients, conditional mean/log marginal, Fourier field
metric, initial-density reconstruction and the short HMC endpoint. White noise and
momenta match **exactly**; the maximum comparison error over the other quantities
is **4.10e-16**, with `abs(a-b)/max(1,abs(a),abs(b))` scaling. On 6 ranks N=8 cannot
use FFTWGrid, so the first pencil result is the reference for other pencil layouts;
on 1/2/4/8 ranks the reference is FFTWGrid. No general bitwise FFT/HMC agreement is
promised, and the four-sample chain is not a new posterior calibration experiment.

N=10 tests use unequal cell-aligned x/y partitions where possible, including a
4-rank mesh not divisible by the total rank count. On 6 and 8 ranks an N=4 model
runs with **more ranks than mesh slices**, using 3x2 and 4x2 pencils respectively.
The model-domain snapshot is also checked by constructing a fresh workspace after
changing the global default domain.

Negative cases reject off-cell edges, rank-inconsistent domain types/edges and
process grids with empty Fourier partitions before FFT construction. Unsupported
2D grid types fail with an explicit compile-time diagnostic. Even meshes and
nonempty real/Fourier domains remain required; this step does not enable the full
odd-mesh capability of PencilGrid.

## Toolchain and compatibility checks

- Clang 17, Open MPI and OpenMP (2 threads), double precision, with FFTWGrid bounds
  checks: all new 1/2/4/6/8-rank runs pass. FFTW uses a matching Open MPI build;
  FFTW thread support is disabled.
- AddressSanitizer and UndefinedBehaviorSanitizer: final generic-grid tests pass
  at 1 and 2 ranks (42 and 61 checks). External FFTW/MPI libraries are not rebuilt
  with instrumentation; leak detection is disabled for the MPI runtime.
- Existing linear regression: Clang serial and GCC 14 serial pass 21 checks;
  Clang MPI/OpenMP at 1/2/4 ranks passes 21/46/92 checks, including the longer
  Gaussian-projection HMC test, joint scalar synchronization and malformed inputs.
- Existing distributed 1LPT/2LPT regression passes at 1/2/4 ranks (44/80/160 checks),
  exercising the unchanged slab LPT backend with the shared layout/linear alias.
- Existing slab survey checkpoint prepare/resume tests pass in separate processes
  on 1/2/4 ranks (prepare: 13/26/52; resume: 60/120/240 reported checks), including
  Gaussian/Poisson continuation and target/input compatibility rejection. This
  verifies slab compatibility; it is not pencil checkpoint validation.
- The registered `framework-linear-grid-mpi` Make target rebuilds and passes on
  4 ranks. Clean/rebuild when changing compiler, MPI, precision or OpenMP flags.

The existing regressions report their historical mixture of collective and summed
rank-local checks, so counts across rank counts are not directly comparable. No
strong-scaling, peak-memory or performance improvement is claimed. Constructor
geometry discovery uses a temporary backend grid, released before the Hermitian
validation workspace is allocated. No full-field gather is introduced in model
prediction, pullback, random draws or HMC execution.

## Reproduction

```sh
make -C FML/FieldLevel/tests clean
make -C FML/FieldLevel/tests framework-linear-grid-mpi \
  USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST='1 2 4 6 8'
```

Set `CXX_MPI`, `FFTW_INCLUDE` and `FFTW_LIB` to matching local installations.
The test constructs its process grids itself; it does not require a global pencil
configuration or modify the default domain permanently. The existing aliases
`DistributedLinearModel` / `DistributedLinearPosterior` remain slab based, and the
legacy `local_start` / `local_width` layout members remain x-only aliases.
