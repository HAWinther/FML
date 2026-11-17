> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Dense-particle PM results (15 November 2026)

**All declared bounded consistency and finest-correction criteria pass.** Native
N>M is implemented without changing the force law. The original sparse-load
failure remains documented in [the earlier results](../pm_resolution_validation/RESULTS.md).
[The protocol](README.md) defines matched fields, finite references and thresholds;
[summary.json (archived)](../../../ARCHIVE.md) and [convergence.csv (archived)](../../../ARCHIVE.md) retain all checks
and intermediate trials.

| Family | Last trial vs finite reference | Density change | Projected gradient change | Density / noise SD per degree |
| --- | --- | ---: | ---: | ---: |
| Force, N=64 | M=32 vs 64 | 0.7351% | 6.1981% | 0.01425 |
| Particles, M=16 | N=48 vs 64 | 1.0357% | 6.1818% | 0.01941 |
| Particles, M=32 | N=48 vs 64 | 0.5343% | 1.8773% | 0.01029 |
| Time, N=32/M=16 | 16 vs 32 steps | 0.00775% | 0.05189% | 0.000146 |

*Archived figure: Dense-particle convergence (`pm_oversampling_validation/figures/convergence.png`). See [archive guide](../../../ARCHIVE.md).*

All families improve relative to their coarsest trial. These checks accept the
**finest corrections**, not every coarse grid: for example N=32/M=16 differs
from N=64/M=16 by 5.74% in density and 25.70% in projected gradient. Increasing N
also increases the sampled white/IC lattice and particle memory as N^3, although
the active physical Fourier band is fixed. The remaining white directions are
prior-null. This is not a measured speedup or a recommendation to oversample every
production run by a fixed factor.

Field and mixed directional derivatives have best refined scaled errors 1.51e-8
and 5.81e-9. Both pass the unchanged multiple-step stability criteria.

*Archived figure: Oversampled directional derivatives (`pm_oversampling_validation/figures/gradients.png`). See [archive guide](../../../ARCHIVE.md).*

## Independent force diagnostic

The native tiny-kick probe now directly reproduces the previously NumPy-only
N=64/M=32 case: longitudinal response 0.991973, versus continuum linear response 1.
For dense N=64 at M=8/16/32 the responses are 0.872222/0.966711/0.991973. This shows
residual force-window error on coarse meshes even with dense particles. At sparse
N=8 the old M=8/16/32/64 responses 0.900/1.322/1.788/3.023 remain unchanged.

All 33 native/NumPy and shrinking-kick assertions pass, with maximum discrepancy
4.55e-7 in the normalized response/RMS-error statistics. Raw [native (archived)](../../../ARCHIVE.md)
and [NumPy (archived)](../../../ARCHIVE.md) data and [force_summary.json (archived)](../../../ARCHIVE.md) are saved.
The probe tests a tiny longitudinal displacement, not nonlinear shell crossing.

*Archived figure: Independent and native dense-force diagnostic (`pm_oversampling_validation/figures/force_response.png`). See [archive guide](../../../ARCHIVE.md).*

## Engineering validation

Clean builds and **31 process invocations** pass across serial, MPI 1/2/4 ranks
with two OpenMP threads, and serial ASan/UBSan (leak detection disabled). The
forward test adds independent dense CIC/direct-DFT evolution at N=8/M=4 and checks
slab/pencil equality, conservation and invalid cutoff/process geometry. The adjoint
checks full-output cotangents, white/spectrum derivatives and checkpoint-budget
replay. The posterior checks normalized likelihood/noise, spectrum/field derivatives
and volume scaling on the coarser observation mesh.

Fresh-process fixed/joint HMC checkpoint continuation is bitwise exact for N=8/M=4
on slabs and pencils, including unequal boxes and the 2x2 process grid. The short
dense joint fixture checks step-size adaptation, with scalar mass adaptation
disabled; the original fixture retains its mass-window checks. The registered
checkpoint target now runs both fixtures. Its two-rank slab/pencil recipe is also
exercised. An old pre-change serial PM checkpoint resumes bitwise exactly under
the new binary. Existing numerical-law tag 1, target encoding and codec remain
unchanged; newly permitted layouts are already distinguished by their mesh metadata.

Clang 17/O2 supplies serial and Open MPI builds; Clang 20/O1 supplies sanitizers.
FFTWGrid bounds checks are enabled. No optional GSL/NBody test or full-library CI
is run in this slice. Exact commands, logs, source hashes and full diff are in the
local review artifact.

## Remaining scope

This resolves native oversampling support and passes the dense-sampling refinement
experiment for **one weakly nonlinear 18-mode realization**. It does not establish
an exact continuum solution, repeated-mock coverage, high-cutoff or shell-crossing
accuracy, a universal convergence order, cosmological derivatives, or large-job
memory/scaling. Broad [PM-S1](../../../KNOWN_ISSUES.md) calibration remains open. Evaluate
any future force-window or assignment change separately, bumping numerical-law
identity when the law changes. Item #52's tracer/RSD/survey and other planned
extensions remain separate work.
