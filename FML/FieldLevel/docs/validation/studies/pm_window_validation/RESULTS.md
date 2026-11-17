> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Explicit CIC force compensation results (15 November 2026)

**The opt-in compensated force substantially reduces dense-grid dependence in
this controlled fixture.** The default remains unchanged. Independent dense-DFT,
adjoint/posterior derivative, backend, replay and checkpoint checks pass.
Three raw scientific criteria remain failed: the original uncompensated
noise-scaled density criterion, and the raw joint-gradient criterion for both
force conventions. All failures are retained in [summary.json (archived)](../../../ARCHIVE.md).

[The protocol](README.md) fixes strong91, 122 initial directions and 256 observed
real degrees. Numerical forces and final observation windows are distinct:
compensation changes the dynamics but leaves the reported CIC density unchanged.
The common-window diagnostic smooths both outputs to the same M32 assignment
window; it does not change the implemented likelihood or its gradients.

## Dense-grid comparison

| M32 vs M64 | Uncompensated | CIC-compensated |
| --- | ---: | ---: |
| Raw density difference, N128/32 steps | 3.7749% | 1.7680% |
| Raw discrepancy / noise SD per degree | 0.106306 | 0.050107 |
| Common-window density difference | 2.0676% | 0.03089% |
| Common-window discrepancy / noise SD | 0.057312 | 0.000862 |
| Raw full projected gradient difference, N64/32 steps | 75.0980% | 34.9141% |
| Raw field-only gradient difference, N64/32 steps | 17.4101% | 8.3111% |

Raw field/noise criteria now pass for compensation, but the full gradient still
fails the unchanged 15% threshold. Spectrum-amplitude derivatives remain
sensitive: at M32/M64 they are -26.99/+11.36 without compensation and
+0.059/+18.84 with it. The likelihood uses raw mesh density and the same frozen
M32 mock, so different force grids also define different observation windows.
The small common-window density correction does not certify joint inference.
The gradient experiment remains at finite N64; its particle/gradient calibration
is not replaced by the N128 forward check.

Compensated N96→N128 particle refinement at M64 changes raw density by 0.1073%
(0.003040 noise SD). Its 32→64-step correction at N128/M64 is 0.01220%
(0.000346 noise SD). Both pass the retained thresholds. The references are finite,
with one realization and fixed bands/background; no universal order or continuum
solution is claimed.

*Archived figure: Force and observation-window comparison (`pm_window_validation/figures/window_comparison.png`). See [archive guide](../../../ARCHIVE.md).*

## Independent force reference and sparse-load limitation

The native tiny-kick and independent NumPy CIC/full-FFT/gather calculations agree
at refined steps. At dense N64, the longitudinal response at M8/M16/M32 changes
from 0.872222/0.966711/0.991973 to approximately **1.000546** at all three meshes.
The continuum linear projection is 1. The declared dense M16/M32 improvement and
<1% response-error checks pass. This projection control does not establish
arbitrary nonlinear/vector-force or high-cutoff accuracy.

Compensation is **not a cure for sparse loads**. At N8/M64, the projected response
changes from about 3.02 to 53.65 (fluid value 1), and the compensated relative RMS
force error is about 182. The inverse window amplifies alias/lattice errors.
The setting remains explicit and opt-in; the original default is retained.

This amplification also exposed finite-kick error in the original probe. For
compensated N8/M64, native/NumPy statistic discrepancies at da=1e-8/1e-9/1e-10
are 2.29e-4 / 2.44e-6 / 6.56e-8. The coarse 1e-8 check and all-step shrinking
check remain failed in the raw force report. After adding the third step,
both refined kicks pass the **unchanged** 2e-5 agreement and shrinking-response
criteria. The largest accepted discrepancy across cases is 2.44e-6, versus
4.55e-7 for the unchanged default probe. All coarse/fine rows and checks are
retained in the summary and raw CSVs; no failed coarse row is erased or called
converged. The original failure/logs are also archived in the local artifact.

## Discrete derivatives and restart

Compensated full-posterior field and mixed field/scalar directional checks at
N32/M16/16 steps pass multiple refined differences. Best refined scaled errors
are 9.68e-10 and 1.60e-9. The independent value-only calculation does not call
the adjoint. This establishes consistency of the implemented target.

*Archived figure: Compensated posterior derivatives (`pm_window_validation/figures/gradients.png`). See [archive guide](../../../ARCHIVE.md).*

The force multiplier is real and even, so its transpose uses the same multiplier
with the existing skew-adjoint force sign. Unit tests cover full-output cotangents,
spectrum/field derivatives, sparse routing and checkpoint-budget replay. An
independent dense CIC/direct DFT EdS trajectory checks the compensated force.
Normalized likelihood, noise and box/P/volume scaling are tested too. An initial
scaled-units test fixture accidentally reset the window to the default; it was
corrected to carry the chosen settings, without changing tolerances.

Fresh-process compensated fixed/joint HMC continuation is bitwise exact on serial,
MPI 1/2/4 slabs/pencils (including unequal boxes), and serial ASan/UBSan. The
registered checkpoint target runs default and compensated fixtures. The force
window is fingerprinted even at a zero field/data state where values and forces
are indistinguishable. New compensated targets use law tag 2. Default tag-1 bytes
are preserved: an old PM checkpoint resumes exactly, and default native case
fields/gradients/FD/metadata plus force-probe CSVs remain bitwise unchanged.
This PM law tag is independent of the shared checkpoint file/identity schema.

## Actual validation scope

Clean Clang17/O2 serial/Open MPI/OpenMP and Clang20/O1 serial ASan/UBSan builds
pass, with FFTWGrid bounds checks. The final module matrix has 33 native process
invocations; the registered serial checkpoint recipe and additional default/MPI
control checks also pass. MPI module checks use two OpenMP threads. The explicit
science campaign uses four pencil ranks and OMP1: eight forward, four full-gradient
and one full-posterior FD trajectory all complete within 300 seconds each.
Native linear probes use a separate serial binary. No new MAP/HMC scientific
chains, full-library CI, optional GSL/NBody comparison or production memory/scaling
benchmark is run. The inherited unused-capture warning in `FieldLevelBAO.h` remains.

## Remaining work

A fixed, explicit PM observation response with a matching pullback/identity is
next, so changing the force mesh need not change the physical observation window.
Then repeat score refinement with adequate particle sampling and validate actual
posterior behavior. Compensation remains an experimental numerical choice pending
more fields, amplitudes, cutoffs, independent references and noise/model-mismatch
calibration. Prior unconverged MAP and production PM tracer/RSD/survey/cosmological
extensions remain open; item #52 is not complete.
