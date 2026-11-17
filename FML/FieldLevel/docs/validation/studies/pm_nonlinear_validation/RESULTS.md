> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Many-mode PM results (15 November 2026)

**The wider-band experiment exposes remaining force-grid sensitivity.** The
implementation/backend and directional derivative checks pass. Particle/time
finest-correction checks pass for all three fixtures, while all three force-grid
projected-gradient checks fail the unchanged 15% limit. Both stronger fixtures
also exceed 0.1 noise SD per observed degree. These failures are retained.

[The protocol](README.md) fixes 122 active initial directions, 256 observed real
degrees, two initial-field seeds and three strength/time fixtures. The noise seed
is shared. This is not an independent-mock coverage ensemble. The force law,
adjoint, sampler and optimizer are unchanged. The [earlier dense 18-mode results](../pm_oversampling_validation/RESULTS.md)
remain valid for their smaller tested band and different realization.

## Finite-reference refinement

| Fixture | Refinement | Density change | Full projected gradient change | Density / noise SD per degree |
| --- | --- | ---: | ---: | ---: |
| weak91 | particle | 2.2865% | 2.9058% | 0.01968 |
| weak91 | force | 2.8248% | 16.2668% | 0.02757 |
| weak91 | time | 0.0020% | 0.0106% | 0.00002 |
| strong91 | particle | 0.5879% | 1.1270% | 0.01406 |
| strong91 | force | 3.7699% | 75.0980% | 0.10615 |
| strong91 | time | 0.0106% | 0.1859% | 0.00029 |
| strong173 | particle | 0.4393% | 1.1279% | 0.01290 |
| strong173 | force | 3.7257% | 60.5880% | 0.12782 |
| strong173 | time | 0.0125% | 0.3412% | 0.00041 |

Particle comparison is N48 vs N64 at M16/16 steps; force comparison is M32 vs M64
at N64/32 steps; time comparison is 32 vs 64 steps at N64/M32. Every family has
its own finite reference, with the same refined dataset within each fixture.
The references are not exact continuum solutions. In particular the M64 force
reference has N64 particles (one particle per force cell); particle refinement at
M64 is still needed before interpreting it as an adequately sampled continuum
reference. This campaign establishes sensitivity, not which force grid is correct.
Additional N32 particle and
M16/32 comparisons remain in [summary.json (archived)](../../../ARCHIVE.md) and [convergence.csv (archived)](../../../ARCHIVE.md).

*Archived figure: Many-mode convergence (`pm_nonlinear_validation/figures/convergence.png`). See [archive guide](../../../ARCHIVE.md).*

At N64/M32/32 steps the final mesh-density RMS is 0.156, 0.497 and 0.641 for
weak91, strong91 and strong173. These are measured field strengths, not a claim
that shell-crossing accuracy is established. A ZA start at a=0.2 remains part of
the fixed numerical model.

The full projected gradient includes 122 initial-field directions plus spectrum
amplitude and noise. The scalar amplitude derivative is especially sensitive:
for strong91 it changes from -26.99 at M32 to +11.36 at M64. The field-only
gradient change is also appreciable (17.41%). Thus the large full-gradient change
is not solely a comparison of noise scalars. These are scores at one fixed true
field and nuisance setting; they do not directly measure cosmological parameter
bias or posterior coverage.

*Archived figure: Force-grid shell-power response (`pm_nonlinear_validation/figures/force_power.png`). See [archive guide](../../../ARCHIVE.md).*

This experiment demonstrates finite discrete force/assignment sensitivity. It
does not by itself identify an optimal kernel correction. Assignment-window or
Green-function changes need an explicit numerical-law study, matched-field
references and derivative tests, followed by checkpoint-law versioning if adopted.

## Derivative checks

At N32/M16/16 steps, field and mixed field/scalar probes pass multiple refined
central-difference steps for both weak91 and strong91. Best refined scaled errors
are 2.04e-7 / 3.16e-7 for weak91 and 4.17e-11 / 5.08e-11 for strong91.
The value-only finite differences do not call the adjoint. This establishes
consistency with the implemented discretization, separately from its discretization sensitivity.

*Archived figure: Many-mode directional derivatives (`pm_nonlinear_validation/figures/gradients.png`). See [archive guide](../../../ARCHIVE.md).*

## Reconstruction from refined data

The revised **12-iteration** runs all lower the posterior objective and pass the
declared recovery checks, but **none reaches the gradient tolerance**. All three
have `MaximumIterations` status. These are improving iterates, not converged MAPs.

| Fixture / particle N | Initial error / zero-field baseline | Initial-field correlation | Projected gradient norm | Residual / noise SD per degree |
| --- | ---: | ---: | ---: | ---: |
| weak91-16 | 0.5854 | 0.8163 | 0.40746 | 0.8615 |
| weak91-32 | 0.5598 | 0.8340 | 0.06759 | 0.8551 |
| strong91-32 | 0.2902 | 0.9690 | 0.78666 | 0.7599 |

Two original 80-iteration attempts were still unfinished after 455 seconds of
line-search work and were interrupted; a briefly queued strong-field attempt
was also interrupted. Their partial outputs/logs are retained. The work budget
was reduced to 12 before these completed fits. Convergence/recovery criteria were
unchanged. The aborted runs supply no converged estimates.

Analysis returns nonzero after saving all reports: **five spatial-accuracy criteria
and three optimizer-convergence checks fail**. All consistency and declared recovery
checks pass. These separate categories are retained in [summary.json (archived)](../../../ARCHIVE.md).

*Archived figure: Initial-field coefficient reconstruction (`pm_nonlinear_validation/figures/reconstruction.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Initial density slices (`pm_nonlinear_validation/figures/field_reconstruction.png`). See [archive guide](../../../ARCHIVE.md).*

All fits hold spectrum/noise/background fixed and start from the same unrelated
small seed-300 field. Data are generated at N64/M32/32 steps, while fitted models
use M16/16 steps and N16 or N32. Coefficient relative error and field correlation
use the orthonormal initial basis and a zero-field error baseline. Images render
the initial density extrapolated to a=1 on a common 32x32 z=0 slice. The optimizer
status and projected gradient remain explicit. A fit is not a posterior sample,
a global-optimum proof, or a measurement of uncertainty/coverage.

## Backend and input-safety evidence

Clean Clang17/O2 serial and Open MPI slab/pencil builds, plus Clang20/O1 serial
ASan/UBSan, pass with FFTWGrid bounds checks. MPI replay uses N16/M16/two PM steps and two OpenMP threads. The full
resolution/reconstruction campaign is serial.
Eleven native engineering invocations cover serial, MPI 1/2/4 slabs and pencils,
sanitizer, incompatible data identity, nonfinite controls, and an MPMD run with
deliberately inconsistent rank controls. All expected successes/rejections pass.
The latter rejects before model construction with inconsistent replicated metadata.

Forward, gradient and three-iteration MAP replay differences are <=1.21e-13.
Independent NumPy direct-DFT initial normalization differs by 3.06e-16. All
backend directional checks pass. The short MAP replay is an engineering check,
not a converged reconstruction. [replay_summary.json (archived)](../../../ARCHIVE.md) retains
checks and errors; local review artifacts retain exact native commands/logs.
ASan leak detection is disabled. The compiler emits an inherited FFTWGrid lambda
capture warning. No full-library CI or optional GSL/NBody build is run.

## Next work

The wider-band force sensitivity and any reported optimizer limit remain open
under [PM-S1](../../../KNOWN_ISSUES.md). Next establish a force reference with controlled particle sampling, then evaluate
explicit force/assignment-window choices before claiming joint-parameter accuracy.
Separate optimizer conditioning/convergence work may be needed for stronger-field
MAP. Broader physical-mode/amplitude tests, independent/refined-data posterior
sampling, repeated-mock coverage, production scaling and PM tracer/RSD/survey
integration remain open. This validation slice does not complete item #52.
