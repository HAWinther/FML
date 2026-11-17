> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# TSC PM: bounded reconstruction result

16 November 2026. [Protocol](README.md), [assignment API](../../../PARTICLE_MESH.md#optional-tsc-assignment).

## Outcome

The opt-in TSC model **converges in 16 iterations** with raw gradient
**5.6502016e-06**, below the unchanged **2e-5** tolerance. Status is
Converged. All consistency, recovery, truth-point derivative and convergence
gates pass (`summary.json: passed=true`). CIC remains the default.

| Assignment | Iterations/status | Raw max gradient | Predictions incl. references | Pullbacks | MAP seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| CIC | 98 / LineSearchFailed | 0.00013902008 | 792 | 99 | 297.75 |
| TSC | 16 / Converged | 5.6502016e-06 | 54 | 17 | 85.40 |

*Archived figure: Convergence and calls (`pm_tsc_validation/figures/comparison.png`). See [archive guide](../../../ARCHIVE.md).*

Each assignment uses its **own matched reference mock**. The physical initial
modes, seeds, power, noise realization convention, common observation band/window,
initialization, nuisance parameters, integration schedules, meshes and optimizer
controls are the same. The target values and data signals differ because the
assignment changes the discrete forward model. This is not an identical-target
optimizer benchmark. Times are indicative under uncontrolled concurrent host
activity; actual call counts include all reference predictions.

## Recovery and numerical checks

Initial coefficient error is 0.268390, correlation
0.964640, and fitted data residual 0.783474
noise standard deviations. The noiseless truth forward mismatch between the fit
and finite reference resolutions is 0.006609
noise standard deviations (CIC was 0.070367
in its matched experiment). The reference is N128/M64/32 steps; the fit is
N32/M16/16 steps. These are finite-resolution measurements, not a continuum test.

*Archived figure: Initial field recovery (`pm_tsc_validation/figures/initial_fields.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Observed fields (`pm_tsc_validation/figures/observed_fields.png`). See [archive guide](../../../ARCHIVE.md).*

Both truth-point finite-difference gates pass with best refined scaled errors
7.05e-10 (white direction) and 1.73e-09
(white plus scalar direction). Every accepted stable change satisfies Armijo;
actual predictions = 21 value trials +
17 full gradients + 16
references. The optimizer keeps at most five history pairs; retained vector
payload is 2.501 MiB summed over ranks.

*Archived figure: Truth-point derivatives (`pm_tsc_validation/figures/derivatives.png`). See [archive guide](../../../ARCHIVE.md).*

## Implementation and validation

TSC uses one shared periodic 27-node weight/derivative stencil for PM deposition,
force gather, final matter density and both source/gather pullbacks. Optional
matching force compensation removes the two product-sinc-cubed windows. The fixed
common CIC observation response divides by the selected fine assignment window.
Assignments and compensation must agree across MPI ranks. A distinct checkpoint
identity rejects changed assignment even at an indistinguishable zero state.

Thirty targeted regression invocations pass in serial, MPI1/2/4 and ASan/UBSan
builds. They cover an independent cardinal kernel, mass/derivative partitions,
first-derivative continuity and finite differences at TSC support changes and
periodic seams, full PM output/spectrum adjoints with/without compensation,
checkpoint replay budgets, posterior gradients and volume scaling, independent
Fourier-window quadrature, signal/noise ordering and slab/pencil consistency.
The kernel test has 359 checks per invocation. MPI assignment mismatches reject.

Thirty fresh-process checkpoint prepare/resume invocations pass for CIC, raw TSC
and compensated TSC, on serial/slab/pencil/sanitizer variants. An old CIC checkpoint
also resumes with this build and bitwise trajectory replay. Fourteen default
CIC numerical output files remain bitwise unchanged. TSC data declarations reject
CIC inputs. The shared reconstruction analyzer preserves the earlier CIC numerical
results and its failed convergence status. Full-library CI was not run; sanitizer
leak detection was disabled.

A preliminary run used 16 reference steps; it is retained only in local review
artifacts and excluded from this result. The reported experiment uses the correct
32 reference steps throughout. Broader seed/amplitude/refinement and posterior
calibration remain deferred. One converged fixed-parameter MAP is not evidence
of calibrated uncertainties, robust convergence for all realizations, or survey
inference support. TSC costs more per force evaluation (27 nodes versus eight).
