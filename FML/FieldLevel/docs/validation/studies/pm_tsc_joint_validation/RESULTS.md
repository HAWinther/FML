> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Joint TSC MAP: initial field, amplitude and noise

16 November 2026. [Protocol and gradient identities](README.md).

## Result

The existing public joint MAP converges in **54 iterations**,
with joint stopping norm **9.8491147e-06 < 2e-5**. This norm is
max(raw field gradient, abs(scalar gradient) times its Gaussian scale). Fresh
endpoint replay gives field max 3.1102103e-06 and scaled scalar max
9.8491147e-06; both are below tolerance. All consistency,
physical-recovery, derivative and convergence gates pass (`passed=true`).

| Quantity | Joint MAP |
| --- | ---: |
| Log power amplitude A (injected 0) | 1.95284891 |
| Power amplitude multiplier exp(A) | 7.048740 |
| Noise power (injected 1e-4) | 6.016334e-05 |
| Noise power / injected | 0.601633 |
| Physical initial-field coefficient error | 0.283386 |
| Physical initial-field correlation | 0.964540 |
| Residual / true noise SD | 0.769468 |

*Archived figure: Parameter path (`pm_tsc_joint_validation/figures/parameters.png`). See [archive guide](../../../ARCHIVE.md).*

## Interpretation: a whitened-coordinate joint mode

**These scalars are joint density modes, not marginalized parameter estimates.**
The amplitude domain is explicitly widened from [-1,1] to [-4,4] while retaining
the Gaussian mean 0 and sigma .5. The fitted mode is interior. Noise keeps its
original Gaussian log prior and support. The frozen TSC data are unchanged.

The spectrum amplitude rescales the physical initial field by exp(A/2), allowing
larger A with smaller white-field coefficients. At exactly stationary field and
amplitude the analytic gradient identity gives A=dot(s,s)/8. Here that right-hand
side is 1.95284959, close to the measured A. The exact
finite-gradient amplitude identity agrees to about 1e-13 across truth, start and
endpoint. The normalized noise identity agrees at the same level; endpoint
chi2=251.935183 for 256 weighted real degrees of freedom.

This verifies the implemented target and explains the high amplitude. It does
not establish recovery of injected scalar parameters or correct uncertainty
coverage. Marginalized amplitude/noise inference requires integration or sampling
of the latent field and calibration, which are outside this bounded MAP study.

## Physical field and residual

Recovery is computed from physical density coefficients using the **fitted**
spectrum amplitude, rather than raw white coefficients. The independent physical
normalization check passes. The fixed-parameter TSC fit had coefficient error
0.268390 and correlation 0.964640;
its residual was 0.783474 true-noise SD. The joint fit
reduces the data residual slightly while giving a slightly larger field error in
this realization. It is not presented as improved scalar/field estimation.

*Archived figure: Physical field (`pm_tsc_joint_validation/figures/initial_fields.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Observed residuals (`pm_tsc_joint_validation/figures/residuals.png`). See [archive guide](../../../ARCHIVE.md).*

## Gradients, work and engineering

All six refined finite-difference gates pass: white, scalar and combined
directions at start and endpoint. Best scaled refined errors range from roughly
7e-11 to 4.1e-9, below the existing 2e-5 gate. No derivative or stopping threshold
was relaxed. Analytic identities include Gaussian scalar priors and normalized
noise, with distributed field contractions and replicated scalars counted once.

*Archived figure: Joint derivatives (`pm_tsc_joint_validation/figures/derivatives.png`). See [archive guide](../../../ARCHIVE.md).*

MAP makes 171 actual predictions and 55
pullbacks: 69 value requests include
7 unsupported proposals without prediction, plus
55 full gradients and
54 reference predictions. The history cap is five.
The accepted parameter path is indexed by full-gradient counters; rejected
values do not become accepted states. MAP wall time was 256.00s,
indicative under uncontrolled host activity; diagnostics are excluded from these
MAP counters and time.

Eight short joint runs pass in serial, slab MPI1/2/4, pencil MPI1/2/4 and
ASan/UBSan. Canonical coefficients, physical fields and observed modes agree;
both scalar parameters update, analytic identities pass, and prediction counters
include unsupported proposals correctly. Fixed mode reproduces the committed
TSC driver's coefficients **bitwise**. Invalid controls and inconsistent MPI
controls reject. All four native variants rebuild with bounds checks. Full-library
CI was not rerun because library/optimizer code is unchanged; sanitizer leak
detection was disabled. Published results replay exactly through the runner.

This adds a validation driver, reports and plots, not a new optimizer or production
prior change. The preceding diagnostics/stable-comparison/TSC work was committed
locally as b3750a7 before this slice. One realization, fixed cosmology, finite
reference and two scalar parameters do not establish a general posterior result.
Broader seed/amplitude/refinement and posterior calibration remain deferred.
