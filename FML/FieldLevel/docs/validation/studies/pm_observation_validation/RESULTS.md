> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Fixed PM observation response: bounded results

15 November 2026. See [protocol and reproduction](README.md) for the frozen
strong91 realization, normalization, tolerances and exact numerical setup.

The actual likelihood now observes a common M32 CIC window for both M32 and M64
force grids. The response and its transpose are inside the model, not only in
postprocessing. Noise is added afterward in this fixed observation space.

| Force convention | Density change, N128 | Full projected posterior-gradient change, N64 | Field-only gradient change |
| --- | ---: | ---: | ---: |
| None | 2.06762% | 41.55333% | 9.63820% |
| CIC compensated | 0.0308943% | 0.475618% | 0.326593% |

Changes compare force M32→M64, 32 steps. Density discrepancy is 0.0573123 and
0.000861767 noise SD per real observed degree, respectively. The compensated
case passes the original 5% density, 15% joint-gradient and 0.1-noise criteria.
The uncompensated joint-gradient baseline still fails: aggregate `passed=false`
in [summary.json (archived)](../../../ARCHIVE.md) deliberately retains that failure. Consistency
checks and compensated accuracy checks both pass.

With compensated forces, particle N96→128 at M64 changes density by 0.105892%
(0.00295375 noise SD); 32→64 steps at N128/M64 changes it by 0.0121361%
(0.000338558 noise SD). Field and mixed finite-difference sweeps at N64/M64,
four steps, with a nontrivial M32 response reach scaled errors 2.22e-7 and 5.00e-8.

The preceding raw-observation study found a 34.9% compensated joint-gradient
change. The fixed observation convention reduces it to 0.476% here. Those are
different likelihood conventions, each with its correctly matched gradient;
the comparison does not establish posterior coverage or optimal inference.

*Archived figure: Observed-density convergence (`pm_observation_validation/figures/observation.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Posterior gradient convergence (`pm_observation_validation/figures/gradients.png`). See [archive guide](../../../ARCHIVE.md).*

## Implementation checks

Clean serial and MPI builds, 1/2/4-rank slab/pencil tests, a four-rank/two-thread
hybrid test, bounds checking and ASan/UBSan pass. Tests cover independent triangular
CIC-kernel quadrature, custom negative/zero coefficients, combined responses,
noise after response, raw reconstruction, full directional derivatives, volume
scaling, malformed and non-Hermitian inputs, and owners without retained modes.
Fresh-process fixed/joint checkpoints pass with observed compensated targets on
both backends. Changed responses reject even when zero fields make U/g agree.

[Compatibility checks (archived)](../../../ARCHIVE.md) confirm bitwise legacy native output
and data-identity rejection. Old raw tag-1/tag-2 checkpoints resume bitwise with
the new binary. [Cross-study checks (archived)](../../../ARCHIVE.md) agree with independent
previous postprocessing to better than 2e-17 absolute; dynamics remain unchanged.

An initial checkpoint-test assertion wrongly required every local fingerprint to
change when coefficients changed: empty-mode owners have no changed local
coefficients. The fixture now requires a global change and still exercises the
collective rejection. Original failure logs are preserved in review artifacts.
The initial aggregate launch reached analysis before its script was written;
all eleven native cases completed, and only analysis was rerun afterward.

## Limits

This is one realization at fixed nuisance parameters, finite N128 density and N64
gradient references, not a continuum solution or broad nonlinear calibration.
The frozen data came from the original uncompensated M32 mock; only its declared
window tag changes, not its numerical values. No truth-recovery claim follows.
Sparse compensated dynamics remain poor, and the feature stays opt-in.
No new MAP/HMC scientific chains, survey/tracer/RSD adapter, changed-layout restart,
cosmological derivative or full-library CI run is claimed.
