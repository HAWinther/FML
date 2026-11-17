> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Refined-data compensated PM reconstruction: results

15 November 2026. [Protocol and reproduction](README.md). Broader seed/amplitude/refinement checks remain deferred.

New mock: N128/M64/32 steps. Fit: N32/M16/16 steps. Both use compensated forces and a fixed M16 CIC observation window. Noise is added after the response. All nuisance parameters are fixed.

| Iteration budget | Accepted iterations | Status | Initial error / zero-field baseline | Initial-field correlation | Data residual / noise SD | Projected gradient L2 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | MaximumIterations | 0.419369 | 0.936416 | 1.236219 | 102.135 |
| 4 | 4 | MaximumIterations | 0.280400 | 0.961134 | 0.805116 | 15.5632 |
| 12 | 12 | MaximumIterations | 0.269720 | 0.964268 | 0.788405 | 1.38964 |
| 24 | 24 | MaximumIterations | 0.269553 | 0.964336 | 0.788088 | 0.0690033 |

The final raw cell maximum gradient is 0.00219002, against tolerance 2e-5; status is **MaximumIterations**. Objective changes from -64.311959 to -1049.632418. Starting data residual is 2.950687 noise SD per real observed degree. Final signal error against the noiseless refined signal is 0.678509 noise SD. These residuals are diagnostics, not posterior-predictive calibration.

Recovery requires initial coefficient error <0.8 and correlation >0.7, retaining the previous thresholds. Convergence is reported separately. The analyzer saves failures and returns nonzero if any criterion fails. See [machine-readable summary (archived)](../../../ARCHIVE.md) and [endpoint table (archived)](../../../ARCHIVE.md).

*Archived figure: Optimization endpoints (`pm_reconstruction_validation/figures/optimization.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Initial-field reconstruction (`pm_reconstruction_validation/figures/initial_fields.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Observed-band reconstruction and residual (`pm_reconstruction_validation/figures/observed_fields.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Coefficient reconstruction (`pm_reconstruction_validation/figures/coefficients.png`). See [archive guide](../../../ARCHIVE.md).*

## Validation and provenance

Clean Clang17/O2 serial, MPI slab/pencil and Clang20 ASan/UBSan builds pass with FFTWGrid bounds checks (ASan leak detection disabled). Twenty-two native engineering runs include 1/2/4-rank slab/pencil replay and two-iteration MAP comparison. The new forward-only mock agrees bitwise with full evaluation within each backend. Legacy full/MAP and external-data forward outputs remain bitwise unchanged. The newly added fit-start prediction is an extra sidecar; the public optimizer and physics are unchanged. No full-library CI run is claimed. See [engineering checks (archived)](../../../ARCHIVE.md).

An initial engineering assertion incorrectly compared rank-local adjoint memory/work reports across decompositions. The fixture now compares physical metadata and numerical outputs; local storage/work counts are excluded. Original failure logs are preserved in local review artifacts. No numerical tolerance was relaxed.

The first N64/M32 attempt (16 time steps) completed one accepted iteration in 167.65 seconds, reaching coefficient error 0.469385. Its subsequent four-iteration run was interrupted after 74.88 seconds. The completed result and interrupted output are retained separately; neither is a converged MAP. The bounded campaign was changed to N32/M16, using eight particles per force cell and a newly generated M16-window mock, before collecting the endpoint series. Criteria were unchanged.

## Limits and next step

One realization, fixed nuisance parameters, finite reference and prescribed initialization. Numerical particle/force/time errors are not independently recalibrated here; that broader campaign is deferred by request. Noise seed 92 is frozen; no uncertainty, posterior coverage, independent gravity reference or global-optimum claim follows. Iteration-budget points are independent deterministic restarts from the same seed-300 field, not a complete accepted-step/line-search trace. Raw cell gradients depend on mesh normalization; projected gradients are reported alongside them.

If the optimizer has not converged, conditioning/work budgeting remains the next distinct task. PM survey/tracer/RSD integration and broader posterior calibration remain open.
