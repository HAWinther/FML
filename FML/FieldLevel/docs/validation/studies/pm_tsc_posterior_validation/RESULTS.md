> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# TSC posterior pilot: active fields, amplitude and noise

16 November 2026. [Protocol and coordinate measure](README.md).

## Status

Four N16 particle chains complete 64 warmup plus 128 retained draws each on the
same frozen data, M16 force mesh and 16 time steps. All 512 retained draws enter
diagnostics. There are **zero post-warmup divergences**, with mean acceptance
about 0.90–0.92. Every monitored Rhat is below 1.05 (maximum
**1.038102**).

**The strict mixing/precision criteria are not met.** Minimum bulk ESS is
**93.29**, below 400; minimum tail ESS is
**180.86**, below 200. Bulk ESS fails for 111 of 248 monitored
features; two fail tail ESS. The slowest bulk feature is the physical coefficient
norm squared. All consistency/target checks pass; the report retains
`passed=false`, `consistency_passed=true`, `mixing_passed=false`. No thresholds,
post-warmup thinning or burn-in removal were changed to obtain a pass.

This is the **user-selected N16 bounded pilot**, not an N32 posterior validation.
Changing particle quadrature changes the discrete target. N32 mixing, broader
refinement and repeated-realization coverage remain open.

*Archived figure: Pilot marginals and chains (`pm_tsc_posterior_validation/figures/parameters.png`). See [archive guide](../../../ARCHIVE.md).*

## Exploratory scalar summaries

| Quantity | Pilot mean | Empirical 2.5–97.5 percentiles | Rhat | Bulk / tail ESS |
| --- | ---: | --- | ---: | ---: |
| Log amplitude A, injected 0 | -0.135587 | [-0.394875, 0.146259] | 1.010968 | 358.4 / 288.4 |
| Log noise N, injected log(1e-4) | -9.067070 | [-9.272358, -8.834573] | 1.021421 | 183.9 / 258.0 |

Noise-power mean is 0.0001161282, with empirical percentiles
[9.3986687e-05, 0.00014561087].
Injected values fall inside these pilot intervals in this realization. This is
not a coverage result or a claim of validated uncertainties while ESS gates fail.

A companion whitened-coordinate MAP on the **same N16 target** converges in
49 iterations at norm 1.35008e-05, giving
A=1.968364 and noise power about 6.0396026e-05.
The difference from pilot marginals illustrates why a latent-coordinate joint
density mode is not a marginalized parameter estimate. The N32 MAP supplies only
physical initialization; it is not the target used for this comparison.

The coordinate transform is checked against the full posterior. Physical
q=exp(A/2)*s_active requires **122*A/2** in the negative log density. Inactive
Gaussian white directions integrate out exactly, leaving a 124-dimensional
active/scalar target. Keeping the Jacobian preserves the parameter marginal;
it does not change the prior. The amplitude prior retains the explicitly declared
[-4,4] support and 0/.5 Gaussian mean/scale of the joint study.

## Physical fields and predictions

The pilot physical mean initial field has coefficient error 0.268112
and correlation 0.964247 to the injected field. Means and pointwise
standard deviations use sqrt(.002)*q, not whitened coefficients at a fixed
amplitude. Their finite-sample uncertainty remains exploratory.

*Archived figure: Initial field summaries (`pm_tsc_posterior_validation/figures/initial_fields.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Latent and noisy predictions (`pm_tsc_posterior_validation/figures/predictive.png`). See [archive guide](../../../ARCHIVE.md).*

Saved latent signals and independent measurement-noise replicates use the correct
full-Hermitian exp(N)/V convention on the common observation band. Across 64
saved predictions, observed discrepancy mean is 256.444,
replicated discrepancy mean 257.807, and descriptive
replicated-greater-than-observed fraction **0.46875**. This gives
no obvious discrepancy warning in this finite realization; it is not calibrated
coverage, a convergence proof, or an independent-physics test.

*Archived figure: Mixing criteria (`pm_tsc_posterior_validation/figures/mixing.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Predictive discrepancy (`pm_tsc_posterior_validation/figures/predictive_check.png`). See [archive guide](../../../ARCHIVE.md).*

## Engineering and performance

Twenty-four short native cases pass across serial, slab MPI1/2/4, pencil MPI1/2/4
and ASan/UBSan. They validate FFT basis normalization/roundtrip, independent
trigonometric expansion, direct full-posterior density and chain-rule gradients
including the Jacobian, finite differences, active-chain canonical replay and the
untransformed full-field sampler adapter. The production HMC engine is unchanged. A separate check at the actual
N16/M16/16-step schedule gives maximum direct-density error 8.9e-12, gradient
error 2.0e-13 and refined finite-difference error 6.5e-9.

Twenty-three targeted PM/LPT regressions pass. One MPI4 regression initially hit
its wall deadline while competing with long chains; its isolated rerun passes,
with the original timeout log retained. Old CIC and TSC checkpoints resume
bitwise through the small-mesh lookup change. The extracted shared validation
factory reproduces the committed joint-driver physical fields, coefficients,
identity checks and accepted trace bitwise. Full-library CI was not run; sanitizer
leak detection is disabled.

A small-mesh assignment optimization replaces hash accumulation/lookup by bounded
dense buffers only for meshes <=16, retaining visit order, sorted packet order
and zero-weight nodes. Larger meshes keep sparse lookup. Dense sum/seen/slot
buffers total at most **68 KiB per rank/assignment** (excluding existing packet
buffers); their memory tradeoff depends on local occupancy. This is not a
universal memory/performance claim.

In a same-source serial N32 four-draw comparison (seven PM gradient evaluations),
states, features and predictions are **bitwise identical**. Process CPU time is
71.77s sparse versus 56.43s dense,
about 21.4% lower in this short case.
Host activity is uncontrolled; early wall-clock outliers were excluded in favor
of the repeated CPU-time comparison. The optimization helps but does not make
large N32 posterior campaigns inexpensive.

The joint MAP study was committed locally as c7e8758 before this work. This pilot,
lookup optimization and shared-factory extraction are a new reviewable slice.
The next statistical requirement is more effective samples, particularly for
norm/variance and noise features, before trusting intervals. Larger N32 inference
and general cosmological/tracer/survey extensions are not demonstrated here.
