# Blocked field/cached-scalar pilot

Recorded 18 November 2026. **Frozen mixing gate: FAIL.**
[Driver and invariant-target argument](../../../../validation/galaxy/blocked/README.md);
[frozen protocol](../../../../validation/galaxy/blocked/plan.json);
[machine-readable results](summary.json); [joint baseline](../joint_pilot/RESULTS.md).

## What changed

This validation prototype alternates one conditional field HMC move with four
existing cached amplitude/bias transitions. Field HMC uses eight leapfrog steps
at size 0.15, without jitter/adaptation; the Gaussian field metric stays frozen
at each chain's original scalar start. The scalar block uses eight conditional
bias-slice sweeps per transition and the physical-coordinate joint amplitude/bias
MH proposal with reflected amplitude step 0.07 and its full proposal-density ratio.
Cached sufficient statistics are rebuilt after field/amplitude changes; only the
cheap linear-bias steps reuse them. Physical priors and the full target are unchanged.

Both conditional kernels preserve their conditional target, so their composition
preserves the joint target. There is no marginal-likelihood approximation,
extra Gaussian bias penalty or physical-coordinate Jacobian. Deterministic block
composition need not itself satisfy detailed balance. Existing production sampler
APIs are unchanged; this is a maintained serial validation prototype.

## Frozen comparison and numerical checks

The serialized data target is identical to the preceding joint pilot. The N4/N8/N16
PSWF model, cutoffs, noise power, spectrum, priors, dispersed field/scalar starts,
four chains, 256 discarded warmup sweeps, 1024 retained draws and eleven-coordinate
gates match. Integration settings, scalar substeps and adaptation differ, so this
is neither a single-setting ablation nor a matched-compute test. No baseline or
restart-replay draws are pooled; there was no post-run extension, retuning, chain
exclusion or threshold change.

Before sampling, 18 cache/full-target comparisons on two independent fields and
72 bias finite differences pass. Maximum scaled cache/full-target gap error is
1.36212e-15. Four uninterrupted blocked sweeps match
two plus two across fresh processes byte-for-byte in state and summary. Wrong-chain
identity and checksum-corrupt snapshots are rejected without advancing CURRENT.
The existing FixedIC regression passes 20 serial checks, including independent
bounded posterior quadrature, proposal normalization and rank-deficient cases.

During the campaign, an independent continuation of chain 0 at sweep 384 produces
the exact same 64 retained draw records and numerical summary as the primary
continuation. The distinct validation checkpoint format saves completed outer
boundaries, field/scalars and counters. Component RNG streams are deterministically
keyed by chain, outer index and block, so no within-block RNG state crosses a saved
boundary. Exactly 4096 primary draws enter diagnostics. This does not test power-loss
durability or establish a production MPI blocked-checkpoint API.

## Results

All eleven coordinates require computable Rhat < 1.01, bulk ESS >= 400 and tail
ESS >= 400; sampling divergences must be zero per chain. Diagnostics reuse the
baseline's six fixed field projections and FML rank-normalized split/folded code.

| Coordinate | Joint Rhat | Blocked Rhat | Joint bulk ESS | Blocked bulk ESS | Blocked tail ESS | Gate |
|---|---:|---:|---:|---:|---:|---|
| sigma8_ratio | 1.3171 | 1.4774 | 10.4 | 7.7 | 32.3 | FAIL |
| b_delta | 2.0918 | 1.5496 | 5.3 | 7.3 | 30.6 | FAIL |
| b_delta2 | 2.8760 | 1.2286 | 4.6 | 14.8 | 488.7 | FAIL |
| b_K2 | 2.0596 | 1.3452 | 5.4 | 9.5 | 50.4 | FAIL |
| b_laplacian | 1.9101 | 1.2732 | 5.6 | 11.4 | 126.5 | FAIL |
| cos100 | 2.1939 | 1.5673 | 5.1 | 7.0 | 31.2 | FAIL |
| sin100 | 2.2680 | 1.4331 | 5.0 | 8.2 | 43.4 | FAIL |
| cos110 | 2.5008 | 1.5919 | 4.8 | 6.9 | 24.1 | FAIL |
| sin110 | 1.9587 | 1.3622 | 5.6 | 9.3 | 26.4 | FAIL |
| cos001 | 1.9347 | 1.5761 | 5.6 | 7.2 | 40.1 | FAIL |
| sin001 | 2.5871 | 1.5564 | 4.8 | 7.2 | 46.6 | FAIL |

Failed coordinates: sigma8_ratio, b_delta, b_delta2, b_K2, b_laplacian, cos100, sin100, cos110, sin110, cos001, sin001.
Field sampling divergence counts: 0, 0, 0, 0.
Mean field acceptance probabilities: 0.9780, 0.9743, 0.9838, 0.9760.
Scalar joint-MH acceptance: 0.5984, 0.2620, 0.5972, 0.5688.
Scalar joint-proposal prior-support rejection counts:
532, 2727, 568, 628.
Each chain builds 6400 sufficient-statistic caches across
warmup and retained sweeps. Bias slices are additional conditional moves;
the scalar MH acceptance is not the acceptance of the whole scalar block.

Main-chunk per-process wall times sum to 525.1 s
for the old joint run and 741.0 s for blocked
(1.411 times). These host-local measurements include work
allocation, target/cache initialization, prediction, draw output and checkpoints;
preflight/replay are excluded. Both campaigns use two concurrent serial chains.
Different run times and system load prevent a calibrated production-speed claim.
ESS-per-second values are retained in the JSON, but an unmixed baseline does not
provide an asymptotic efficiency reference.

## Interpretation and remaining work

The declared joint mixing gate still fails. Preserve both campaigns; diagnose the remaining field/scalar dependence and compare another separately declared change before scaling or claiming a marginalized amplitude result.

Pooled scalar quantiles/correlations are descriptive when gates fail. Plots are
not evidence of an amplitude shift or calibration. Six projections do not validate
all field coordinates. Fitted-data residual bands show forward-prediction variation
without fresh observation noise; they are not independent predictive tests.
This is same-model synthetic sampling, not repeated-mock coverage, physical
resolution convergence, N-body/HOD robustness or a real-catalog result.

Source/binary/plan hashes, commands, checkpoints, primary/replay draws and figures
remain outside the library at workspace `outputs/fieldlevel-blocked-pilot/`.
Figures under `campaign/`: `comparison.png`, `joint-summary.png`, `traces.png`,
`scalar-posterior.png` and `residuals-diagnostics.png`. The preceding joint campaign
is retained intact. Item 52 remains ongoing.

A concrete [proposed next kernel](../../../../validation/galaxy/blocked/NEXT_STEPS.md)
rescales amplitude and whitened field together, with the full-posterior MH ratio
and the real-coordinate Jacobian. It has not been implemented or run here.
The observed blocked comparison is not a successful sigma8 sampler replacement.
