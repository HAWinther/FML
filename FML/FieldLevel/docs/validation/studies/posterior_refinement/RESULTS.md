# Posterior numerical-refinement check

Frozen numerical-sensitivity gate: **PASS**. All six resolution/tolerance
settings use all 8192 existing cold states (four chains × 2048).
Largest estimated monitored mean shift: 1.08721e-07 baseline posterior SD;
largest absolute shift plus two paired MCSE: 1.62468e-07 SD (gate 0.1).
Largest per-state noise-weighted model distance: 1.57584e-05 (gate 0.1).
Minimum per-chain importance-overlap fraction: 0.999999999999442 (gate 0.9).

The source [2048-draw mixing check](../tempering_long/RESULTS.md) passes all eleven
monitored convergence gates. Its chains, prior failed runs and raw records remain
unchanged. This experiment creates predictions and reweighted estimates only.

## Fixed physical target

Compact mesh 4, 64 real white coordinates, box 100, original spectrum
20000 exp(-5k) with amplitude ratio squared, initial spherical cutoff 1.7 kf,
matter cubic cutoff 1.8 kf, likelihood cubic cutoff 1.6 kf, fixed noise power
1000 and the original mock modes. Same 2LPT evolution, four physical bias priors
and sigma8/reference-sigma8 prior; no input mock is regenerated.

| Case | Product/particle lattice | Deposition | PSWF tolerance |
|---|---:|---:|---:|
| baseline | 8 | 16 | 1e-08 |
| tolerance | 8 | 16 | 1e-12 |
| deposition | 8 | 32 | 1e-12 |
| product12 | 12 | 32 | 1e-12 |
| reference16 | 16 | 48 | 1e-12 |
| reference24 | 24 | 64 | 1e-12 |

The product/displacement and particle-lattice meshes remain coupled. These
comparisons cannot independently assign their contributions.

## Frozen comparisons

Every refinement is compared to baseline; the two finest references are compared
separately. All five physical scalars and six predefined field projections enter
the monitored marginal checks. Every case must retain per-chain and pooled
importance overlap >=0.9, with maximum weight <=5 times uniform. Every comparison
also gates absolute log-ratio <=1, sampled noise-weighted model distance <=0.1,
mean shift <=0.1 posterior SD, absolute mean shift plus two MCSE <=0.1 SD,
5/50/95-percentile shifts <=0.1 SD, relative SD changes <=2%, and weighted
predictive-mean distance <=0.1. No thresholds or cases were retuned.

| Comparison | Max model distance | Max mean shift / SD | Max shift + 2 MCSE / SD | Max quantile shift / SD | Gate |
|---|---:|---:|---:|---:|---|
| baseline → tolerance | 3.6791e-07 | 2.50533e-08 | 2.80247e-08 | 2.55201e-08 | PASS |
| baseline → deposition | 3.67948e-07 | 2.50569e-08 | 2.80286e-08 | 2.5524e-08 | PASS |
| baseline → product12 | 1.5758e-05 | 1.08721e-07 | 1.62468e-07 | 1.62881e-07 | PASS |
| baseline → reference16 | 1.57584e-05 | 1.08721e-07 | 1.62468e-07 | 1.6288e-07 | PASS |
| baseline → reference24 | 1.57584e-05 | 1.08721e-07 | 1.62468e-07 | 1.6288e-07 | PASS |
| reference16 → reference24 | 1.06758e-11 | 2.79953e-13 | 3.55681e-13 | 8.03315e-13 | PASS |

The finest pair agrees within 1.06758e-11 in sampled
noise-weighted model distance. Maximum relative monitored SD change across all
comparisons is 1.0897e-07.
The largest absolute likelihood log ratio is 1.49072e-05.
Full baseline replay errors are 0 for
predictions and 0 for likelihood quadratics.
Independent Python reconstruction of every quadratic agrees within
8.17764e-16 scaled error.

## Method, uncertainty and validation

Since the latent dimension, priors, noise and observed mode basis remain identical,
likelihood-only ratios exp(Q_baseline-Q_case) are exact. Estimates are normalized
importance estimates on the baseline posterior. This is not a new refined-target
chain. Weight-overlap indices are not independent draw counts or MCMC ESS.

Mean-shift MCSE uses paired ratio-estimator influence differences at identical
source states. Batch means at lengths 64 and 128 preserve within-chain time
ordering; a conservative independent-chain-mean error is also considered. The
largest estimate is used. Common samples cancel much of the error in numerical
differences; the tiny shift errors do not imply equally precise absolute posterior
means. Quantile and SD shifts are point estimates. This is a sampled-posterior
sensitivity result, not a uniform bound over unvisited prior space.

Preflight validates analytic Gaussian shifts/scales, stable log normalization,
paired-null cancellation, exact-zero-weight quantiles and autocorrelated MCSE.
The first 1024 baseline predictions and the full 8192-state baseline both reproduce
the source exactly. Invalid numerical controls, mode weights and scalar support
reject. Bounded and SIGTERM pauses drain active batches and preserve completed
SHA-verified files; resume verifies immutable inputs and receipts. All four source
checkpoint hashes remain unchanged. No hot or replay samples are pooled.

All 49,152 required state predictions are retained outside the library. Summed
native process wall times by case (not CPU time; four workers):

- baseline: 44.1 s.
- tolerance: 92.0 s.
- deposition: 356.8 s.
- product12: 605.5 s.
- reference16: 1607.9 s.
- reference24: 4735.5 s.

## Interpretation and remaining scope

The baseline numerical settings comfortably pass this bounded posterior-sensitivity gate. Independent-mock recovery/calibration is the next scientific check before expanding the target or treating the real-catalog deliverable as validated.

This covers one same-model synthetic mock, fixed bands and this compact target.
It does not establish all-field/global continuum convergence, nonlinear model-order
accuracy, independent-mock coverage, real-catalog fidelity or production MPI replica
orchestration. The baseline sampler and its target are unchanged.

[Plan](../../../../validation/galaxy/refinement/plan.json),
[driver/method instructions](../../../../validation/galaxy/refinement/README.md),
FROZEN.json and summary.json preserve controls, identity and all measured gates.
Raw input states, predictions, executable, pause/replay evidence, plots and review
patches are in
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-posterior-refinement/`.
Plots: campaign/refinement-overview.png and campaign/scalar-stability.png.
