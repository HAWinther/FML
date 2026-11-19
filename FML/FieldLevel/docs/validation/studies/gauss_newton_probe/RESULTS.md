# Frozen nonlinear Gauss-Newton metric: local feasibility

Completed 19 November 2026. Operator controls and the prospectively fixed local screen **PASS**. This is an exploratory in-sample conditional-field test on twelve previously inspected dataset-203 states, not posterior-chain mixing or calibration.

| Quantity | Original frozen metric | Nonlinear metric |
|---|---:|---:|
| Native trajectories | 96 | 96 |
| Actual field-gradient calls | 1632 | 1632 |
| Divergences | 0 | 0 |
| Mean acceptance | 0.997297 | 0.996552 |
| Worst frame mean acceptance | 0.993872 | 0.989500 |

| Projection | Nonlinear / original accepted squared jump per gradient |
|---|---:|
| cos100 | 1.1263 |
| sin100 | 1.1099 |
| cos110 | 2.1743 |
| sin110 | 1.6025 |
| cos001 | 0.9240 |
| sin001 | 0.8589 |

Geometric-mean jump/call ratio: **1.2297**. Four projections improve and two worsen. These are accepted one-trajectory jumps, not ESS, evidence of global transport or a certified speedup.

Each per-ensemble metric averages the three saved-state Gauss-Newton matrices with equal weights and retains eighteen eigenvectors. The full 64-dimensional Gaussian field prior remains; the 46 prior-only directions retain unit kinetic mass. The operator stores a rank-bounded basis, not a dense field mass. Exact SVD construction is confined to this small serial fixture; production randomized/distributed construction remains open.

All twelve source predictions and likelihoods reproduce. Maximum independent directional Jacobian discrepancy: 8.118e-06, within the predeclared 1e-5 gate; maximum full-gradient reconstruction discrepancy: 4.885e-15. Independent dense inverse/root checks, 48 analytic harmonic HMC comparisons, null-direction and invalid-metric tests pass. Model geometry, field/scalar priors, half-step trajectory length and Gaussian target are unchanged.

The nonlinear mass is fixed before any subsequent chains and changes only the momentum distribution. It includes no scalar cross blocks and no residual-weighted Hessian terms. The same saved states construct and probe it; this local screen establishes feasibility only. Next is the separate [fixed-budget actual-chain comparison](../metric_comparison/RESULTS.md) against the completed refined-ladder baseline.

[Method and gates](../../../../validation/galaxy/gauss_newton/README.md), summary.json and PROVENANCE.json retain the complete result. Raw rows, builds, controls, metric files and metric-probe.png stay at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-gauss-newton`.
