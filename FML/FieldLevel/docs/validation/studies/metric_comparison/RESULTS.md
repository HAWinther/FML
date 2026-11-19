# Frozen nonlinear field-metric actual-chain comparison

Completed 19 November 2026. The fresh candidate completes four chains of 256 warmup +512 retained cold draws. The complete cached thirteen-replica refined-ladder run is reused as the baseline. Same target, starts, seed domains, ladder, half-step HMC, scalar/rescaling kernels, priors and all original gates. Only the field mass changes. No original baseline analysis or checkpoint is rewritten.

| Quantity | Cached original metric | Frozen nonlinear metric |
|---|---:|---:|
| All gates | FAIL | FAIL |
| Failed coordinate gates /12 | 12 | 12 |
| Measured positive-beta divergences | 0 | 0 |
| Maximum Rhat | 1.3081 | 1.2575 |
| Minimum bulk ESS | 10.138 | 11.968 |
| Minimum tail ESS | 64.279 | 60.613 |
| Transport gate | FAIL | FAIL |
| Minimum pooled edge acceptance | 0.28271 | 0.27588 |
| Qualified return journeys by ensemble | 5,0,1,3 | 3,4,0,0 |
| Primary field-gradient calls, warmup included | 626688 | 626688 |
| Primary field-gradient calls, sampling only | 417792 | 417792 |
| Raw active seconds (different concurrency) | 3170.1 | 2103 |

## All twelve diagnostics and work

Each cell is cached original / fresh nonlinear metric. Failed-chain ESS and efficiency estimates remain provisional. No wall-time speedup is certified: the original ran alongside another four-chain cohort (eight workers total); the candidate uses four workers. Primary gradient counts include warmup and exclude auxiliary replay, which enters wall timing. Construction cost is recorded separately in the local probe, not amortized into these chain counts.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Bulk ESS per million primary gradients | GN / baseline bulk ESS per gradient |
|---|---:|---:|---:|---:|---:|
| sigma8_ratio | 1.0462 / 1.0251 | 121.27 / 322.31 | 389.28 / 495.09 | 193.51 / 514.31 | 2.6578 |
| b_delta | 1.0607 / 1.07 | 49.757 / 41.243 | 259.07 / 60.613 | 79.396 / 65.811 | 0.82889 |
| b_delta2 | 1.3081 / 1.2575 | 10.138 / 11.968 | 66.261 / 205.84 | 16.176 / 19.097 | 1.1805 |
| b_K2 | 1.085 / 1.1515 | 31.417 / 19.691 | 290.92 / 760.1 | 50.132 / 31.42 | 0.62675 |
| b_laplacian | 1.0331 / 1.034 | 125 / 116.33 | 1006 / 698.19 | 199.47 / 185.62 | 0.93057 |
| cos100 | 1.2339 / 1.2021 | 25.248 / 14.514 | 64.279 / 120.01 | 40.289 / 23.16 | 0.57485 |
| sin100 | 1.1185 / 1.1921 | 49.398 / 38.599 | 203.57 / 182.7 | 78.823 / 61.592 | 0.7814 |
| cos110 | 1.0903 / 1.1327 | 63.86 / 23.856 | 196.26 / 418.49 | 101.9 / 38.066 | 0.37356 |
| sin110 | 1.0809 / 1.0894 | 39.645 / 38.854 | 418.83 / 295.36 | 63.26 / 61.999 | 0.98006 |
| cos001 | 1.1599 / 1.1714 | 24.721 / 16.495 | 91.753 / 210.5 | 39.447 / 26.321 | 0.66726 |
| sin001 | 1.1656 / 1.2072 | 62.114 / 21.977 | 93.82 / 205.05 | 99.114 / 35.068 | 0.35382 |
| likelihood_Q | 1.0143 / 1.0313 | 299.84 / 162 | 737.08 / 671.49 | 478.45 / 258.51 | 0.5403 |

## Decision and next experiment

Do **not adopt** this conditional-field metric as the new reference. It improves
sigma8 bulk ESS from 121.27 to 322.31 at equal gradient work (2.66 times), but
sigma8 still fails Rhat=1.0251 and bulk ESS<400. All six field-projection bulk ESS
values worsen, to 0.35–0.98 times baseline. Both versions fail all twelve coordinate
gates and transport. Qualified returns change from 5,0,1,3 to 3,4,0,0; both have
zero measured divergences. Worst Rhat improves only from 1.3081 to 1.2575. The
local accepted-jump screen was not a reliable predictor of full joint exploration.

Retain the stable half-step update, refined temperature ladder and original field
mass as the comparison reference. The nonlinear metric remains a documented
experimental candidate, not a production/default change. Point 1 (robust joint
sampling and calibration) remains open.

The next targeted candidate is support-aware amplitude/field interweaving: rescale
only the exact initial-band component, leaving prior-only white directions fixed.
For an orthogonal model-supplied support projector P of global real rank r,
white_new=(I-P)white+(alpha/alpha_new)P white and
log_J=r*log(alpha/alpha_new). Use the full posterior MH correction and the same
symmetric reflected physical-alpha proposal. The existing all-coordinate move
and its dimension-64 Jacobian are correct; this proposal avoids moving forty-six
uninformed coordinates in this fixture. Derive P from the actual model's initial
filter, not the fitted Jacobian/metric rank. Prediction invariance, inverse/Jacobian,
analytic marginal, prior-null and slab/pencil ownership checks must precede chains.
This is a proposal only; no support-aware kernel or new chains were run here.

## Scope and controls

The [local feasibility screen](../gauss_newton_probe/RESULTS.md) used twelve saved states to fit and probe the same frozen candidate. That in-sample result establishes operator/trajectory feasibility only. The new chains test the complete joint sampler, not just conditional field jumps. Neither experiment is independent calibration or real-catalog inference.

The rank-eighteen operator preserves the full 64-dimensional prior and changes only fixed kinetic geometry. Its exact small-reference construction is not a scalable distributed metric builder. Production design must retain broad FFT preconditioning and add controlled nonlinear corrections or matrix-free actions; a fixed-rank prior-base sketch alone need not precondition all informative modes at large dimension.

Native compatibility controls preserve old-kernel draws/summary/target; new-metric fresh-process restart reproduces checkpoint/summary/draw bytes; changed eigenvalues/kind and missing/truncated metric files reject without native writes. The complete metric bytes are included in native checkpoint identity. The real runner passes bounded pause, exclusive-writer, missing-ledger and original-target checks. Four candidate chains also perform the full-ensemble native replay. Every reused baseline record/checkpoint and frozen source/binary is reverified at analysis. controls.json and PROVENANCE.json retain the evidence.

[Method and fixed budgets](../../../../validation/galaxy/metric_comparison/README.md). Raw chains, builds, generated native-hooks.patch, full hash inventories, logs and metric-comparison.png remain at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-metric-comparison`.
