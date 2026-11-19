# Dataset-203 cold-end ladder comparison

Completed 19 November 2026. Both fixed four-chain cohorts completed 256 warmup +512 retained cold draws per chain. Both use field step 0.075/16, integration time 1.2, the original fixed per-ensemble metric and four exchange passes. The refined ladder adds beta=sqrt(0.5) and sqrt(0.125). All cold targets exactly match the original dataset. No failed run was extended, excluded, replaced or retuned.

| Quantity | Baseline: 11 replicas | Refined: 13 replicas |
|---|---:|---:|
| All gates | FAIL | FAIL |
| Measured positive-beta divergences | 0 | 0 |
| Failed coordinate gates /12 | 12 | 12 |
| Maximum Rhat | 1.62485 | 1.30805 |
| Minimum bulk ESS | 6.80 | 10.14 |
| Minimum tail ESS | 28.64 | 64.28 |
| Transport gate | FAIL | FAIL |
| Minimum pooled edge acceptance | 0.04614 | 0.28271 |
| Qualified round trips by ensemble | 1,0,0,1 | 5,0,1,3 |
| Primary field-gradient calls, warmup included | 522240 | 626688 |
| Primary field-gradient calls, sampling only | 348160 | 417792 |
| Active cohort seconds | 2776.50 | 3170.07 |

## All monitored coordinates

Each cell is baseline / refined. All twelve coordinate gates remain Rhat<1.01 and bulk/tail ESS>=400. Failed-chain ESS, posterior intervals and cost-normalized estimates remain provisional.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Bulk ESS per million total field-gradient calls | Bulk ESS per active second |
|---|---:|---:|---:|---:|---:|
| sigma8_ratio | 1.0355 / 1.0462 | 193.28 / 121.27 | 426.07 / 389.28 | 370.1 / 193.51 | 0.069612 / 0.038254 |
| b_delta | 1.0428 / 1.0607 | 81.44 / 49.757 | 232.33 / 259.07 | 155.94 / 79.396 | 0.029332 / 0.015696 |
| b_delta2 | 1.0703 / 1.3081 | 39.45 / 10.138 | 58.132 / 66.261 | 75.54 / 16.176 | 0.014209 / 0.0031979 |
| b_K2 | 1.1254 / 1.085 | 22.548 / 31.417 | 39.551 / 290.92 | 43.175 / 50.132 | 0.0081209 / 0.0099106 |
| b_laplacian | 1.0393 / 1.0331 | 106.26 / 125 | 415.62 / 1006 | 203.46 / 199.47 | 0.03827 / 0.039433 |
| cos100 | 1.2275 / 1.2339 | 12.602 / 25.248 | 45.809 / 64.279 | 24.131 / 40.289 | 0.0045389 / 0.0079646 |
| sin100 | 1.6075 / 1.1185 | 6.8018 / 49.398 | 28.643 / 203.57 | 13.024 / 78.823 | 0.0024498 / 0.015582 |
| cos110 | 1.5142 / 1.0903 | 7.3648 / 63.86 | 49.515 / 196.26 | 14.102 / 101.9 | 0.0026525 / 0.020145 |
| sin110 | 1.4366 / 1.0809 | 8.1127 / 39.645 | 158.55 / 418.83 | 15.534 / 63.26 | 0.0029219 / 0.012506 |
| cos001 | 1.2188 / 1.1599 | 18.143 / 24.721 | 55.456 / 91.753 | 34.741 / 39.447 | 0.0065346 / 0.0077982 |
| sin001 | 1.6248 / 1.1656 | 7.1786 / 62.114 | 30.484 / 93.82 | 13.746 / 99.114 | 0.0025855 / 0.019594 |
| likelihood_Q | 1.1271 / 1.0143 | 24.29 / 299.84 | 159 / 737.08 | 46.51 / 478.45 | 0.0087482 / 0.094584 |

## Work and interpretation

The cold-end overlap bottleneck is much reduced on this selected target: minimum
pooled edge acceptance rises from 4.61% to 28.27%, and qualified return journeys
increase from 2 to 9. Ensemble 1 still has no complete return. Both cohorts fail
all twelve coordinate gates. Worst Rhat improves from 1.625 to 1.308, while the
refined worst coordinate is now b_delta2. Provisional bulk ESS per gradient improves
for all six monitored field projections (about 1.1–7.2 times), but decreases for
sigma8 and several bias coefficients. The sigma8 Rhat is 1.0355 / 1.0462 and its
bulk ESS 193.28 / 121.27. This is useful transport/field-exploration evidence,
not a globally better converged posterior or a calibration success.

Next: a separately frozen field/scalar-coupling or preconditioning comparison,
using the refined ladder as the transport reference and retaining the stable
half-step field update. The actual existing scalar kernel already uses joint
amplitude/bias proposals; avoid mistaking it for simple coordinate-only Gibbs.
No additional draws were taken.

Refinement uses 1.200 times the actual field-gradient work and 1.142 times the active cohort wall time. These are equal cold-draw budgets, not equal work. Active times include warmup, startup, replay and shared-hardware contention, but exclude waiting for the other cohort. Primary gradient counters exclude auxiliary validation replay; wall times include it. Scalar/cache and exchange work are outside the gradient counter and enter timing separately.

Exchange and qualified trips are diagnostics separate from coordinate convergence. Any improvement at the cold end can be offset by the longer ladder or slower traversal elsewhere; more trips alone is not a posterior-convergence claim. This is a previously inspected selected dataset with replica-index seed reuse, not independent calibration. Production kernels/defaults and the original failed pilot remain unchanged.

## Controls and provenance

The 13-replica native controls reproduce complete checkpoint/summary/target bytes and 32 cold draws across fresh-process restart, reject changed beta values/count without writes, and verify warmup/sampling gradient counts. All eight study chains pause at a complete 16-sweep checkpoint; concurrent writers and missing budget state reject without native writes. Each cohort also performs its native full-ensemble replay. Final analysis verifies frozen runtime sources/binaries, every original input hash, all twelve coordinates, target identity and independent native exchange/qualified-trip reconstruction.

[Method](../../../../validation/galaxy/ladder_comparison/README.md), summary.json and PROVENANCE.json retain all outcomes and compact provenance. Full manifests, original hashes, native builds, raw chains/logs/cost ledgers and ladder-comparison.png remain at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-ladder-comparison`
