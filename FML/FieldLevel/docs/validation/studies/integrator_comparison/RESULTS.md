# Dataset-204 paired integrator comparison

Completed 19 November 2026. Both fixed four-chain cohorts completed 256 warmup +512 retained draws per chain. Same data, starts, random streams, frozen original mass, priors, integration time 1.2 and temperature ladder. Only field step/count differs. This selected-dataset experiment is not independent calibration.

| Quantity | Baseline 0.15/8 | Half step 0.075/16 |
|---|---:|---:|
| All gates | FAIL | FAIL |
| Measured divergences (all positive replicas) | 1417 | 0 |
| Failed coordinate gates /12 | 12 | 11 |
| Maximum Rhat | 1.25888 | 1.26896 |
| Minimum bulk ESS | 12.02 | 11.58 |
| Minimum tail ESS | 18.48 | 15.59 |
| Transport gate | FAIL | FAIL |
| Actual field-gradient calls, warmup included | 276480 | 522240 |
| Actual field-gradient calls, sampling only | 184320 | 348160 |
| Active cohort seconds | 2021.11 | 2790.46 |

## All coordinate diagnostics and cost

Each cell below is baseline / half step. Failed-chain ESS and efficiency estimates remain provisional. Cost includes all positive-temperature replicas and warmup; scalar/cache work enters wall time separately.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Bulk ESS per million field-gradient calls | Bulk ESS per active second |
|---|---:|---:|---:|---:|---:|
| sigma8_ratio | 1.0347 / 1.0024 | 203.44 / 484.79 | 546.33 / 623.65 | 735.82 / 928.29 | 0.10066 / 0.17373 |
| b_delta | 1.0505 / 1.06 | 95.876 / 70.552 | 130 / 413.11 | 346.77 / 135.09 | 0.047437 / 0.025283 |
| b_delta2 | 1.0441 / 1.0442 | 93.09 / 100.24 | 124.26 / 334.71 | 336.7 / 191.94 | 0.046059 / 0.035922 |
| b_K2 | 1.0626 / 1.1391 | 54.412 / 20.479 | 83.883 / 22.547 | 196.8 / 39.213 | 0.026922 / 0.0073389 |
| b_laplacian | 1.0528 / 1.1387 | 60.653 / 21.354 | 64.56 / 22.337 | 219.38 / 40.89 | 0.03001 / 0.0076526 |
| cos100 | 1.2004 / 1.1746 | 15.379 / 17.513 | 32.533 / 27.966 | 55.626 / 33.534 | 0.0076094 / 0.0062759 |
| sin100 | 1.0763 / 1.1684 | 48.012 / 17.382 | 38.184 / 16.876 | 173.65 / 33.284 | 0.023755 / 0.0062291 |
| cos110 | 1.2589 / 1.269 | 12.335 / 11.585 | 18.479 / 15.595 | 44.614 / 22.183 | 0.006103 / 0.0041516 |
| sin110 | 1.0969 / 1.171 | 39.721 / 18.517 | 31.782 / 23.041 | 143.67 / 35.457 | 0.019653 / 0.0066358 |
| cos001 | 1.1868 / 1.2165 | 17.034 / 13.754 | 21.505 / 22.131 | 61.612 / 26.336 | 0.0084282 / 0.0049289 |
| sin001 | 1.2549 / 1.2559 | 12.022 / 12.663 | 23.489 / 17.526 | 43.483 / 24.248 | 0.0059483 / 0.004538 |
| likelihood_Q | 1.0583 / 1.0136 | 61.629 / 379.34 | 289.97 / 823.56 | 222.91 / 726.36 | 0.030493 / 0.13594 |

## Interpretation and controls

The half-step case removes the observed divergences in this bounded actual-chain comparison. Its sigma8 coordinate passes (Rhat 1.0024, bulk ESS 484.79, tail ESS 623.65), but eleven other coordinates and transport fail. Worst field-coordinate bulk ESS remains about 11.6; this is a stability improvement, not a solution to joint mixing. Most coordinates do not show improved work-normalized efficiency, and all failed-chain ESS estimates remain provisional. Coordinate and transport gates must still be assessed separately; zero divergences alone does not certify convergence or calibration. No failed run is dropped or extended. Production defaults are unchanged.

Half step uses 1.889 times the field-gradient evaluations and 1.381 times the active cohort wall time. Cohorts ran concurrently on shared hardware; active seconds include startup/replay overhead and contention but exclude waiting for the other cohort. This timing is specific to this run.

All 2048 retained baseline records exactly match the corresponding original-pilot prefix. Frozen original plans, checkpoints, records, sources and binaries are unchanged. Preflight checks preserve legacy checkpoint/summary/target bytes and trajectories, reject changed/disabled settings on resume, and verify separate warmup/sampling counters. Thirty-two independent harmonic-reference trajectories and an early-stop evaluation-count test pass.

[Method](../../../../validation/galaxy/integrator_comparison/README.md), summary.json and PROVENANCE.json retain the complete outcome and compact provenance. Raw logs, full hash inventories, chains, cost ledgers and comparison-overview.png remain outside the library at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-integrator-comparison`

The separate [dataset-203 mixing audit](../mixing_audit/RESULTS.md) motivates a bounded cold-end ladder refinement before scaling up calibration. No refined ladder has yet been sampled.
