# Frozen-donor joint field/scalar proposals

Completed 19 November 2026. Fresh four-chain candidate: 256 warmup +512 retained draws, thirteen replicas, half-step field HMC and exact-support amplitude rescaling. The cached exact-support cohort is reused read-only. Targets, starts, original field metric, scalar kernels and exchange schedule match. The only extra kernel translates the projected initial white field and all five physical scalars by an ordered difference from twelve fixed source states.

The donor pool, two scales (1 and 0.35, equally likely), one-attempt schedule and source identities were frozen before the candidate run. Swapping the ordered pair reverses the translation with identical probability and unit Jacobian. Full posterior MH preserves the target; out-of-support scalar proposals reject directly. Donor differences leave prior-null field coordinates fixed. Source labels are ensemble identities, not demonstrated mode labels.

| Quantity | Exact-support baseline | With donor jumps |
|---|---:|---:|
| Overall gates | FAIL | FAIL |
| Failed original coordinate gates /12 | 12 | 12 |
| Failed additional physical-mode gates /6 | 6 | 6 |
| Measured divergences | 0 | 0 |
| Transport gate | True | False |
| Qualified returns per ensemble | 6,2,1,4 | 5,0,2,3 |
| Minimum pooled exchange acceptance | 0.29199 | 0.28687 |
| Primary HMC gradients including warmup | 626688 | 626688 |
| Primary HMC gradients sampling only | 417792 | 417792 |
| Extra donor target-value calls including warmup | 0 | 13553 |
| Extra donor target-value calls sampling only | 0 | 9078 |
| Raw four-worker active seconds | 2259.8 | 2138.9 |

## Interpretation

The candidate improves all six physical-initial-mode bulk ESS estimates by 1.50 to 10.33 times at equal primary HMC gradient counts. Their Rhat range falls from 1.129–1.280 to 1.035–1.134, but bulk ESS is still only 21–119 against the required 400. Every original and additional coordinate gate fails. Sigma8 bulk ESS worsens from 360 to 296 and Rhat worsens from 1.0102 to 1.0227; one ensemble has no qualified prior-refresh return, so transport also fails. All divergence counts remain zero.

No cold donor proposal is accepted (0/2048); 1269/24576 proposals across positive-temperature replicas accept, including 33/12275 large proposals. The physical-mode improvement therefore occurs through the altered hotter-replica trajectories and exchanges, without a direct accepted cold donor move. This single selected-target comparison does not establish a repeatable gain. The optional experimental kernel is retained with its failures; it is not promoted to a production default. Robust field/bias mixing remains the next gate before broader calibration or actual-catalog inference.

## Every monitored coordinate

Cells show baseline / candidate. Physical initial-mode monitors are alpha times the white-field Fourier projection. They were predeclared for this study and are required in addition to the unchanged twelve original gates. Failed-chain ESS remains provisional.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Pass |
|---|---:|---:|---:|---|
| sigma8_ratio | 1.0102 / 1.0227 | 360.02 / 296.43 | 1043.8 / 808.15 | False / False |
| b_delta | 1.0527 / 1.0537 | 64.948 / 51.474 | 263.77 / 205.5 | False / False |
| b_delta2 | 1.0625 / 1.1507 | 56.354 / 19.477 | 367.39 / 511.16 | False / False |
| b_K2 | 1.0205 / 1.0424 | 126.46 / 82.026 | 353.56 / 935.09 | False / False |
| b_laplacian | 1.0258 / 1.0185 | 199.34 / 351.32 | 1020.1 / 243.63 | False / False |
| cos100 | 1.2079 / 1.1494 | 14.728 / 19.278 | 137.59 / 34.834 | False / False |
| sin100 | 1.2444 / 1.0442 | 12.348 / 93.794 | 40.264 / 162.06 | False / False |
| cos110 | 1.2652 / 1.0503 | 11.769 / 109.9 | 192.93 / 98.014 | False / False |
| sin110 | 1.1279 / 1.0618 | 22.569 / 64.441 | 128.05 / 473.82 | False / False |
| cos001 | 1.1972 / 1.0984 | 15.442 / 101.26 | 191 / 174.18 | False / False |
| sin001 | 1.2666 / 1.0279 | 11.478 / 111.82 | 124.72 / 162.4 | False / False |
| likelihood_Q | 1.0465 / 1.0146 | 72.854 / 253.21 | 513.94 / 895.93 | False / False |
| alpha_white_cos100 | 1.2185 / 1.1345 | 14.06 / 21.091 | 115.58 / 46.616 | False / False |
| alpha_white_sin100 | 1.2329 / 1.0419 | 13.089 / 111.88 | 116.94 / 159.56 | False / False |
| alpha_white_cos110 | 1.263 / 1.0518 | 11.921 / 107.07 | 227.9 / 120.45 | False / False |
| alpha_white_sin110 | 1.1292 / 1.0637 | 23.296 / 69.727 | 318.11 / 98.811 | False / False |
| alpha_white_cos001 | 1.204 / 1.0914 | 14.737 / 118.78 | 116.5 / 191.9 | False / False |
| alpha_white_sin001 | 1.2803 / 1.0349 | 11.319 / 116.87 | 74.523 / 200.88 | False / False |

## Proposal acceptance and work

Sampling donor acceptance: 1269/24576 (0.051636); cold: 0/2048 (0). Large-scale proposals: 33/12275; cold large-scale: 0/1027. Support rejections: 15498. All rejected proposals remain in the trajectories.

Primary HMC gradient counts exclude donor value evaluations; these are reported separately. The JSON additionally reports ESS divided by gradient plus donor-value calls as a conservative mixed-call proxy, not an equal-cost timing unit. Both cohorts use four workers at different times on shared hardware. Raw wall times include auxiliary replay and do not establish a dedicated speedup. No extension, replacement, pooling or online donor adaptation occurred.

## Controls and limits

Thirty independent native controls cover proposal symmetry, reciprocal translations, prior-null preservation and a known unequal-weight correlated Gaussian mixture. A local-only negative control remains trapped. The frozen local model screen reproduces twelve predictions/prior energies, validates all 132 endpoint transfers, and records all 768 feasibility proposals. Its 2.6% mean acceptance and 3.7% cross-ensemble large-jump acceptance use source states that also constructed the pool; this is an in-sample feasibility check, not posterior evidence.

Native controls reproduce disabled-kernel legacy draws, targets and summaries, and enabled 32+16 versus direct-48 checkpoint/summary/draw bytes. Missing, malformed or changed donor pools reject before writes. Real four-chain writer-lock, bounded-pause, missing-ledger and target-identity controls pass. Complete candidate-ensemble restart replay, source/binary/pool hashes and every baseline record/checkpoint are verified. The donor loader/driver is serial; the reused model-support projector has separate slab/pencil verification.

This is selected-dataset sampler development. Independent calibration, larger fields and actual-catalog inference remain open. [Method](../../../../validation/galaxy/donor_jumps/README.md). Complete summaries/provenance and controls are retained here; raw records, binaries, pool, inventories, logs, generated hook patch and plot remain at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-donor-jumps`.
