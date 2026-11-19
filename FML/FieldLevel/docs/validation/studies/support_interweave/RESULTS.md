# Exact initial-support amplitude/field interweaving

Completed 19 November 2026. The fresh candidate completes four chains of 256 warmup +512 retained draws. The full cached refined-ladder/original-field-metric cohort is reused read-only as the baseline. Same model, target, starts, seed domains, thirteen replicas, half-step HMC, field mass, scalar kernels and gates; only rescaling support and its matching Jacobian differ.

The proposal rescales the exact spherical initial-band component and fixes its prior-null complement. Its transformation rank is eighteen; all 64 white prior coordinates remain. The projector is derived from the model filter, not observations or fitted numerical rank. Full posterior MH uses the global-rank Jacobian and the same reflected physical-alpha proposal. The original all-white kernel and full-dimension Jacobian are correct.

| Quantity | All-white reference | Exact initial support |
|---|---:|---:|
| All gates | FAIL | FAIL |
| Failed coordinate gates /12 | 12 | 12 |
| Measured divergences | 0 | 0 |
| Maximum Rhat | 1.3081 | 1.2666 |
| Minimum bulk ESS | 10.138 | 11.478 |
| Minimum tail ESS | 64.279 | 40.264 |
| Transport gate | FAIL | PASS |
| Minimum pooled exchange acceptance | 0.28271 | 0.29199 |
| Qualified returns by ensemble | 5,0,1,3 | 6,2,1,4 |
| Cold rescaling acceptance | 0.65808 | 0.81531 |
| All positive-beta rescaling acceptance | 0.63386 | 0.78964 |
| Primary gradients, warmup included | 626688 | 626688 |
| Primary gradients, sampling only | 417792 | 417792 |
| Raw active seconds (unmatched concurrency) | 3170.1 | 2259.8 |

## Complete coordinate diagnostics

Each cell is all-white / exact-support. All failed-chain ESS and efficiency estimates remain provisional. These are equal primary-gradient/cold-draw budgets, not a dedicated wall-time benchmark. The original ran alongside another four-chain cohort; the candidate uses four workers. Extra projector FFT work is outside gradient counts and inside raw timing. Primary gradient counts include warmup and exclude auxiliary replay, which enters time.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Bulk ESS per million primary gradients | Support / all-white bulk ESS per gradient |
|---|---:|---:|---:|---:|---:|
| sigma8_ratio | 1.0462 / 1.0102 | 121.27 / 360.02 | 389.28 / 1043.8 | 193.51 / 574.48 | 2.9688 |
| b_delta | 1.0607 / 1.0527 | 49.757 / 64.948 | 259.07 / 263.77 | 79.396 / 103.64 | 1.3053 |
| b_delta2 | 1.3081 / 1.0625 | 10.138 / 56.354 | 66.261 / 367.39 | 16.176 / 89.924 | 5.559 |
| b_K2 | 1.085 / 1.0205 | 31.417 / 126.46 | 290.92 / 353.56 | 50.132 / 201.8 | 4.0253 |
| b_laplacian | 1.0331 / 1.0258 | 125 / 199.34 | 1006 / 1020.1 | 199.47 / 318.08 | 1.5946 |
| cos100 | 1.2339 / 1.2079 | 25.248 / 14.728 | 64.279 / 137.59 | 40.289 / 23.501 | 0.58331 |
| sin100 | 1.1185 / 1.2444 | 49.398 / 12.348 | 203.57 / 40.264 | 78.823 / 19.704 | 0.24998 |
| cos110 | 1.0903 / 1.2652 | 63.86 / 11.769 | 196.26 / 192.93 | 101.9 / 18.78 | 0.1843 |
| sin110 | 1.0809 / 1.1279 | 39.645 / 22.569 | 418.83 / 128.05 | 63.26 / 36.013 | 0.56928 |
| cos001 | 1.1599 / 1.1972 | 24.721 / 15.442 | 91.753 / 191 | 39.447 / 24.64 | 0.62465 |
| sin001 | 1.1656 / 1.2666 | 62.114 / 11.478 | 93.82 / 124.72 | 99.114 / 18.315 | 0.18479 |
| likelihood_Q | 1.0143 / 1.0465 | 299.84 / 72.854 | 737.08 / 513.94 | 478.45 / 116.25 | 0.24298 |

## Decision and physical-field follow-up

Retain this as a validated **optional** coupled move; do not promote it to a
production sampler default or declare robust joint mixing. Cold coupled-move
acceptance rises from 65.81% to 81.53%; every scalar bulk ESS improves at equal
primary-gradient work: sigma8 2.97 times, b_delta 1.31, b_delta2 5.56, b_K2 4.03
and b_laplacian 1.59. Qualified returns rise from 9 to 13 and every candidate
ensemble completes a return, so the candidate passes the original transport gate.
Neither run has divergences.

All six white-field projection bulk ESS values worsen, to 0.18–0.62 times the
reference. All twelve original coordinate gates remain failures. Sigma8 is close
but still fails: Rhat=1.0102, bulk ESS=360.02 (original requirements <1.01 and >=400).
No threshold or budget is relaxed. Improved scalar/transport behavior is useful,
but does not establish the joint posterior.

The post-hoc physical_diagnostics.py supplement multiplies the six in-band white
projections by alpha, giving physical initial-mode coordinates up to fixed
reference-spectrum factors. Original white diagnostics reproduce to 1e-9. These
extra monitors were requested after seeing the outcome and do not redefine any
predeclared gate. Candidate physical-mode Rhat reaches 1.2803 and bulk ESS is
11.3–23.3; all six physical-mode bulk ESS values worsen relative to the baseline.
The field problem therefore persists in physical coordinates, not only in the
whitened parameterization. The coupled alpha/support move preserves these physical
coordinates and cannot directly make a physical-field mode jump.

| Physical initial projection | Baseline Rhat / active-support Rhat | Baseline bulk ESS / active-support bulk ESS |
|---|---:|---:|
| alpha_white_cos100 | 1.2324 / 1.2185 | 27.52 / 14.06 |
| alpha_white_sin100 | 1.0974 / 1.2329 | 43.04 / 13.09 |
| alpha_white_cos110 | 1.1014 / 1.2630 | 53.58 / 11.92 |
| alpha_white_sin110 | 1.0605 / 1.1292 | 65.97 / 23.30 |
| alpha_white_cos001 | 1.1516 / 1.2040 | 27.08 / 14.74 |
| alpha_white_sin001 | 1.1364 / 1.2803 | 41.84 / 11.32 |

Next proposed experiment: joint physical-field/bias moves. A concrete candidate is
a symmetric fixed-donor difference proposal in the stored white initial-band
coordinates plus physical scalar coordinates, with full posterior MH and out-of-support rejection.
Freeze the donor pool and ordered-pair selection before sampling; symmetric pair
ordering gives a symmetric translation proposal, so no learned-density or new
parameterization Jacobian is needed. Test detailed balance and multimodal analytic
references, then a bounded actual-chain comparison. This is a proposal only;
no donor kernel or additional chains were run here. Robust broad-prior calibration
and production-scale/catalog validation remain open.

## Validation and limits

Serial tests pass 121 checks. Actual MPI tests pass 225 checks at each of 1/2/4 ranks, exercising both FFTWGrid slabs and PencilGrid pencils. They cover global rank across shared/compact meshes and narrow likelihoods; independent Fourier tones; orthogonality/self-adjointness/idempotence; prior-null preservation; 1LPT/2LPT prediction/energy-gap/inverse identities; signed global Jacobian and full-target MH reciprocity; collective rank-local input/allocation failures. Independent Gaussian marginals/quadrature cover 1/64-dimensional priors and detect the intentionally wrong full-dimension Jacobian on an active-only move. The actual MPI communicator sizes are verified, not inferred from process launch counts.

Native controls preserve legacy draws/summary/target, reproduce active-move full checkpoint/summary/draws across restart and reject changed, missing or malformed policy without native writes. Real four-chain pause, exclusive-writer, missing-ledger and original-target controls pass. Candidate chains also perform complete-ensemble replay. Frozen sources, unit binaries/logs, baseline records/checkpoints and policies are reverified after sampling.

No chain is extended, replaced, pooled or retuned. This selected-data development study is not independent calibration or a real-catalog analysis. Distributed kernel verification is separate from production MPI replica orchestration. [Method and equations](../../../../validation/galaxy/support_interweave/README.md), controls.json, summary.json and PROVENANCE.json retain all outcomes. Raw builds, complete hash inventories, chains, logs, native-hooks.patch and support-comparison.png stay at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-support-interweave`.
