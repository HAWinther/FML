# Independent fixed-truth mock recovery stress test

Both predeclared fresh mock budgets completed. Computational gate: **PASS**.
This is a two-realization fixed-truth recovery stress test, **not** prior-predictive
SBC or coverage calibration. No interval-inclusion success gate was invented.

Each mock has four fresh chains, 256 discarded warmup and 2048 retained draws per
chain, 11 replicas and four exchange passes. All physical priors, numerical
settings, kernels and original convergence/transport gates remain fixed. No
prefix reuse, seed replacement, excluded chains/mocks, extensions or pooling
across mocks occurred. All failed and successful outcomes are retained.

Scalar truth is [1, 1.6, .3, -.2, 7], with sigma8/reference sigma8 as the first
coordinate. Mock 1 uses truth/noise/initial-base seeds 5101/5102/5103; mock 2 uses
6101/6102/6103. Layout 3 uses full 64-bit disjoint mock/sweep/purpose keys.
The prior original dataset and all earlier results remain unchanged.

## Computational and amplitude recovery results

All eleven monitored coordinates separately require rank-normalized split/folded
Rhat <1.01 and bulk/tail ESS >=400, plus the unchanged connectivity and zero
measured positive-beta divergence gates. Recovery estimates from failed-mixing
mocks are explicitly provisional.

| Mock | Computational gate | Failing coordinates | Amplitude Rhat | Bulk ESS | Tail ESS | Amplitude median | Central 95% interval | True ratio 1 included? |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | PASS | 0/11 | 1.0078 | 582.112 | 1211.24 | 0.760836 | [0.512891, 1.42187] | yes |
| 2 | PASS | 0/11 | 1.00277 | 861.213 | 1493.45 | 0.800776 | [0.528189, 1.40435] | yes |

An interval miss is retained as a recovery finding; it does not justify retuning
priors or extending chains until the known truth is included. Two mocks cannot
determine a coverage rate, and a fixed-truth experiment is not an SBC rank test.
CDF fractions below are posterior ranks descriptive of these realizations.

## Mock 1

Computational gate: **PASS**; recovery summaries conditional on the passing monitored mixing checks.

Qualified cold/prior-refresh/cold round trips: [22, 21, 17, 18]. Measured positive-beta field divergences: 0.

| Coordinate | Truth | Mean | Median | Posterior SD | Truth CDF fraction | 50% / 90% / 95% inclusion |
|---|---:|---:|---:|---:|---:|---|
| sigma8_ratio | 1 | 0.829037 | 0.760836 | 0.25731 | 0.755493 | no / yes / yes |
| b_delta | 1.6 | 2.54597 | 2.58195 | 0.771567 | 0.123901 | no / yes / yes |
| b_delta2 | 0.3 | 1.50903 | 1.42461 | 0.981297 | 0.0944824 | no / yes / yes |
| b_K2 | -0.2 | 1.26978 | 1.63792 | 1.97557 | 0.222412 | no / yes / yes |
| b_laplacian | 7 | 22.8338 | 30.4989 | 53.0807 | 0.359985 | yes / yes / yes |
| cos100 | -1.23802 | -0.715376 | -0.720747 | 0.277172 | 0.0279541 | no / no / yes |
| sin100 | 0.773844 | 0.746827 | 0.731455 | 0.219184 | 0.582764 | yes / yes / yes |
| cos110 | 1.10576 | 0.902918 | 0.886654 | 0.320637 | 0.770874 | no / yes / yes |
| sin110 | 1.12038 | 0.501107 | 0.491021 | 0.246116 | 0.987793 | no / no / no |
| cos001 | 0.227646 | 0.206462 | 0.209536 | 0.191894 | 0.543457 | yes / yes / yes |
| sin001 | 0.780233 | 0.346987 | 0.350419 | 0.245964 | 0.965942 | no / no / yes |

Full per-coordinate diagnostics, interval endpoints and fitted-data residual shells are retained in summary.json. Residuals use fitted data; no independent predictive-check claim is made.

## Mock 2

Computational gate: **PASS**; recovery summaries conditional on the passing monitored mixing checks.

Qualified cold/prior-refresh/cold round trips: [22, 15, 17, 12]. Measured positive-beta field divergences: 0.

| Coordinate | Truth | Mean | Median | Posterior SD | Truth CDF fraction | 50% / 90% / 95% inclusion |
|---|---:|---:|---:|---:|---:|---|
| sigma8_ratio | 1 | 0.847192 | 0.800776 | 0.231378 | 0.762695 | no / yes / yes |
| b_delta | 1.6 | 2.46552 | 2.48285 | 0.619289 | 0.0822754 | no / yes / yes |
| b_delta2 | 0.3 | 2.28962 | 2.30224 | 0.997353 | 0.0134277 | no / no / no |
| b_K2 | -0.2 | -0.862068 | -0.772204 | 1.4076 | 0.680298 | yes / yes / yes |
| b_laplacian | 7 | 12.9273 | 19.3912 | 54.7266 | 0.429199 | yes / yes / yes |
| cos100 | 2.64986 | 1.85836 | 1.84635 | 0.356698 | 0.980591 | no / no / no |
| sin100 | -0.0349218 | -0.0209053 | -0.02306 | 0.158052 | 0.470337 | yes / yes / yes |
| cos110 | -1.09277 | -0.949969 | -0.930461 | 0.256879 | 0.263062 | yes / yes / yes |
| sin110 | 0.977438 | 1.04829 | 1.02198 | 0.285626 | 0.428345 | yes / yes / yes |
| cos001 | 1.70127 | 1.30752 | 1.29934 | 0.295135 | 0.911743 | no / yes / yes |
| sin001 | 1.37477 | 1.0791 | 1.07566 | 0.291121 | 0.883911 | no / yes / yes |

Full per-coordinate diagnostics, interval endpoints and fitted-data residual shells are retained in summary.json. Residuals use fitted data; no independent predictive-check claim is made.

## Interpretation of the recovery findings

The generating amplitude ratio 1 lies in both central 90% and 95% intervals.
The medians are approximately 0.761 and 0.801, with broad intervals; the truth CDF
fractions are about 0.755 and 0.763. These small targets do not tightly determine
the amplitude, and the central estimates should not be described as unbiased.

Other central 95% interval misses are retained: sin110 in mock 1, and b_delta2
plus cos100 in mock 2. Both datasets pass all computational gates. With two
fixed-truth realizations and correlated parameter summaries, these misses neither
establish a coverage problem nor certify calibration. No prior was adjusted using
the known truth. A separately designed prior-sensitivity/calibration study is
needed before a statistically calibrated recovery claim.

## Controls, validation and remaining scope

The study plans were declared before generating either mock. No new mock was
chosen by its recovery outcome. Source settings/metadata and namespaces enter
checkpoint identity. Enumeration verifies 608256 distinct sampler keys covering
both full budgets. Truth/noise controls are independently exercised; differing
data are reproducible across the four ensembles within each mock.

Preflight passes 25 exchange checks, old/new continuation on a real 11-replica
checkpoint, all eight fresh/restarted short trajectories and 512 indexed cold
records, namespace/data-seed rejection, independent per-mock analysis, and whole-
study SIGTERM pause/resume matching individual runs. Both new data realizations
pass actual tempered-target/cache and prior-refresh checks. Actual full-budget
replays match all replica states and cold records. Final fresh-process reloads
preserve all eight checkpoints/summaries exactly.

The truth-field projections use an independent orthonormal FFT calculation and
match the direct real-space cosine/sine basis. Central intervals and misses are
unit checked. Amplitude ratios are not absolute sigma8 values. Original numerical
stability was established on the original mock; it is not re-certified for all
new mock posterior states here. No real catalog, independent coverage-calibration
claim, larger target or production MPI replica orchestration is established.

[Study plan](../../../../validation/galaxy/tempering/plan_mocks.json),
[mock 1](../../../../validation/galaxy/tempering/plan_mock1.json),
[mock 2](../../../../validation/galaxy/tempering/plan_mock2.json),
[runner/method](../../../../validation/galaxy/tempering/README.md), FROZEN.json and
summary.json preserve all controls and outcomes. Raw executables, logs, indexed
records, checkpoints, plots and review evidence live outside the library:
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-independent-mocks/`.
Study plot: campaign/mock-recovery.png. Each mock also has its complete traces,
scalar distributions, residual/ESS plots and replica-connectivity figures.
