# Amplitude/field interweaving pilot

Recorded 18 November 2026. **Frozen mixing gate: FAIL.**
[Implementation/target argument](../../../../validation/galaxy/interweave/README.md);
[frozen protocol](../../../../validation/galaxy/interweave/plan.json);
[machine-readable report](summary.json); [blocked baseline](../blocked_pilot/RESULTS.md).

## Kernel and comparison

The only algorithm change from the blocked baseline is four amplitude/field MH
attempts after each unchanged outer sweep. A reflected normal amplitude increment
of width 0.10 proposes alpha_new; all 64 stored real coordinates rescale by
r=alpha/alpha_new. The full-posterior log acceptance ratio is
`U_old - U_new + 64 log(r)`. The proposal's real-coordinate Jacobian includes
prior-only directions. It is not a target penalty. Biases stay fixed in this move.
The sampler evaluates the full existing posterior rather than assuming numerical
likelihood cancellation. No amplitude prior, bias prior or noise law changes.

The N4/N8/N16 PSWF synthetic target, physical cuts, data, four dispersed initial
states, field/scalar component settings and seeds, 256 discarded warmup sweeps,
1024 retained draws and eleven-coordinate gates are identical to the blocked
baseline. The new component stream is keyed by chain and outer index. Added target
evaluations cost time; this is not a matched-compute comparison. Previous primary
and all replay draws remain separate. No post-run extension, retuning, chain
exclusion or threshold change was made.

This is a serial validation prototype. Existing production sampler/model APIs are
unchanged. There is no active-only projection or production MPI rescaling API.

## Numerical/invariant-target validation

The 26 dedicated checks pass. Independent analytic Gaussian marginals integrated
in alpha agree with toy chains using uniform/truncated-Gaussian amplitude priors,
one/three white dimensions, observed/unobserved likelihoods and unused Gaussian
coordinates. Exact conditional field refreshes make these tests ergodic; rescaling
alone would keep alpha*white fixed. Unused coordinates retain unit Gaussian
variance. Signed Jacobian, reciprocal transform/log acceptance, reflected bounds,
invalid controls and nonfinite proposal rejection are checked separately.

Shared/compact 1LPT/2LPT predictions before/after amplitude rescaling agree to
1.71659e-16 maximum scaled error. The full-posterior difference matches the
actual white-prior difference to 8.50046e-15. Scalar cache/full-target checks
at 18 points and 72 bias finite differences also pass. Earlier FixedIC independent
quadrature tests remain applicable; production FixedIC sources are unchanged.

Four uninterrupted outer sweeps match two plus two across fresh processes in raw
checkpoint and summary bytes. Wrong-chain and checksum-corrupt states reject
without advancing CURRENT. During the campaign a copy at chain 0 sweep 384
reproduces 64 retained records, predictions and numerical summary exactly. Replay
draws are excluded; exactly 4096 primary draws enter diagnostics. Versioned target,
plan, chain and rescaling counters prevent mixing this format with older pilot
states. Component seeds cross no partially consumed saved stream boundary.
This is process-restart validation, not power-loss durability or MPI composition.

## Results

All eleven coordinates require computable rank-normalized split/folded Rhat < 1.01,
bulk ESS >= 400 and tail ESS >= 400, plus zero sampling field divergences in each
chain. Shared analysis uses the same physical scalars, six fixed field projections
and FML::MCMC diagnostics as both preceding campaigns.

| Coordinate | Blocked Rhat | Interwoven Rhat | Blocked bulk ESS | Interwoven bulk ESS | Interwoven tail ESS | Gate |
|---|---:|---:|---:|---:|---:|---|
| sigma8_ratio | 1.4774 | 1.5233 | 7.7 | 7.4 | 34.6 | FAIL |
| b_delta | 1.5496 | 1.5473 | 7.3 | 7.2 | 29.9 | FAIL |
| b_delta2 | 1.2286 | 1.0924 | 14.8 | 39.5 | 502.6 | FAIL |
| b_K2 | 1.3452 | 1.5192 | 9.5 | 7.4 | 30.5 | FAIL |
| b_laplacian | 1.2732 | 1.2790 | 11.4 | 10.9 | 96.2 | FAIL |
| cos100 | 1.5673 | 1.0548 | 7.0 | 62.8 | 114.3 | FAIL |
| sin100 | 1.4331 | 1.3599 | 8.2 | 9.4 | 42.1 | FAIL |
| cos110 | 1.5919 | 1.4287 | 6.9 | 8.1 | 16.6 | FAIL |
| sin110 | 1.3622 | 1.5315 | 9.3 | 7.3 | 44.6 | FAIL |
| cos001 | 1.5761 | 1.0169 | 7.2 | 286.4 | 358.2 | FAIL |
| sin001 | 1.5564 | 1.4644 | 7.2 | 8.0 | 15.6 | FAIL |

Failed coordinates: sigma8_ratio, b_delta, b_delta2, b_K2, b_laplacian, cos100, sin100, cos110, sin110, cos001, sin001.
Field divergence counts: 0, 0, 0, 0.
Field mean acceptance: 0.9762, 0.9727, 0.9840, 0.9775.
Scalar joint-MH acceptance: 0.5925, 0.1182, 0.5801, 0.6001.
Amplitude/field rescaling acceptance: 0.6040, 0.5522, 0.6221, 0.6184.
Acceptance rates alone do not establish joint exploration.

Summed main-chunk per-process wall time is
741.0 s blocked and
827.2 s interwoven
(1.116 times). Both run two concurrent serial chains.
Timing includes target/work allocation, scalar caches, rescaling evaluations and
output/checkpoints; preflight/replay are excluded. Host load and added work limit
production extrapolation. ESS/time is saved, but an unmixed baseline cannot
calibrate an asymptotic efficiency factor.

## Interpretation and remaining work

The declared mixing gate still fails. Preserve this benchmark; amplitude/field rescaling alone has not established a scientifically usable joint sampler. Diagnose the remaining chain-dependent configurations before scaling or claiming marginalized sigma8 convergence.

Pooled scalar quantiles/correlations are descriptive when gates fail. They do not
establish an amplitude shift or coverage. Six projections do not validate all
field coordinates. Residual bands use fitted data and exclude fresh observation
noise. Same-model synthetic sampling does not establish physical convergence,
external LEFTfield agreement, N-body/HOD robustness or real-catalog inference.

Raw primary/replay chains, checkpoints, exact commands, hashes and figures remain
outside the library under workspace `outputs/fieldlevel-interweave-pilot/`.
Figures under `campaign/`: `comparison.png`, `joint-summary.png`, `traces.png`,
`scalar-posterior.png`, `residuals-diagnostics.png`. Item 52 remains ongoing.

## Chain separation diagnosis from the existing draws

No additional Monte Carlo draws were run. A direct decomposition of the saved
full-field prior and mode residuals gives:

| Chain | Median amplitude ratio | Median white prior | Median data quadratic | Median relative U |
|---|---:|---:|---:|---:|
| 1 | 0.6574 | 30.36 | 10.94 | 41.70 |
| 2 | 1.4605 | 44.84 | 18.63 | 63.90 |
| 3 | 0.7114 | 30.58 | 10.82 | 41.93 |
| 4 | 0.7311 | 30.71 | 10.69 | 41.55 |

Relative U includes `0.5 sum white^2 + 0.5 sum weight |prediction-data|^2/variance`;
uniform scalar-prior and shared likelihood constants cancel. The median sum
need not equal the sum of medians. Chain 2 remains near amplitude=1.46, b_K2=3.75
and b_laplacian=89, while the other chains visit different bias/field
configurations. Point-energy comparisons do not determine configuration masses;
the separated chain cannot be dropped to manufacture convergence.
[Per-chain decomposition](branch-diagnostics.json) records the descriptive values.
This motivates a posterior-geometry/barrier audit and a separately declared
tempering/global-move comparison; rescaling is insufficient by itself.

The runner also repairs a deliberately stale summary from authoritative native
checkpoint state using zero-transition resumes, with no new draws or checkpoint
changes. Changed binaries and old blocked checkpoint formats are rejected.
