# Prior sensitivity on three completed synthetic datasets

The separately frozen four-case analysis is complete. All twelve dataset/case
**overlap and mean-shift precision screens pass**. This is importance reweighting
of existing cold chains, not newly validated alternative-prior MCMC or SBC.

Every dataset uses all four 2048-draw cold chains. All eleven monitored scalar
and field-projection coordinates are reported in summary.json. No source chain,
checkpoint, data realization, physical band or likelihood was changed. The prior
choices and screens were saved before computing the sensitivity estimates.

## Declared prior changes

Baseline physical scalar priors are independent uniforms on alpha in [0.5,1.5],
b_delta in [0,4], b_delta2 and b_K2 in [-4,4], and b_laplacian in [-100,100].
All 64 latent white coordinates keep their standard Gaussian prior.

- `log_amplitude`: normalized density 1/[alpha log(1.5/0.5)] on the same interval.
- `regularized_nuisance`: independent zero-centered truncated Gaussians for
  b_delta2, b_K2 and b_laplacian, with SD 2, 2 and 50. Their supports stay fixed;
  scales are half the original half-widths and are not chosen by truth recovery.
- `combined`: both changes. b_delta and white priors stay fixed in every case.

Normalized prior-density ratios are implemented explicitly. Constants cancel in
posterior moments, but independent quadrature verifies the full normalized ratios.
No change of sampled coordinates or extra field Jacobian is needed here: this is
a density change on stored physical alpha, not a new sampler transformation.

## Amplitude results

Mean shifts and their paired MCSE are measured in each dataset's baseline
posterior SD. SD uses normalized importance moments (population convention),
and quantiles use midpoint weighted-CDF interpolation; baseline tail endpoints
can therefore differ slightly from earlier unweighted NumPy quantiles.

| Dataset | Prior case | Amplitude mean | Shift / baseline SD | Paired MCSE / baseline SD | Minimum overlap fraction |
|---|---|---:|---:|---:|---:|
| original | baseline | 0.752233 | +0.0000 | 0.0000 | 1.0000 |
| original | log_amplitude | 0.703195 | -0.2301 | 0.0109 | 0.9410 |
| original | regularized_nuisance | 0.803492 | +0.2405 | 0.0159 | 0.6256 |
| original | combined | 0.746147 | -0.0286 | 0.0143 | 0.6395 |
| mock1 | baseline | 0.829037 | +0.0000 | 0.0000 | 1.0000 |
| mock1 | log_amplitude | 0.760083 | -0.2680 | 0.0121 | 0.9214 |
| mock1 | regularized_nuisance | 0.860048 | +0.1205 | 0.0161 | 0.5819 |
| mock1 | combined | 0.787200 | -0.1626 | 0.0157 | 0.5591 |
| mock2 | baseline | 0.847192 | +0.0000 | 0.0000 | 1.0000 |
| mock2 | log_amplitude | 0.791026 | -0.2428 | 0.0093 | 0.9379 |
| mock2 | regularized_nuisance | 0.941860 | +0.4092 | 0.0281 | 0.5747 |
| mock2 | combined | 0.880186 | +0.1426 | 0.0169 | 0.6147 |

The log-amplitude prior lowers the mean by 0.23–0.27 baseline SD across the three
datasets. Nonlinear-bias regularization raises it by 0.12–0.41 SD. The combined
changes partly cancel and need not reproduce either separate result. The largest
monitored-coordinate shift is about -0.558 SD for b_delta2 in mock 2.

These shifts exceed their estimated paired Monte Carlo errors. The small target
has materially prior-sensitive scalar inferences. This does not select a preferred
prior or establish an amplitude bias. In particular, moving a mean toward the known
truth is not a criterion for choosing priors. The baseline results remain intact.

## Reliability and limitations

The declared screens require source monitored convergence, pooled and every-chain
importance overlap fraction >=0.5, maximum normalized weight relative to uniform
<=10, and paired mean-shift MCSE <=0.1 baseline SD for every monitored coordinate.
All pass: the smallest overlap is 0.5591; largest relative weight is 5.394; largest
paired shift MCSE is 0.0387 SD. These thresholds screen the importance estimate;
they are not a convergence theorem for a new target.

Paired influence functions retain the covariance between original and reweighted
means. MCSE is the maximum of batch estimates at lengths 64/128/256 and the
between-independent-chain mean estimate. Serially correlated records are never
treated as 8192 independent draws. Importance overlap is not MCMC ESS. The quoted
MCSE applies to mean shifts, not to tail quantiles or interval inclusion.

All alternative priors have the original support. No claim about expanded-support
priors or unseen modes follows from this analysis. Fixed-truth posterior summaries
for three realizations do not measure coverage or prior-predictive rank uniformity.

Independent tests verify prior normalization, a nonuniform-likelihood posterior
against quadrature, identity/paired cancellation, failed-source and failed-overlap
propagation, invalid draws, changed-plan rejection and repeated-source rejection.
The shared weighting controls verify analytic Gaussian tilts and an autocorrelated
AR(1) MCSE reference. All source/input hashes are checked before analysis; source
CURRENT checkpoints and cold records remain unchanged. The plot was inspected.

## Reproduction and next stage

[Driver and instructions](../../../../validation/galaxy/prior_sensitivity/README.md),
[declared plan](../../../../validation/galaxy/prior_sensitivity/plan.json),
FROZEN.json and summary.json retain the method and all outcomes. Raw inputs,
figure and checks stay outside the library at:
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-prior-sensitivity/`.
The figure is `campaign/prior-sensitivity.png`.

Next: implement independent prior-predictive truth controls and validate them
before a separately frozen nonlinear SBC campaign. The proposed finite-budget
protocol is in [CALIBRATION_PLAN.md](../../../../validation/galaxy/prior_sensitivity/CALIBRATION_PLAN.md).
No nonlinear SBC campaign has been launched and no priors are changed by this study.
