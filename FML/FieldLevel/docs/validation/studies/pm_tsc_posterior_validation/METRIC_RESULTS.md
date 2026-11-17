> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Fixed field metric: bounded N16 pilot (#52)

The fixed metric raises median bulk ESS per PM evaluation **2.50x** over the
alternating sampler after setup cost, while median tail efficiency falls about
9.4%. Field-power bulk efficiency improves 1.70x; energy and noise remain slow.
The strict ESS gates still fail, so this remains an opt-in experiment and does
not establish calibrated posterior uncertainty.

A fixed dense field metric was tested on the same N16/M16, 16-step target,
frozen data, priors, common reference initialization and active coordinates.
The alternating scalar kernel, 64 warmup + 128 retained draws per chain, four
chains, jittered 4..8 leapfrog steps and original diagnostic thresholds remain.
The original joint sampler stays the default.

Compared with the alternating sampler, the median bulk ESS per actual PM
evaluation ratio is **2.502**; the median tail
ratio is **0.906**. Against the original joint
sampler these ratios are 2.537 and
0.894. These are finite-pilot estimates, not
asymptotic guarantees or a production benchmark.

| Quantity | Alternating, uniform field mass | Fixed field metric |
|---|---:|---:|
| PM forward/adjoint evaluations, including setup and warmup | 4576 | 4828 |
| Metric setup evaluations (one shared build) | 0 | 252 |
| Maximum Rhat | 1.028900 | 1.036121 |
| Minimum bulk ESS | 148.44 | 110.11 |
| Minimum tail ESS | 158.25 | 166.32 |
| All original mixing criteria passed | False | False |

Strict report status: passed=False, consistency_passed=True,
mixing_passed=False. Failed checks: all bulk ESS > 400, all tail ESS > 200.
Intervals and predictions remain exploratory whenever mixing criteria fail.
The criteria are Rhat <1.05, bulk ESS >400, tail ESS >200 for all 248 features,
acceptance >.5 per chain and zero retained divergences. There is no post-warmup
thinning, burn-in removal or selection of favorable draws.

| Feature | Bulk efficiency ratio vs alternating | Tail efficiency ratio |
|---|---:|---:|
| A | 0.944 | 0.808 |
| N | 0.803 | 1.053 |
| potential | 0.703 | 1.009 |
| norm2 | 1.699 | 1.257 |

*Archived figure: Efficiency against alternating sampler (`pm_tsc_posterior_validation/preconditioned/comparison/efficiency.png`). See [archive guide](../../../ARCHIVE.md).*

The pattern suggests testing shorter trajectory lengths at a matched total PM
budget next. Large linear-mode bulk ESS can coexist with slow quadratic energy
or near-zero-mode squares. This pilot alone does not establish the cause of the
slow features. Longer-chain uncertainty validation is still needed.

*Archived figure: Reference metric spectrum (`pm_tsc_posterior_validation/preconditioned/analysis/metric_spectrum.png`). See [archive guide](../../../ARCHIVE.md).*

## Metric and accounting

Central gradient differences at h=1e-4 measure the field Hessian at the common
unperturbed reference field, A=0 and N=log(1e-4). The symmetrized matrix passes
four refined directional checks. Cholesky regularization is allowed only for
positivity; actual ridge, errors and setup time are saved in
[the native setup metadata (archived)](../../../ARCHIVE.md).
The full target's metric eigenvalues span 8.4889 to
57.1707, a ratio of 6.735.

The 122x122 matrix is fixed before warmup and shared by all chains. In the
existing HMC convention, M is precision-like: p=L*z, LL^T=M, velocity=M^-1*p.
The exact posterior and Metropolis correction are unchanged. No new coordinate
transformation or determinant is introduced; the existing 122*A/2 Jacobian stays.
The production HMC engine is unchanged.

Setup costs **252 actual forward/adjoint evaluations once for the campaign**:
244 coordinate perturbations and eight directional checks. These are included
in the efficiency denominator, not treated as free or charged four times.
Every new density evaluation performs the same forward/adjoint pair as before.
Scalar conditionals and gradient refresh still reuse the reference cache without
additional PM evaluations. Predictive-only forward calls are excluded equally.
The setup cost is not amortized over hypothetical future chains.

## Validation

Seventeen short native cases check metric construction and sampling in serial,
slab MPI at 1/2/4 ranks, and pencil MPI at 1/2/4 ranks. These include a sanitized
sampling run and exact default/alternating-path regression checks. The short
metric matrices agree within 2e-7 and sampled states within 2e-8 across backends.
The PM finite-difference setup itself was not run under sanitizers; factorization,
mass application, sampling and a separate Gaussian test were.

An independent 50,000-draw correlated-Gaussian test verifies inverse-mass algebra,
factorization, rejection of indefinite matrices and analytic posterior moments;
it also passes ASan/UBSan. Malformed metric files (missing, wrong row count,
ragged, nonfinite, asymmetric or indefinite) fail, including a two-rank collective failure.
Matrix values are checked for replication before sampling. The actual 16-step
target passes the existing density/gradient/Jacobian checks. The saved matrix
hash, independent positive eigenvalues and derivative checks enter analysis.

## Saved evidence and scope

[Protocol (archived)](../../../ARCHIVE.md), [commands (archived)](../../../ARCHIVE.md),
[all-feature diagnostics (archived)](../../../ARCHIVE.md),
[summary and failure flags (archived)](../../../ARCHIVE.md),
[comparison to alternating (archived)](../../../ARCHIVE.md),
[comparison to original joint HMC (archived)](../../../ARCHIVE.md),
[raw states (archived)](../../../ARCHIVE.md),
[validation (archived)](../../../ARCHIVE.md) and
[source hashes (archived)](../../../ARCHIVE.md) are retained.
All other chain files sit beside chain0. See [README](README.md) for reproduction.

This is a small active-space diagnostic. Dense O(d^2) storage and O(d) PM setup
calls do not scale to full-grid inference. A scalable production metric would
need operator/Fourier structure. N32 mixing, repeated-mock coverage and broader
resolution convergence remain outside this bounded experiment.
