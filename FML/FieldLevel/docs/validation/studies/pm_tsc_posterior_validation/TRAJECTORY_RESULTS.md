> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Shorter trajectories at matched PM cost (#52)

The shorter schedule improves median tail efficiency 55% and removes all four
previous tail-ESS failures. Amplitude, noise and posterior-potential bulk efficiencies improve
about 2.2x, while median bulk efficiency falls 69%. Bulk failures increase from
20 to 182 of 248 features. The minimum bulk ESS rises from 110 to 189, but the
all-feature criterion still fails. Keep this as an opt-in experiment; it does
not justify replacing the longer-trajectory schedule.

Four chains use the same N16/M16, 16-step target and fixed field metric as the
preceding [metric experiment](METRIC_RESULTS.md). The trajectory maximum changes
from 8 to 4 leapfrog steps (jittered 4..8 to 2..4), and retained draws increase
from 128 to 320 per chain. Warmup stays 64, with identical seeds/starts, priors,
step-size adaptation and convergence thresholds. No native C++ source changes
were needed. Existing runner defaults remain 128 draws and maximum 8 steps.

With total PM cost included, the median bulk ESS efficiency ratio is
**0.306**, and the median tail ratio is
**1.549**, relative to the preceding fixed-metric
campaign. The actual cost ratio is **1.0002**.
These finite-chain estimates do not establish asymptotic efficiency or coverage.

| Quantity | Maximum 8 steps, 128 draws | Maximum 4 steps, 320 draws |
|---|---:|---:|
| PM forward/adjoint evaluations, including metric setup/warmup | 4828 | 4829 |
| Metric setup evaluations, charged once | 252 | 252 |
| Maximum Rhat | 1.036121 | 1.023945 |
| Minimum bulk ESS | 110.11 | 189.17 |
| Minimum tail ESS | 166.32 | 264.73 |
| Features failing bulk ESS >400 | 20 | 182 |
| Features failing tail ESS >200 | 4 | 0 |
| All original mixing criteria passed | False | False |

Strict status is passed=False, consistency_passed=True,
mixing_passed=False. Failed checks: all bulk ESS > 400.
Every retained draw, including repeats, enters all 248 feature diagnostics.
The criteria remain Rhat <1.05, bulk ESS >400, tail ESS >200, mean acceptance >.5
per chain and zero retained divergences. There is no diagnostic thinning,
post-warmup removal or choice of favorable subsets. Intervals remain exploratory
if these criteria fail; passing them alone would not establish repeated-mock coverage.

| Feature | Bulk ESS efficiency ratio | Tail ESS efficiency ratio |
|---|---:|---:|
| A | 2.199 | 2.763 |
| N | 2.217 | 2.099 |
| potential | 2.236 | 1.995 |
| norm2 | 0.681 | 1.281 |

*Archived figure: Efficiency comparison (`pm_tsc_posterior_validation/short_trajectories/comparison/efficiency.png`). See [archive guide](../../../ARCHIVE.md).*

The next controlled candidate is an intermediate maximum of 6 steps (jittered
3..6), with 192 retained draws and 64 warmup. Its expected work is also 1152 force
evaluations per chain. This is a proposed experiment, not a demonstrated cure
for the slow modes; it has not been run in this report.

## Controlled differences and accounting

Expected trajectory work is identical: (64+128)*6=(64+320)*3=1152 force
evaluations per chain. Actual random trajectory lengths and early termination
can change these totals. The comparison explicitly permits the draw-count,
trajectory-length and prediction-stride differences and requires actual costs
within 5%. It continues to require identical target settings, priors, seeds,
warmup/adaptation settings and metric hashes. Saved plans must agree with native
metadata. All changes are listed in the comparison JSON.

The short campaign computes its own metric, using the same 252 evaluations;
its matrix hash matches the preceding frozen matrix exactly. Setup is charged
once per campaign, not treated as free, charged four times, or amortized over
hypothetical future use. Warmup cost is included. There are twice as many outer
iterations (384 vs192), hence twice as many cheap scalar updates. Shorter warmup
trajectories also leave more of the fixed PM budget for retained sampling.
This is a comparison of complete sampling schedules, not a change of trajectory
length at fixed retained-draw count.

Prediction stride changes from 8 to20 so both campaigns save 16 predictions per
chain. This limits output only; all 1280 new retained draws enter diagnostics.
Predictive-only forwards are excluded equally from the PM efficiency denominator.
The matrix itself and exact target/Jacobian remain unchanged.

## Validation and saved evidence

No C++ interfaces or native algorithms changed, so the preceding serial,
slab/pencil MPI, sanitizer and Gaussian-moment validation remains applicable to
the reused binary. Its source hashes are verified. New controls reject invalid
counts/strides. The comparison retains strict default protocol matching, requires
an explicit trajectory-comparison option, checks saved metadata and refuses an
unmatched PM budget. Native target checks and fresh metric checks run again in
the full campaign. Raw features, native diagnostics and copied-report analysis
are independently replayed after completion.

[Protocol (archived)](../../../ARCHIVE.md), [commands (archived)](../../../ARCHIVE.md),
[diagnostics (archived)](../../../ARCHIVE.md),
[full summary and failure flags (archived)](../../../ARCHIVE.md),
[comparison and protocol changes (archived)](../../../ARCHIVE.md),
[raw states (archived)](../../../ARCHIVE.md),
[control checks (archived)](../../../ARCHIVE.md) and
[native source hashes (archived)](../../../ARCHIVE.md) are retained.
All other chain files sit beside chain0. See [README](README.md) for reproduction.

Scope remains one finite N16 realization. N32 posterior mixing, particle
convergence, broader noise/signal regimes and repeated-mock coverage are still
unvalidated. The dense metric is a small active-space experiment, not a scalable
production full-grid preconditioner. No production sampler default changes.
