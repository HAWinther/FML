> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Intermediate trajectories at matched PM cost (#52)

The intermediate schedule leaves 18 bulk-ESS failures (versus 20 for maximum 8
and 182 for maximum 4) and one tail failure (versus 4 and 0). The minimum bulk ESS
is 189.74 and minimum tail ESS 199.98; the latter stays failed against the strict
>200 criterion. Maximum Rhat improves to 1.0193 and no retained divergences occur.
Joint posterior mixing remains insufficient.
The next useful validation is a longer run at fixed settings with a declared
larger compute budget. These tuning pilots do not establish calibrated
uncertainty estimates. Defaults remain unchanged.

The intermediate schedule uses maximum 6 leapfrog steps (jittered 3..6),
64 warmup and 192 retained draws per chain. Four chains keep the same N16/M16,
16-step target, frozen physical reference, priors, 122 active coordinates and
Jacobian, seeds/starts, fixed metric and adaptation policy. No native or runner
code changes were needed; production and validation defaults stay unchanged.

Compared with the maximum-8-step schedule, median bulk ESS per PM evaluation is
**0.437x** and median tail efficiency is
**1.369x**. Relative to the maximum-4-step
schedule, these ratios are 1.354x and
0.878x. These finite-pilot measurements are
not asymptotic efficiency estimates or repeated-mock coverage.

| Quantity | Maximum 8, 128 draws | Maximum 6, 192 draws |
|---|---:|---:|
| Total PM forward/adjoint evaluations, including setup and warmup | 4828 | 4849 |
| Maximum Rhat | 1.036121 | 1.019325 |
| Minimum bulk ESS | 110.11 | 189.74 |
| Minimum tail ESS | 166.32 | 199.98 |
| Features failing bulk ESS >400 | 20 | 18 |
| Features failing tail ESS >200 | 4 | 1 |
| All mixing criteria passed | False | False |

Strict status: passed=False, consistency_passed=True,
mixing_passed=False. Failed checks: all bulk ESS > 400, all tail ESS > 200.
Every retained state, including repeats, enters all 248 diagnostics. Gates remain
Rhat <1.05, bulk ESS >400, tail ESS >200 for every feature, mean field-HMC
acceptance >.5 per chain and zero retained divergences. There is no diagnostic
thinning, post-warmup removal or selection of favorable draws. Intervals remain
exploratory if the criteria fail; passing them alone would not establish coverage.

| Feature | Bulk efficiency ratio vs maximum 8 | Tail efficiency ratio |
|---|---:|---:|
| A | 2.027 | 1.923 |
| N | 1.393 | 1.181 |
| potential | 1.716 | 1.178 |
| norm2 | 0.772 | 1.151 |

*Archived figure: Three trajectory schedules (`pm_tsc_posterior_validation/intermediate_trajectories/overview.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Comparison with maximum 8 (`pm_tsc_posterior_validation/intermediate_trajectories/comparison/efficiency.png`). See [archive guide](../../../ARCHIVE.md).*

## Controls and accounting

Expected trajectory work per chain is identical for all three schedules:

```text
maximum 4: (64+320)*3   = 1152
maximum 6: (64+192)*4.5 = 1152
maximum 8: (64+128)*6   = 1152
```

Each campaign independently builds its metric with 252 forward/adjoint calls,
charged once. The intermediate matrix hash matches both preceding matrices
exactly. Actual randomized trajectory lengths are counted, with comparison
requiring total PM budgets within 5%. Metric setup and warmup are included;
predictive-only forwards are excluded equally. Predictions use stride 12,
saving 16 per chain, matching the other schedules without diagnostic thinning.
Different outer-iteration counts also change the number of cheap scalar sweeps
and the fraction of total work spent in warmup. This compares complete schedules.

Native density/gradient/Jacobian checks and fresh metric checks run again.
Unchanged source hashes support reuse of the previously validated native binary;
its prior serial/slab/pencil MPI, sanitizer and Gaussian-moment checks were not
needlessly rerun for these input changes. Intervening unrelated ReACT changes
have no overlap with binary dependencies. Raw feature reconstruction, native
rank diagnostics and analysis replay verify the saved report after completion.

## Reproduce and inspect

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/run.py --bin-dir /path/to/binaries --output /fresh/MID --scheme preconditioned --draws 192 --leapfrog 6 --predictive-stride 12 --timeout-seconds 2400
python3 FML/FieldLevel/pm_tsc_posterior_validation/compare.py --baseline /LONG --candidate /MID --output /COMPARE --allow-trajectory-change
python3 FML/FieldLevel/pm_tsc_posterior_validation/intermediate_trajectories/overview.py --root FML/FieldLevel/pm_tsc_posterior_validation --output /OVERVIEW
```

[Protocol (archived)](../../../ARCHIVE.md),
[commands (archived)](../../../ARCHIVE.md),
[all-feature diagnostics (archived)](../../../ARCHIVE.md),
[summary and failure flags (archived)](../../../ARCHIVE.md),
[comparison to maximum 8 (archived)](../../../ARCHIVE.md),
[comparison to maximum 4 (archived)](../../../ARCHIVE.md),
[raw states (archived)](../../../ARCHIVE.md),
[control checks (archived)](../../../ARCHIVE.md),
[compiled source hashes (archived)](../../../ARCHIVE.md) and
[three-schedule summary (archived)](../../../ARCHIVE.md) are retained.
Other chain files sit beside chain0. See the [guide](README.md) and
[shorter-trajectory report](TRAJECTORY_RESULTS.md) for context.

Scope remains one N16 realization. N32 mixing, particle convergence, different
signal/noise regimes and repeated-mock coverage remain unvalidated. The dense
small active-space metric is not a scalable production full-grid preconditioner.
