> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Bounded PM HMC results

The [campaign](README.md) passes all 125 recorded analysis gates. The fixed and
adapted two-mode conditional ensembles agree with independent finite-grid posterior
integration; the separate short full-field/joint runs pass integration/replay checks.
All results concern one same-model periodic mock with fixed background.

## Conditional posterior reference and moments

The fine-grid reference gives

```text
mean = (1.3595425901, -1.0468753709)
covariance = [[ 0.0516400360, -0.0353115170],
              [-0.0353115170,  0.0511813540]]
```

The largest change among the five raw moments on refinement from 41^2 to 81^2 is
6.87e-4. Expanding the box from [-6,6]^2 to [-8.1,8.1]^2 at the same fine spacing
changes those moments by at most 8.89e-16. The fine boundary-grid mass is about
1.9e-68. These are finite-grid/range checks, not a rigorous infinite-domain tail bound.

| Ensemble | Retained draws | Mean z0 | Mean z1 | Mean acceptance probability | Max coordinate Rhat | Min coordinate bulk ESS | Post-warmup divergences |
|---|---:|---:|---:|---:|---:|---:|---:|
| Four fixed-step chains | 1024 | 1.360386 | -1.049675 | 0.967836 | 1.005607 | 1729.19 | 0 |
| Two adapted chains | 512 | 1.359919 | -1.046417 | 0.926477 | 1.016926 | 1059.72 | 0 |

The adapted final epsilons are 0.134124 and 0.128891. Both first moments, both
second moments and the cross moment pass the declared Monte Carlo/reference
agreement envelope. The largest mean-coordinate discrepancy is 0.00280; the
corresponding conservative Monte Carlo error estimate is about 0.00707. Raw-moment
comparisons and all diagnostics are retained in `summary.json` and the review
bundle's `moment_comparison.csv`. ESS values above draw count reflect negative
autocorrelation; the uncertainty calculation caps those values at actual count.
The fixed/adapted traces contain 33/37 consecutive repeated draws respectively;
rejected proposals remain represented in the estimates.

The plotted reconstruction is the conditional initial density from these two
modes, with a pointwise quadrature 95% interval. Truth falling within this one
mock's interval is not a coverage measurement. All other white directions and
spectrum/noise parameters are fixed in this comparison.

## Energy and reversibility

Fixed-length 0.8 leapfrog probes include positions outside the high-probability
region. Maximum absolute energy increments across four probes are:

| Epsilon | Max absolute energy increment |
|---|---:|
| 0.4 | 414.077 |
| 0.2 | 14.7821 |
| 0.1 | 3.53181 |
| 0.05 | 1.01137 |
| 0.025 | 0.442011 |

Maximum reversal error is below 5.56e-15. Smaller steps reduce energy error, while
CIC cell-face crossings prevent a general second-order energy-error claim. The
large coarse-step errors show why the step size needs checking. The actual chain
acceptance/divergence results above apply to their sampled states and selected
steps, not all these deterministic probe states. Divergence threshold 10 means
positive H_new-H_old > 10, as in the library sampler; warmup divergences are not
reported by its result structure.

## Full-field, joint and parallel execution

Full-field runs call the public `FRAMEWORK::sample` on 64 white cells; joint runs
add two replicated spectrum/noise parameters. They use 16 warmup and 16 retained
draws. Conditional replay uses 16 warmup and 24 retained draws. These short cases
verify dimensions, finite records, parameter support, no post-warmup divergences
and retention of every draw. They do not establish full-field/joint mixing.

Serial, MPI slabs at 1/2/4 ranks, 4-rank 2x2 pencils, 1/2 OpenMP threads and serial
Clang 20 ASAN/UBSAN all pass. Maximum absolute trace/final-state replay difference
is 8.82e-13. The joint canonical last state has 64 global field entries plus two
scalar entries, checking the local-field/replicated-parameter split. No sampler,
posterior, PM forward/adjoint or shared execution implementation change was needed.

The lightweight eight-draw smoke helper passes serial and 1/2/4-rank MPI. Its actual
registered `pm-hmc-smoke` Make recipe was also exercised in serial. Actual clean
Makefile builds used Clang 17 normally, matching Open MPI/FFTW-MPI, separate serial
FFTW, bounds checks and no GSL. FFTW threading is disabled; MPI controlled runs use
1/2 OpenMP threads and passive waiting. Leak detection is disabled for ASAN/UBSAN.
Full library CI was not rerun.

*Archived figure: Bounded PM HMC validation (`pm_hmc_validation/figures/hmc_validation.png`). See [archive guide](../../../ARCHIVE.md).*

## Artifacts and limits

Native command logs, frozen build sources and hashes, 2D quadrature grids, chain
traces, metadata, diagnostics, canonical replay states, energy sweeps, PDF/PNG
figures, and complete review diff are preserved in the local
`outputs/fml-fieldlevel-pm-hmc/` bundle. Pilot runs are kept separately from final
`science/` and `replay/` evidence. The final campaign uses the public CLI protocol;
the local build orchestrator additionally configures this machine's library paths.

This validates a bounded conditional target and full-sampler plumbing. Larger
fields, longer joint nuisance runs, all-direction calibration, posterior predictive
behavior, repeated-mock coverage and production mesh/particle/time convergence
remain open. The piecewise CIC map, identity metric and simple MAP optimizer retain
the limitations documented by the earlier PM milestones. No total-memory, scaling,
optimal-checkpoint, production-survey or cosmological-parameter claim is made.
