# Matched-ensemble validation results

Validated 12 November 2026. This adds a phase-tracked `ControlMean` helper and an
end-to-end matched ZA/2LPT versus small EdS PM integration experiment. The reusable
statistical API continues to accept arbitrary target/control summary vectors.
The PM target in this fixture is deliberately coarse and **not mesh-converged**.
No production cosmological calibration or equal-cost speedup is claimed.

## Independent fits and finite cheap-mean uncertainty

Each main case uses 64 independent matched training phases, 1024 independent cheap
phases, and 256 held-out paired evaluation phases. Three raw CIC power bins have
centers `(2pi,4pi,6pi)/200`. Both diagonal and full fits are frozen before evaluation.
The two cases use a=0.5 and a=1 with common phase IDs; they are sensitivity examples,
not independent cosmological validation suites.

NumPy independently reconstructs coefficients, means, residual covariances, raw
moments and `Cov(mean)`, including the shared cheap-mean uncertainty term. Maximum
scaled differences over every entry are:

| Case | Maximum scaled difference | Standardized control condition range |
|---|---:|---:|
| fiducial | 1.665e-15 | 1.345–1.403 |
| late | 9.725e-16 | 1.316–1.471 |

Scaling uses `max(1, max absolute reference entry)` for each reported quantity.
Conditioning is measured for the standardized training control covariance, before
adding the default ridge 1e-8. These low-dimensional matrices are well conditioned;
this does not establish safe conditioning for a large number of bins.

The following are held-out **variance** ratios, not standard-error or speed ratios.
Each triplet corresponds to increasing k:

| Case / control / fit | Raw / residual realization variance | Raw / corrected mean variance |
|---|---|---|
| fiducial / ZA-diagonal | 206.398, 24.681, 7.066 | 4.057, 3.555, 2.783 |
| fiducial / ZA-full | 201.108, 24.508, 7.541 | 4.046, 3.415, 2.847 |
| fiducial / 2LPT-diagonal | 143.008, 23.360, 7.652 | 4.055, 3.478, 2.896 |
| fiducial / 2LPT-full | 139.519, 24.033, 7.972 | 4.043, 3.384, 2.936 |
| late / ZA-diagonal | 129.127, 12.695, 3.668 | 4.062, 3.278, 2.211 |
| late / ZA-full | 127.488, 13.158, 4.794 | 4.051, 3.108, 2.511 |
| late / 2LPT-diagonal | 85.061, 12.402, 4.168 | 4.057, 3.150, 2.376 |
| late / 2LPT-full | 83.485, 13.536, 5.220 | 4.048, 3.091, 2.619 |

The independent cheap mean limits the gain strongly: it supplies roughly 40–98%
of the reported mean variance in these cases. Residual variance reductions as large
as ~200 do not imply a ~200-fold reduction in mean variance or simulation cost.
More independent cheap simulations reduce that floor; more evaluation rows alone
do not. Full fitting is not uniformly better than diagonal fitting.

The small 12-training/32-cheap/16-evaluation checks can have mean variance ratios
below one (about 0.55 in one full-fit bin), despite reduced residual variance.
Those checks are retained in the summaries. There is no enforced success-only
selection or assertion that all fitted controls help.

## Repeated known-population coverage

`test_coverage` uses independent Gaussian experiments with the analytic target mean
5, trained coefficients and a finite independently measured control mean. Each
experiment has 32 training, 256 cheap and 64 evaluation rows. The 2048 complete
experiments are distributed across MPI ranks without sharing phase blocks.

| Quantity | Measured value |
|---|---:|
| Mean bias | -0.0015727259 |
| Empirical variance of estimates | 0.017320034 |
| Average reported variance | 0.017169166 |
| Average variance omitting cheap-mean term | 0.005814532 |
| Coverage of nominal 95% Gaussian interval | 0.93994141 |
| Coverage omitting cheap-mean term | 0.74462891 |

The reported/empirical variance ratio is about 0.9913, and the mean bias is within
the asserted four-standard-error Monte Carlo bound. Nominal 95% normal intervals
cover about 94%; finite-sample normal intervals are approximate, and this is not a
general confidence-interval guarantee. Omitting the shared cheap-mean term yields
about 74% coverage, exposing an important failure mode. The coarse PM experiment
has no analytic target population mean; coverage is not asserted for its bandpowers.

## Numerical sensitivity, MPI and cost

For the same first 12 phases, the maximum target-bandpower change from 20 to 40
steps is 0.00418711, and from 40 to 80 is 0.0010914, scaled
to the largest fine-reference bandpower. Halving the step improves the agreement.
The combined IC/particle/force/measurement mesh change from 16 to 32 changes target
bandpowers by 0.378131 on the corresponding scale. This large change means the
coarse PM target is not a converged physical reference. Statistical consistency
checks refer to the declared discrete target ensemble, not continuum N-body truth.

The 2-rank field workflow differs from serial by at most 8.882e-16 in scaled
raw auto/cross/coherence outputs; its fitted results separately pass the NumPy check.

The 4-rank field workflow differs from serial by at most 1.221e-15 in scaled
raw auto/cross/coherence outputs; its fitted results separately pass the NumPy check.


Measured serial stage costs for the main cases are shown below. These are small,
machine/load-dependent timings with existing particle/force diagnostics enabled,
not a performance forecast. `total` includes measurements and streamed row work;
per-stage and per-row data are retained in the CSVs.

| Case / role | Rows | Median total seconds / row | Sum of row seconds |
|---|---:|---:|---:|
| fiducial / train | 64 | 0.0149633 | 0.997547 |
| fiducial / mean | 1024 | 0.00216465 | 2.62692 |
| fiducial / eval | 256 | 0.0196839 | 5.14598 |
| late / train | 64 | 0.019148 | 1.20132 |
| late / mean | 1024 | 0.00225765 | 2.84646 |
| late / eval | 256 | 0.0169463 | 4.36565 |

Training and evaluation rows compute both controls and PM; cheap rows compute both
controls and skip PM. Both controls share the same 2LPT setup, so these timings do
not measure an optimized standalone ZA cost. Model/mean reductions and process
startup are not included in summed row time. Actual process times are retained in
`commands.json`. A production fixed-budget comparison must include all training,
cheap-mean and evaluation costs and use covariance of the mean. This experiment
reports those counts and costs without inferring an equal-cost production speedup.

## Build and failure coverage

Core control/streaming tests pass 28 checks in serial and 33 at multiple MPI ranks;
the repeated coverage test passes 4 checks, and the FFT spectrum integration test
passes. Fresh C++17 builds cover serial double, single-precision FFTW, OpenMP
(1 and 4 threads), ASan+UBSan, and Open MPI/Clang17 (1,2,4 ranks). Both the public
helper and matched driver compile in every configuration. Captured builds have no
compiler warnings. The standalone matched-driver Makefile builds in a clean copied
directory. Core tests remain independent of FFTW/GSL; only the field driver needs
FFTW and the transitive GSL headers from `NBody.h`.

The full serial NumPy reference suite and 2/4-rank comparisons pass. Single-precision,
OpenMP and sanitizer matched-field smoke runs also pass every independent NumPy
comparison. External FFTW libraries are not sanitizer-instrumented. One initial
MPI launch was blocked by sandbox socket permissions; after restoring the local
network grant, the complete runtime checks passed.

New helper tests cover invalid rows without identity reservation, insufficient
samples, duplicate phase blocks, training/mean/evaluation overlap, covariance
propagation, empty owner ranks, and rank-dependent metadata/dimensions.
`ControlMean` retains the same O(d^2) moments and O(N) phase registry contract; it
reduces caller bookkeeping rather than changing ensemble storage complexity.

Remaining work includes converged cosmological/COLA versus N-body ensembles,
production-bin conditioning and regularization choices, calibrated non-Gaussian
coverage, fixed-total-cost studies and the specialized target-covariance CARPool
extension. Mean estimation and residual covariance remain distinct from estimating
the target covariance matrix with control variates.
