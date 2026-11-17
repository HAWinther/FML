> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Bounded full-field PM posterior results

Four fixed-parameter chains and four joint field/spectrum/noise chains each use
256 warmup and 1024 retained draws. Every chain samples all 64 white coordinates;
the joint target adds two scalar parameters. [The guide](README.md) specifies the
mock, initial/force meshes, priors, starts, seeds and exact diagnostics.

## Mixing and reference checks

| Ensemble | Maximum Rhat | Minimum bulk ESS | Minimum tail ESS | Mean acceptance range | Post-warmup divergences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed parameters | 1.01282 | 622.361 | 718.337 | 0.85927–0.911364 | 0 |
| Joint spectrum/noise | 1.00782 | 1214.42 | 1338.23 | 0.893147–0.908256 | 0 |

These extrema cover all 195 fixed-target and 197 joint-target features, including
all white cells, active projections and their squares, independent null projections
and their squares, norms, potential and varying parameters. All declared Rhat <1.05,
bulk ESS >400 and tail ESS >200 gates passed. This is finite-chain evidence, not a
proof of convergence. Diagnostics use every retained draw, including 481
fixed and 388 joint consecutive repeats after rejection.

The initial cutoff leaves six active Fourier directions and 58 exact independent
N(0,1) null directions. All analytic first/second-moment checks passed. Maximum
null-moment discrepancies are 2.32833 and
2.95234 Monte Carlo standard errors; mean
null squared norms are 58.0293 and 58.5268
(expected 58). ESS is capped at the actual sample count for these comparisons.
The direct-DFT initial-reconstruction errors are 1.00074e-16
and 8.77708e-17. This checks support and whitening normalization;
it does not independently reference the six-dimensional nonlinear active posterior.

## Reconstruction and posterior predictions

Posterior-mean/truth correlations are 0.993825 (fixed)
and 0.994406 (joint) for the initial linear density;
0.974156 and 0.975978 for
final observed-band density. This is one same-model noisy mock, not coverage or
a production accuracy test. Initial/final MSEs and chain metadata are in [summary.json (archived)](../../../ARCHIVE.md).

Posterior predictions use every eighth retained draw (512 per ensemble), with
fresh independently seeded Gaussian measurement noise. The observed band has
56 real degrees of freedom. Mean replicated chi-square discrepancies are
56.1358 and 56.1358; observed-data
means are 52.2551 and 55.2104.
Replicate-above-observed tail fractions are 0.638672
and 0.523438. They are descriptive posterior-predictive
checks, not frequentist p-values or coverage estimates. The normalization check
uses the known chi-square-56 replicated-noise mean/variance.

Joint scalar marginals (injected values 0 and log(0.001) = -6.907755):

- log_amplitude: mean 0.160128, central 95% [-0.520165, 0.834956].
- log_noise: mean -6.95943, central 95% [-7.33094, -6.56637].

*Archived figure: Joint initial/final density reconstruction (`pm_full_validation/figures/density.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Latent and noisy posterior-predictive power (`pm_full_validation/figures/power.png`). See [archive guide](../../../ARCHIVE.md).*

The initial spectrum has one supported shell: its intervals are error bars.
Final latent power and noisy prediction use distinct pointwise 95% intervals.
Density slices display latent standard deviations; the raw analysis exports
pointwise 95% density intervals separately. Final density comparisons/residuals
use only the observed Fourier band. Initial density is normalized at a=1.

*Archived figure: Joint scalar marginals (`pm_full_validation/figures/parameters.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Joint retained traces (`pm_full_validation/figures/traces.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Posterior-predictive discrepancies (`pm_full_validation/figures/predictive.png`). See [archive guide](../../../ARCHIVE.md).*

## Backend, runtime and artifact evidence

Short fixed and joint replays compare every retained canonical white/scalar state,
potential and saved density prediction across serial, MPI slabs (1/2/4 ranks),
2x2 four-rank pencils, one/two OpenMP threads and serial ASan/UBSan. Maximum
absolute difference is 1.98952e-13; the sanitizer
replay matches serial exactly. Replay is distinct from the longer statistical
ensembles, which use serial builds and four concurrent independent processes.

Clean builds use Clang 17, matching Open MPI/FFTW-MPI or serial FFTW, bounds checks
and no GSL; sanitizer builds use Clang 20 with ASan/UBSan and leak detection disabled.
Explicit Open MPI launcher arguments and all native commands are recorded. The
public `run.py` completed end-to-end, including native sampling, feature generation,
C++ diagnostics, serial replay and analysis. A separate local launcher exercised
the additional MPI/pencil/sanitizer variants. Four invalid-input checks also pass.
No production PM, posterior, shared execution or HMC implementation changed.
Full library CI was not rerun for this explicit validation-only campaign.

All 748 final analysis gates passed. Local raw states, predictions, interval
CSV, diagnostics, PNG/PDF plots, logs, build-source copies and complete review diff
are preserved under `outputs/fml-fieldlevel-pm-full/`. Pilot and fixed-only preview
analysis are separate from the final `campaign/science/` and `campaign/analysis/`.

## Remaining limits

This supplies bounded full-field/joint mixing and posterior-predictive evidence for
one N=4/M=8 mock with six active IC directions. Larger fields, more active modes,
multiple-mock coverage, independent active-posterior references and mesh/particle/time
convergence remain open. PM restart identities, tracer/RSD/survey likelihoods,
cosmological derivatives and production performance remain extensions. Item 52
remains open. No scaling or total-memory improvement is claimed.
