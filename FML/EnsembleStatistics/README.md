# Control variates and streaming ensemble summaries

`ControlVariates.h` supplies C++17 streaming moments, independently trained scalar/vector control variates, phase provenance and MPI ensemble reductions. `MatchedSpectra.h` optionally connects Fourier density fields to the existing FML power/cross-spectrum binning. No simulation evolution is performed here: run the existing ZA/2LPT/COLA and target workflows with matching initial phases and feed their measured summary vectors to this API.

## Statistical contract

For target observable **y**, cheap surrogate **c**, independently fitted fixed matrix **B**, and surrogate population mean **mu**, each held-out row becomes

```
z = y - B (c - mu)
mean_estimate = average(z)
Cov(mean_estimate) = sampleCov(z) / N_eval + B Cov(mu_estimate) B^T
```

`sample_covariance` describes the corrected *realization* observable, conditional on the fitted model. `covariance_of_mean` includes the independent surrogate-mean uncertainty. That second term is shared by every evaluation row and does not decrease with `N_eval`. These are covariance estimates, not a guarantee of Gaussian confidence intervals. Coefficient uncertainty changes the conditional residual variance but introduces no extra mean-error term when training is independent and the surrogate mean is unbiased.

Training, surrogate-mean estimation, and evaluation must use disjoint independent phase blocks. Target/control phase keys within a pair must match exactly. Never use the training or evaluation control sample mean as an allegedly known population mean: this loses the desired correction and invalidates the error interpretation. `MeanEstimate::exact` is only for a genuinely known analytic mean; simulations generally need `MeanEstimate::independent` from a larger cheap ensemble. Bias in either a supplied mean or the target ensemble is not corrected by this algorithm.

One row represents one independent draw under the declared sampling convention. If using antithetic pairs, average the two measurements *before* adding a row, set `phase_pair_averaged=true` and key `phase_sign=0`. The registry rejects adding both signs of one phase block as independent rows. Fixed-amplitude results estimate the fixed-amplitude population, which need not equal the ordinary Gaussian-amplitude ensemble. MCMC-correlated rows, overlapping subvolumes and unblocked antithetic members do not satisfy the independent-row covariance formula.

## Minimal use

```cpp
#include <FML/EnsembleStatistics/ControlVariates.h>
using namespace FML::ENSEMBLE;
Metadata tags{"cosmology/config digest", "Pk: k centers, units, estimator, v1", "Nbody", "2LPT"};
PairedTraining training(2, 2, tags);
// Add each independent matched training row once, using actual phase keys:
training.add({10, 20}, {9, 18}, PhaseKey{1, 1234}, PhaseKey{1, 1234});
training.add({12, 23}, {11, 21}, PhaseKey{2, 5678}, PhaseKey{2, 5678});
FitOptions options; options.mode = FitMode::Diagonal;
// Toy known means ONLY. See independent-mean recipe below for measured means.
auto model = training.global().fit(MeanEstimate::exact({10, 20}, tags), options);
ControlEvaluation held_out(model);
held_out.add({11, 21}, {10, 19}, PhaseKey{3, 9012}, PhaseKey{3, 9012});
held_out.add({13, 24}, {12, 22}, PhaseKey{4, 3456}, PhaseKey{4, 3456});
auto estimate = held_out.global().estimate();
```

For an independently measured mean:

```cpp
RunningCovariance cheap_local(number_of_control_bins);
std::vector<PhaseKey> all_cheap_keys; // complete global provenance, identical on all ranks
// Fill cheap_local only with rows owned by this rank; construct all_cheap_keys.
auto cheap_global = cheap_local.global();
auto mu = MeanEstimate::independent(cheap_global, all_cheap_keys, tags);
auto model = training.global().fit(mu);
```

The caller supplies complete keys corresponding to the moments; moments cannot prove what simulations generated them. Choose globally unique realization IDs and unique `(seed,stream)` phase blocks. The key also records phase-generator backend (`Philox`, `NGenIC`, `External`), fixed-amplitude setting and phase sign. Setting a tag does not change the initial-condition generator. Encode external-generator version and other configuration in metadata. Tags cover cosmology, model versions and observable schema; include redshift, units, mesh, bin centers, tracer selections, shot-noise and assignment corrections, and normalization. Strings are caller-defined and compared exactly; the API cannot verify their scientific truth. `fingerprint()` is a deterministic consistency identifier, not a cryptographic digest or a serialized checkpoint.

## Fitting and streaming APIs

- `RunningCovariance(d)`: `add`, `merge`, `global`, `summary`, `reset`. Means and centered second moments use long-double Welford/Chan updates. Covariance uses `N-1`; empty means and covariance with fewer than two rows are NaN. A global reduction uses centered moment arrays and sample counts, never sample values.
- `PairedTraining(n_target,n_control,tags)`: streams joint moments, exposes the joint summary for raw/cross covariance, and fits a frozen `ControlModel`. `Full` fitting uses `B=C_yc C_cc^-1` with controls standardized before a Cholesky solve. The default ridge `1e-8` is dimensionless in those standardized units; tune it on independent validation data for high-dimensional or noisy training. `Diagonal` requires matching widths and fits each corresponding bin separately. Zero-variance controls are ignored. Singular unregularized fits fail explicitly.
- `ControlEvaluation(model)`: streams held-out corrected rows, enforces independence, returns the mean, residual sample covariance, covariance of the mean and metadata. At least two evaluation rows are required for covariance. `ControlModel::apply` provides individual corrected vectors with the same train/mean disjointness checks; use `ControlEvaluation` to check duplicate evaluation rows too.

A covariance matrix costs O(d^2), each moment update O(d^2), and a full fit O(n_control^3). Moment storage does not grow with sample count. The safety registry stores O(N) phase identities, not observable rows; these identities are replicated by `global()` to detect duplicate phase blocks. This first version is intended for modest ensembles, not billion-row provenance databases. It does not implement the specialized CARPool estimator of the *target covariance matrix*; the residual covariance is a different object.

## MPI and threads

All `global()` calls are collective over `FML::communicator`, return the same combined object on each rank and do not mutate the local object. Each independent ensemble row must be added on **one rank only**. Distributed FFT/power-spectrum calculations, by contrast, are collective over all grid ranks: after obtaining their common measured vector, add the row only on its designated ensemble owner (often rank zero). Calling `global()` again on an already replicated global object duplicates samples and is an error for provenance-tracked classes. Keep local accumulators and reduce snapshots when needed. Collective validation rejects incompatible metadata/models and duplicate phase blocks across ranks.

Local `add`/`fit` errors throw locally; coordinate caller-side input failures before the next collective. Classes are not internally synchronized. For moments, use one `RunningCovariance` per thread and `merge` after the parallel region; invoke MPI outside parallel regions.

## Matched Fourier spectra

```cpp
#include <FML/EnsembleStatistics/MatchedSpectra.h>
auto s = FML::ENSEMBLE::matched_spectra(target_fourier, control_fourier,
                                      boxsize, kmin, kmax, number_of_bins);
// s.target, s.control, s.cross, s.correlation, s.k, s.modes
```

Inputs are compatible Fourier-space `FFTWGrid<3>` density contrasts. `kmin`/`kmax` are the existing FML **bin-center** limits in physical inverse-length units, not bin edges; at least two bins are required. Output auto/cross powers have volume units; the helper applies `k -> k/L`, `P -> L^3 P`. It uses existing mode-count and Hermitian-weight conventions. It applies no assignment, shot-noise, survey-window, selection or tracer corrections. Empty/zero-power coherence is NaN. Cross coherence is a useful diagnostic, not a guarantee of held-out variance reduction. Keep preprocessing and phase/amplitude conventions consistent before fitting.

## Build and validation

Only `Global.cpp` is linked for the core; no FFTW/GSL dependency. From `example/` or `tests/`:

```
make USE_MPI=false USE_OMP=false
make run USE_MPI=false USE_OMP=false   # tests directory only
make run MPIRUN=mpirun NP_LIST="1 2 4" # MPI build, start from clean objects
make USE_FFTW=true test_spectra       # optional FFTW integration test
mpirun -np 4 ./test_spectra
```

Clean between configurations. `example/Main.cpp` runs a reproducible correlated Gaussian toy and prints raw/controlled means and error, coefficient, model identifier and variance reduction. It is not an N-body validation or a performance forecast.

Tests independently check hand-computed covariance, large-offset/unequal-count reductions, empty ranks, exact affine coefficients, diagonal/ridge/singular/constant controls, analytic propagation of cheap-mean uncertainty (including its nonshrinking floor), and held-out Gaussian variance reduction. Tests reject training/mean/evaluation overlap, duplicated phase blocks, backend mismatch, unblocked antithetic pairs and inconsistent MPI schemas. Optional FFTW tests use analytically proportional fields and verify auto/cross ratios, coherence, physical box scaling and the morphology field adapter. Production follow-up: calibrate on matched ZA/2LPT/COLA versus N-body ensembles, measure coverage and conditioning versus binning/redshift, and compare total simulation cost at fixed mean error.

References: [CARPool mean estimation](https://arxiv.org/abs/2009.08970); [separate covariance-estimation extension](https://arxiv.org/abs/2106.11718). Implementation here is independent and deliberately limited to frozen-coefficient mean estimation and streaming sample moments.
