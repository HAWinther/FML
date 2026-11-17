# BAO prototype gradient and inference validation

Added 13 November 2026 for FL-4 and the bounded scientific follow-ups in FL-5.
These tests concern the original serial `FieldLevelBAO` prototype. The separate
general MPI framework is documented in `../FRAMEWORK.md`.

## Portable local derivatives and CIC crossings

The nine existing linear/1LPT/2LPT, centering, bias and RSD gradient fixtures in
`test_field_level.cpp` now sweep ten central-difference steps, each four times
smaller than the previous step. A passing derivative needs two adjacent steps
with error below 2e-6, at least one below 1e-6, and improvement from the coarse
step (or an already accurate coarse result). The scale is the maximum of one
and both derivative magnitudes. This resolves an interval before roundoff
rather than selecting whichever single step happens to pass. A biased-derivative
negative control verifies that the criterion can fail.

`test_bao_gradients.cpp` separately checks deterministic, small-amplitude CIC
fixtures away from assignment knots. It covers white and partially centered
1LPT and biased/RSD 2LPT. Its manufactured plane-wave fixture puts particles
exactly at a CIC knot: the potential must be continuous, and its two one-sided
slopes must converge to different limits. A unique smooth derivative is not
asserted at that point. This distinction is required for piecewise CIC.

```sh
make -C FML/FieldLevel/tests USE_MPI=false bao-gradients
```

The normal `run` target includes the new CIC suite and the short deterministic
validation below. FFTW double precision is the tested contract for these tight
derivative checks; float precision has not been validated here.

## Independent particle Fourier reference and model mismatch

The short invocation of `test_bao_validation` compares fixed physical low-band
initial modes on 8^3 and 16^3 particle grids with a 32^3-particle direct Fourier
sum. The reference bypasses deposition, its Fourier window and the density FFT;
it reuses the 1LPT displacement routine. It is an independent deposition/reference
quadrature, not an independent gravitational simulation. Tests also vary smooth
assignment widths 4/6/8 and the initial/likelihood cutoff while preserving the
physical input modes.

A separate higher-cutoff reference adds modes 4 and 5 along one axis. Their
nonlinear beat changes observed low-band modes. This deliberately omitted-mode
fixture must produce a mismatch above numerical deposition error. It tests a
failure mode of low-cutoff predictions; it does not establish posterior robustness
under general model mismatch.

The short suite also verifies that adding field-summary recording leaves the
public BAO sampler's trajectory, potentials, metric and beta rows exactly unchanged
for 1LPT and biased/RSD 2LPT.

```sh
make -C FML/FieldLevel/tests USE_MPI=false test_bao_validation
FML/FieldLevel/tests/test_bao_validation
```

## Multi-chain inference, information and mock ensembles

The longer, optional `bao-validation` target runs two dispersed chains per mock,
starting beta at 0.88 and 1.12 with independent initial fields. Defaults are
250 warmup, 600 sampling trajectories and ten leapfrog steps. Both warmup and
sampling use the shared HMC implementation. It monitors rank-normalized Rhat,
bulk ESS and tail ESS for beta, real/imaginary low-mode physical-white field
projections and mean white-field power. Each requires Rhat < 1.05, bulk ESS > 80
and tail ESS > 50. Sampling divergences must be zero and acceptance > 0.6.

The inexpensive toy has 12^3 initial cells, 24^3 density cells, L=600 Mpc/h,
kmax=0.06 h/Mpc, kernel width 4, noise power 300 (Mpc/h)^3 and P_smooth(0.1)=1500
(Mpc/h)^3. The wiggle amplitude is deliberately strengthened from 0.08 to 0.5
so information is detectable with a small grid. The beta prior is uniform on
[0.85,1.15]. For 2LPT, b2=0.3, bG2=-0.2 and f=0.7 are fixed; b=1.5 in both models.
These are same-model sampler tests, not a physical-spectrum precision forecast.
The prototype's quadratic sources are not padded: this ensemble also does not
establish high-cutoff/dealiasing convergence of 2LPT.

The posterior beta width must be below 75% of the prior width. A no-wiggle
negative control has exactly flat beta likelihood in white coordinates. Its
prior-only width fails this information check, although it passes the historical
weak rule that beta_true=0.97 is within 3.5 posterior standard deviations of one.
Thus removing all BAO information cannot pass the new inference criteria merely
by retaining a broad posterior. The original one-chain recovery test remains a
smoke test; it is not scientific calibration by itself.

Mock truths are drawn independently from the prior. The executable writes raw
beta and field summaries, along with truth, mock, chain and draw indices. The
NumPy script independently measures empirical equal-tail 68%/95% coverage,
truth quantiles, widths and z scores. Its 95% Wilson intervals show the large
binomial uncertainty of small ensembles; they do not include MCMC error. Truth
compatibility checks in C++ are broad smoke checks, while coverage estimates
are reported rather than tuned to pass one small ensemble.

```sh
make -C FML/FieldLevel/tests USE_MPI=false bao-validation \
  BAO_MODEL=1lpt BAO_MOCKS=8 BAO_SAMPLES=600 BAO_OUTPUT=1lpt.tsv
make -C FML/FieldLevel/tests USE_MPI=false bao-validation \
  BAO_MODEL=2lpt BAO_MOCKS=4 BAO_SAMPLES=600 BAO_OUTPUT=2lpt.tsv
python3 FML/FieldLevel/tests/analyze_bao_validation.py 1lpt.tsv 1lpt.json
```

The legacy adapter retains its public API and beta-only default recorder. The
validation adapter uses the same potential, mass, bounds and integration and
records a few extra summaries; a direct trajectory parity test guards this.
No production forward-model or sampler formulas were changed.

## Remaining scientific scope

Broader ensembles at physical wiggle amplitude, converged cosmological mocks,
posterior predictive checks against independent simulations, field-summary/tail
coverage, nuisance-parameter inference and model-mismatch posterior calibration
remain separate work. This suite does not remove those limits or make all of #52
complete. Workspace profiling (FL-6) is also separate.

Measured local results and commands are recorded in the accompanying review
artifacts under `outputs/fml-fieldlevel-validation/` in the Codex workspace.

## Recorded local results (13 November 2026)

Apple Clang and GCC 14 pass all 37 legacy checks, including the nine field/beta
step sweeps, and all 17 dedicated CIC/sweep checks. Clang and GCC also pass the
final short 10-check convergence/recorder-parity suite. The isolated Makefile
build/gradient target passes. ASan/UBSan checks are recorded in the review logs.
These tests use serial FFTW double precision; the prototype remains single-rank.

The 8^3 and 16^3 smooth-width-8 predictions differ from the 32^3 direct particle
reference by 5.71e-7 and 1.34e-7, respectively. Widths 4/6/8 at 16^3 give
1.77e-3 / 2.41e-5 / 1.34e-7. Adding initially omitted high-k waves changes the
low-band prediction by 1.81e-2, well above the numerical errors. This is a
manufactured 1LPT convergence/mismatch experiment with three low input modes,
not a cosmological mesh-convergence result.

| Toy model | Prior-drawn mocks / chains | Maximum Rhat (all summaries) | Minimum bulk / tail ESS | Mean posterior width / prior width | Empirical 68% / 95% coverage |
| --- | --- | --- | --- | --- | --- |
| 1LPT | 8 / 16 | 1.0203 | 227 / 320 | 0.321 | 5/8 / 7/8 |
| Biased/RSD 2LPT | 4 / 8 | 1.0103 | 338 / 384 | 0.298 | 2/4 / 4/4 |

All 24 chains have zero sampling divergences and pass the acceptance and
information checks. For 95% credible-interval coverage, the 95% Wilson intervals
are [0.529, 0.978] for 1LPT and [0.510, 1.000] for 2LPT. For 68% coverage they are
[0.306, 0.863] and [0.150, 0.850]. These broad intervals are consistent with the
nominal probabilities but constrain calibration weakly. They are not evidence
of validated production coverage.
