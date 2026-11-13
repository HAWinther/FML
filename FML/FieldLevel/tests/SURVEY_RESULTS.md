# Bounded nonlinear inference and Cartesian survey validation

Recorded 13 November 2026. These are engineering/small same-model inference
results, not production cosmological or survey calibration. Stage 5a's contracts
and examples are in [../SURVEY.md](../SURVEY.md).

## Nonlinear pass before the survey extension

`test_framework_nonlinear_mpi.cpp` tests the existing periodic distributed
Zel'dovich and EdS 2LPT models. Each has two independent prior-drawn initial
fields and two dispersed HMC chains per mock (eight chains per execution).
All scalar/nuisance values are fixed: b1=1.4, b2=0.3, b_s2=-0.2,
b_laplacian=5 (Mpc/h)^2, f=0.6 and noise power 5 (Mpc/h)^3. The spectrum is
constant P=2000 (Mpc/h)^3. Meshes are 8^3 initial / 16^3 deposition, L=100 Mpc/h,
with initial/bias/likelihood cutoffs 0.115/0.10/0.11 h/Mpc. This stays within the
backend's conservative half-Nyquist initial cutoff.

Each chain uses 150 warmup + 300 retained trajectories, eight leapfrog steps,
target acceptance 0.85 and global-cell-keyed field momenta. Initial fields have
independent phases and amplitudes scaled by 0.5 or 2. Recorded summaries are
real/imaginary fundamental white-field projections and total white-field power.
Assertions require Rhat < 1.05, bulk ESS > 60, tail ESS > 40, acceptance > 0.7 and
zero sampling divergences. Field information is checked through a fundamental
mode variance below 0.25 (its prior variance is one), and posterior-mean initial
physical density RMSE below half the prior-zero RMSE. The initial density map is
linear at fixed spectrum, so transforming the mean white field gives its sample
mean. No analogous assertion about the nonlinear evolved-density mean is made.

| Profile | Checks | Maximum Rhat | Minimum bulk / tail ESS | Initial-density RMSE / prior-zero RMSE |
| --- | --- | --- | --- | --- |
| Serial and one-rank MPI | 32 each | 1.0131 | 227 / 249 | 0.0336–0.0523 |
| Two-rank MPI | 64 | 1.0133 | 226 / 249 | 0.0336–0.0523 |
| Four-rank MPI | 128 | 1.0128 | 230 / 249 | 0.0335–0.0523 |

All eight chains per profile pass and have zero sampling divergences. Posterior
fundamental-mode variances are 0.00128–0.00135. Serial and one-rank results agree;
multi-rank reductions give small trajectory/statistic differences, not a claim
of bitwise rank-independent chains. The largest reported mode-mean change across
rank counts is below 4e-5, much smaller than its Monte Carlo uncertainty.

The forward/tracer prediction is also required to differ from the simple
linear-RSD baseline. That comparison includes the higher-derivative bias term;
it is not a separate measurement isolating second-order dynamics. Independent
plane-wave/adjoint checks of those dynamics remain in the pre-existing LPT suite.
This small high-information, fixed-nuisance experiment establishes useful
same-model inference beyond a short sampler smoke test. Two mocks per model do
not establish posterior coverage, independent-simulation accuracy or realistic
survey performance. Performance/strong-scaling calibration is not claimed.

The nonlinear suite is opt-in (`framework-nonlinear-mpi`) rather than part of
ordinary short module runs. It was completed in serial and 1/2/4 MPI ranks before
final survey validation; serial/one-rank results anchored implementation while
the additional rank profiles ran in the background.

## Survey observation and likelihood verification

`test_survey_likelihood_mpi.cpp` includes:

- analytic real Fourier waves for synthesis and an independent direct DFT for
  its adjoint, including N^3 normalization and full-Hermitian inner products;
- independently summed masked heteroscedastic Gaussian and positive-link
  Poisson likelihoods, count factorials, exposure/selection and Gaussian
  log-noise normalization;
- finite-difference Gaussian log-noise and full ZA/2LPT posterior field/scalar
  derivatives with nonlinear bias, higher-derivative bias and RSD;
- MAP decrease and short fixed-parameter HMC integration for both likelihoods
  and both nonlinear evolution models;
- masked NaNs, an entirely empty footprint, a rank with no observed cells,
  zero-rate/zero-count behavior and positive-count rejection at zero exposure;
- one-rank bad dimensions, variance, completeness, Hermitian prediction,
  geometry callback, origin or offset, failing collectively before mismatched
  FFT/MPI calls;
- stable Poisson log-rate evaluation under exponential underflow and collective
  proposal rejection under overflow.

The final suite passes 53 checks per rank:

| Profile | Passed checks |
| --- | --- |
| Apple Clang serial | 53 |
| GCC 14 serial | 53 |
| Clang ASan + UBSan | 53 |
| MPI 1 / 2 / 4 ranks | 53 / 106 / 212 |
| MPI two ranks, two OpenMP threads, NO_AUTO_FML_SETUP | 106 |

The isolated Makefile `survey-mpi` target also passes all 53 serial checks; the
long nonlinear target and normal `run` plans were inspected. Both Gaussian and
Poisson 2LPT masked-survey examples pass with two MPI ranks, and the isolated
Makefile example builds/runs. Existing serial framework (53 checks) and two-rank
LPT (80 checks) regressions pass. These checks use FFTW double precision; float
precision has not been validated for this extension.

Likelihood values/derivatives in the independent-wave and nonlinear fixtures
agree across rank counts to numerical precision. The survey HMC checks use
short, small-step chains to exercise integration only; the longer nonlinear
mixing experiment above uses the periodic likelihood. There is no claim of
converged survey posterior sampling from those integration smokes/examples.

## Evidence and remaining scope

Codex workspace artifacts are under `outputs/fml-framework-survey/`:
`nonlinear-*.log` and `nonlinear-summary.json` are the bounded periodic evidence;
`verified/` contains final survey compiler/runtime/Makefile command records and
logs. `validate.py serial` and `validate.py mpi` reproduce those final profiles.
Earlier survey development logs under the root and `final/` are superseded;
final validation includes the finite-origin/offset and Hermitian failure tests.

Realistic survey mock recovery, selection/pixelization convergence, wide-angle
or lightcone evolution, correlated voxel noise, survey mean-density nuisances,
parameterized exposure/selection, additional likelihood parameters, multiple
correlated datasets and survey checkpoint identities remain explicit extensions.
Poisson data do not constrain the existing Gaussian-noise scalar; its posterior
remains its independent prior when all scalar parameters are sampled. This is
stated in the API guide and exercised by the full-posterior derivative checks.
No production forward/sampler formula or prior normalization was changed.
