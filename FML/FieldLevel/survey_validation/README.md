# Bounded survey inference validation

This standalone experiment exercises the production Cartesian survey likelihoods,
2LPT forward/pullback and shared HMC sampler. It measures conditional initial-field
recovery, mixing, repeated-mock coverage, posterior predictions and separate
selection/pixelization convergence. See [RESULTS.md](RESULTS.md) for the measured
results and [../SURVEY.md](../SURVEY.md) for the observation contracts.

## Reproduce

Build both executables with `make` in this directory, supplying the usual FML
compiler/FFTW settings. `survey_validate` uses the complete white grid and the
production same-rank checkpointed sampler. `survey_reference` is a small exact
active-mode reference. Keep raw outputs outside the source directory.

```sh
./survey_reference --check-reference
python3 run.py --bin-dir . --output /absolute/path/to/survey-validation
```

The runner executes 12 mocks per observation model with the active reference,
three full-grid production mocks per model, exact masked linear-Gaussian reference
chains in both representations, and the convergence scan. It records every command
and exit status and stops on failure. This is a scientific experiment, not a fast
CI test. Runtime depends strongly on compiler and hardware. NumPy and Matplotlib
are needed for analysis; SciPy is not required.

For an MPI build, use the normal `USE_MPI=true` settings. Add, for example,
`--mpi-bin-dir /path/to/mpi-build --mpirun 'mpirun --oversubscribe'` to run the
2-rank reference bridges. Only use launcher flags supported by your MPI vendor.
`--check-reference` and the existing `tests/test_survey_likelihood_mpi` regression
also run at 1/2/4 ranks. Do not mix object files built with different MPI settings.

## Fixed scientific problem

- A periodic 100 Mpc/h box contains an 8^3 initial/observation mesh and a 16^3
  deposition mesh, assignment width 6, constant initial power 2000 (Mpc/h)^3.
- Cutoffs are 0.115, 0.10 and 0.11 h/Mpc for the initial field, bias and likelihood.
  They leave 13 independent complex initial modes (26 real Gaussian coordinates).
- EdS 2LPT, plane-parallel RSD with f=0.6, and fixed bias
  (b1,b2,bs2,blaplacian)=(1.4,0.3,-0.2,0) generate the tracer field.
- The footprint is r<38 Mpc/h and z>−25 Mpc/h, with
  selection 0.85 exp(−r^2/(2*35^2)). Coordinates start at (−50,−50,−50).
- Gaussian observations have mean selection*tracer and voxel variance
  50/Vcell*(1+0.4*x^2/50^2). Counts have mean
  0.015*Vcell*selection*exp(tracer). The count link has zero offset.
- All spectra, biases, growth, noise and exposure values are fixed. This is
  conditional same-model calibration; it does not validate nuisance inference,
  cosmological parameter inference, model misspecification or N-body accuracy.

Each mock uses a fresh initial field, observation and independent future
observation. The Gaussian and Poisson families share their initial fields. Mock
truths/observations are synthesized through a direct Fourier sum of the model's
compact coefficient table; inference uses the actual FFT-based likelihood. The
initial-density normalization is independently checked in Python.

The Gaussian reference noise stored in the model configuration is rescaled by
mean(mask*selection^2/heterogeneity) to improve its diagonal HMC metric. The
likelihood's relative variance is inversely rescaled, preserving the physical
variance above exactly. Counts similarly use an approximate diagonal reference
noise for preconditioning. These choices do not change either likelihood.

## Exact active-mode reference and full-grid comparison

The orthonormal projector Q spans every initial mode that enters this fixed
forward model. Writing the white grid s=Q phi+s_perp gives

    p(s | d) = p(phi | d) p(s_perp),   s_perp independent standard Gaussian.

`survey_reference` integrates out s_perp and evaluates the **same production
posterior and gradient** on s=Q phi. It uses a fixed dense 26x26 linear Fisher
metric, M=I+B^T W B; its Cholesky momentum draw and inverse solve are checked
against the same kinetic energy. Replicated active coordinates count kinetic
energy once, while likelihood/gradient contributions reduce over MPI slabs.
This dense reference is deliberately small and is not a proposed scalable solver.
The full-grid production path remains unchanged.

`--check-reference` tests prediction invariance, the discarded Gaussian prior,
finite-difference active gradients and metric consistency for linear/2LPT and
Gaussian/Poisson. Independent masked linear-Gaussian posterior means/covariances
are computed using NumPy matrix algebra, with batch-mean Monte Carlo errors.
Three nonlinear production mocks per likelihood compare posterior coordinate
means/covariances to the reference ensemble. Agreement is assessed statistically;
compiler/FFTW builds and rank layouts and different mass matrices need not reproduce individual draws.

## Sampling and analysis

Four dispersed production chains run on mock 0 (1000 retained draws/chain); two
run on mocks 1 and 2 (800 draws). All 12 reference mocks use two chains and 800
retained draws; Poisson mock 3 is rerun with 1600 after its all-voxel check
exceeded Rhat=1.02 at 800. The original run is retained for audit. Warmup is 400; production/reference trajectories use 24/6 leapfrog
steps, with jitter and step adaptation targeting acceptance 0.9. The schedule has
a fixed 2400-draw cap; all chains extend by 800 if diagnostics fail. No bad mock
is discarded. Starts use different phases and amplitudes of 0.5 or 2.

Native rank-normalized split/folded Rhat and bulk/tail ESS cover the 26 modes,
white power, 16 physical-field probes and potential. Acceptance must exceed 0.65,
Rhat<1.02, bulk ESS>150, tail ESS>100, with zero retained divergences. Predictive
summaries have separate Rhat<1.05 and ESS thresholds. Python independently checks
Rhat and also checks all 512 physical-field values. Convergence diagnostics are
finite-sample evidence, not proof of convergence.

Every eighth state emits slab-local observable snapshots and conditional
predictive statistics. Python reconstructs physical initial fields exactly from
the 26 modes and draws independent Gaussian/Poisson predictive replicas. Future
voxel/total interval estimates use 128 independent conditional noise/count
replicates per saved posterior state, with 32/64/128-replicate stability checks.
These draws improve observation-noise integration; they are not additional
independent posterior samples. Discrepancy histograms keep one replica per state. It
reports central 50/68/90/95% intervals for modes, field values inside/outside the
footprint, future voxels and the future total. Integer count intervals use inverse
empirical CDF quantiles and can be conservative. Error bands use 4000 bootstrap
resamples of **independent mock fields**, never independent-voxel assumptions.
Twelve mocks provide only coarse calibration precision; rank bars are correlated
and are not a formal uniformity test.

The predictive plots compare independent future observations to marginal
posterior predictive totals, standardized squared residuals and a radial score.
Residuals use the posterior predictive mean/variance, including latent uncertainty.
The raw C++ conditional holdout residual columns are retained for audit but are
not treated as calibrated p-values. Fitted-data posterior predictive checks are
descriptive and are not expected to have uniform p-values. A single predictive
outlier need not indicate a failed model; inspect the whole mock ensemble.

## Separate convergence experiment

A fixed manufactured physical initial field is run at N=8,12,16,24,32,48 and
assignment widths 4,6,8. Retained tracer modes compare with N=48,width=8. Geometry
and selection are integrated with 1^3,2^3,4^3,8^3 sub-voxel midpoint samples;
the intensity field remains evaluated at the voxel center. Analytic sphere-cap
volume and radial-Gaussian selection integrals provide independent references.
Continuous total counts use cylindrical Gauss-Legendre quadrature, with a second
quadrature resolution checking stability. This is a total-integral and smooth
manufactured-field test, not voxelwise convergence or a volume-integrated
production likelihood. Mesh dependence need not be monotonic at sharp boundaries.

Raw trace, slab snapshots, predictive tables, complete HMC states and metadata are
kept in the requested output directory. `complete.txt` distinguishes PASS/FAIL.
The archived small figures and summary are review artifacts; raw chains are not
source-controlled.
