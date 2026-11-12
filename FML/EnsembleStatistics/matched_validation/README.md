# Matched ZA/2LPT versus PM ensemble validation

This runnable C++ example connects existing FML components to control-variate mean
estimation. It generates Gaussian Philox initial phases, creates 2LPT particles,
constructs ZA/2LPT controls and evolves a small EdS particle-mesh target. Independent
training, cheap-mean and evaluation phase blocks feed the public ensemble API.
The optional NumPy runner independently checks every coefficient, mean and covariance.

This is an integration/statistical experiment, not a calibrated cosmological mock
suite or a precision N-body solver. See [RESULTS.md](RESULTS.md), especially the
large mesh sensitivity and the finite cheap-mean uncertainty floor.

## Build and run

From the repository root, using normal FML compiler/FFTW/GSL configuration:

```sh
make -C FML/EnsembleStatistics/matched_validation USE_MPI=false
# prefix mesh steps afinal ntrain nmean neval seed amplitude
FML/EnsembleStatistics/matched_validation/matched_pm /tmp/matched 16 20 .5 64 1024 256 12340 .001
python3 FML/EnsembleStatistics/matched_validation/regression/reference.py \
  --binary FML/EnsembleStatistics/matched_validation/matched_pm --output /tmp/matched-reference
```

The reference runner needs NumPy. Its full suite includes a=0.5 and a=1, 20/40/80-step
comparisons, and combined IC/particle/force mesh refinement from 16 to 32. Use
`--smoke` for one small 12-training/32-cheap/16-evaluation case. The driver writes
verbose existing NBody/particle diagnostics to stdout; redirect these to a log.
The GSL headers are included transitively by `NBody.h`; the EdS time integrals are
analytic and no ODE integration is performed by this driver.

For optional MPI checks, build another copy with `USE_MPI=true` and pass
`--mpi-binary /path/to/mpi/matched_pm --launcher 'mpirun'`. The runner tests 2 and 4
ranks against serial. Each density field is distributed over all ranks and each
FFT/power operation is collective. Only rank zero adds the measured ensemble row;
`global()` then returns common moments/models, including empty ensemble-owner ranks.
MPI calls occur outside OpenMP regions. Clean between build configurations.

## Physical and statistical conventions

- EdS background, comoving L=200 (e.g. Mpc/h), a_ini=.05. The supplied amplitude A
  specifies the linear a=1 spectrum `P(k)/V = A*(q/6)*exp(-(q/20)^2)` for `q=kL<=12pi`,
  zero beyond that support. The initial Fourier spectrum includes a_ini squared.
  Amplitudes are Gaussian, not fixed; no antithetic pairing is used.
- Mesh and particle count per dimension are equal. The existing 2LPT IC routine
  supplies stored displacements and Lagrangian coordinates. ZA uses
  `q+(a/a_ini)D1`, 2LPT adds `(a/a_ini)^2 D2`. The initial PM positions/velocities
  are restored from these same stored fields before evolution.
- The example uses the existing continuous Fourier force kernels, CIC density and
  force interpolation, and no force deconvolution. Momentum is
  `p=a^2 dx/d(H0 t)`. With the force grid `i k delta/k^2`, the EdS kick coefficient
  is `3*(sqrt(a_hi)-sqrt(a_lo))` and drift coefficient
  `2*(1/sqrt(a_lo)-1/sqrt(a_hi))`. The staggered kick/drift loop has nsteps+1 force
  evaluations, with first/last synchronization half-kicks and uniform steps in a.
- All measured auto/cross spectra retain the same raw CIC assignment convention,
  with physical volume scaling and centers `(2pi,4pi,6pi)/L`. There is no particle
  shot-noise subtraction, assignment compensation or survey correction. The target
  is the discrete PM ensemble under this convention, not a continuum spectrum.
- Phase identity is `(id,seed+id,stream=0,Philox,Gaussian,+1)`. Training rows precede
  independent cheap-mean rows, which precede held-out evaluation rows. Each is
  added once. Cheap-mean rows do not evolve a PM target. Neither the training nor
  evaluation control mean is substituted for the independent population mean.
- Each control gets separate diagonal and full fits, using the default dimensionless
  standardized ridge 1e-8. Coefficients are frozen before evaluation. The reported
  mean covariance is `Cov(residual)/N_eval + B Cov(cheap)/N_mean B^T`.
  Field cross coherence, training power correlation, held-out residual variance
  reduction and mean variance reduction are different quantities.

`ControlMean` is the reusable addition in `ControlVariates.h`. Feed actual cheap
summary vectors and their actual phase keys to it, then call
`training.global().fit(cheap.global().estimate(), options)`. Scientific metadata
must describe the real cosmology, model versions and observable preprocessing;
the API cannot verify that caller-provided labels are truthful. Production runs
can use their own ZA/2LPT/COLA/N-body summaries without using this fixture's PM loop.

## Outputs and independent checks

For each prefix, the driver writes `-rows.csv` (all phase keys, bins, powers and
cross diagnostics), `-timing.csv`, `-config.txt`, `-models.csv` (metadata/model hashes)
and `-results.csv` (coefficients, raw/corrected means and covariance matrices,
including the separate mean-uncertainty floor). Missing target/cross values on
cheap-mean rows are explicit NaNs. Output-directory creation is the caller's job.

The runner reconstructs all four fits with NumPy sample covariance and a dense
linear solve. It compares every result and saves reference arrays, conditioning,
variance reductions, timing summaries, commands and source/binary/data hashes.
It checks complete disjoint phase blocks and consistent bin layouts. Timestep
comparisons reuse the same first 12 phases and physical input support. Mesh
refinement changes particle, IC, force and measurement resolution together.
These sensitivity checks do not assert that the coarsest PM target is converged.

Stage timings are maximum rank-local wall times; total row times include the
measurement and streamed output work. Both surrogate fields share IC generation.
The ZA timing therefore is not a standalone optimized ZA simulation cost: all
rows pay for the same 2LPT setup, and both controls are measured in one workflow.
Logging, startup, I/O, machine load and resolution affect these small timings.
No equal-cost production speedup is inferred. For a real cost comparison include
training targets, paired controls, the independent cheap mean and evaluation,
and compare `Cov(mean)` at the same total budget; do not use residual variance
reduction alone.

## Repeated statistical coverage

`tests/test_coverage.cpp`, included in the ordinary core test target, runs 2048
independent scalar experiments with target `y=5+1.7c+0.6epsilon`, independent unit
Gaussian c and epsilon. Each has 32 paired training rows, 256 independent cheap
rows and 64 evaluation rows. It checks known mean bias, reported versus empirical
mean variance, approximate coverage of nominal 95% Gaussian intervals, and the
undercoverage caused by omitting the shared cheap-mean term. MPI distributes whole
experiments across ranks; no experiment shares its phase blocks. Finite-sample
Gaussian intervals are approximate and are not a general confidence guarantee
for cosmological bandpowers. The PM experiment does not have an analytic target
mean, so coverage is not claimed for it.

Reference: [CARPool mean estimation, Chartier et al.](https://arxiv.org/abs/2009.08970).
This fixture validates the existing frozen-coefficient mean estimator; it does not
implement the separate CARPool target-covariance estimator or new COLA dynamics.
