# Bounded active-marginal TSC HMC pilot (#52)

This follows the [joint MAP validation](../pm_tsc_joint_validation/RESULTS.md).
It reuses the frozen N128/M64/32-step TSC data and common M16 observation window.
**The user selected N16 particles for this pilot**, retaining the M16 force mesh,
16 time steps, 122 active real Fourier coordinates, priors and fixed cosmology.
This changes the particle discretization from the N32 MAP target. It does not
validate N32 posterior mixing or a particle/refinement limit.

## Exact active marginal and coordinates

The initial spectral cutoff makes all other white directions independent unit
Gaussians, with no likelihood dependence. Integrate these inactive directions out
exactly. On the active orthonormal Fourier basis, sample physical coefficients
`q=exp(A/2)*s_active`, log amplitude A and log noise power N with the **existing
`FML::MCMC::run_hmc` engine**. This is a 124-dimensional target, not two scalar
parameters conditioned on a fixed field. The active marginal is the same as the
full model at the declared discretization, up to a parameter-independent constant.
Inactive white draws could be restored independently; physical fields do not need
them. The full-field `FRAMEWORK::sample` adapter is separately smoke-tested.

The negative log density in physical coordinates is

```text
U(q,A,N) = likelihood(q,N) + .5*exp(-A)*dot(q,q)
         + GaussianPrior(A) + GaussianPrior(N) + 122*A/2.
```

The last term is the change-of-variables Jacobian. It uses **122**, not the total
number of grid cells: only active coordinates were transformed. Omitting it would
change the amplitude marginal. Gaussian log-prior means/scales and numerical
support match the joint study: A=0±.5 in [-4,4], N=log(1e-4)±.5 in [-16,-4].

The implementation uses one reference-noise forward/adjoint per new q, then
rescales the quadratic residual/noise normalization and Gaussian white prior
analytically for A,N. This is exact for this declared multiplicative-amplitude,
normalized Gaussian fixture, not a general cosmology/likelihood surrogate.
Repeated q values can reuse the reference result. Cache invalidation is exception
safe. Direct full-posterior values and chain-rule gradients (including the
Jacobian) and central differences are checked independently, including the
actual N16/M16/16-step sampling schedule.

A compact FFT basis replaces repeated trigonometric expansion/projection. It
matches the independent real Fourier basis, round-trip projections and norm
convention. No active-by-grid dense basis matrix is stored in the native target.

## Sampling and diagnostics

Four chains use dispersed physical-field perturbations and scalar starts,
seeds 12000..12003, 64 explicit warmup and 128 retained draws each, dual-averaged
step size initially .25, target acceptance .85 and jittered 4..8 leapfrog steps.
Mass adaptation is disabled: fixed diagonal masses are 21 for q, 65 for A and 132
for N. These are sampling choices, not a calibrated production Fisher metric.
Bounds use the existing reflective HMC support. Divergences are positive energy
increments above 10, counted after warmup.

Every retained state, including repeats after rejection, enters diagnostics;
there is no post-warmup thinning or burn-in removal. The library's standalone
rank-normalized split/folded Rhat and bulk/tail ESS monitor all 122 q coordinates,
their squares, both scalars, potential and norm squared (248 features). Preserve
the existing finite-chain criteria: Rhat <1.05, bulk ESS >400, tail ESS >200,
mean acceptance >.5 and no retained divergences. A bounded pilot may fail these
criteria; failures stay in the report and intervals then remain exploratory.

Every eighth retained draw saves the observed-band latent signal. Analysis adds
independent Gaussian measurement noise with variance exp(N)/V, preserving the
full-Hermitian Fourier convention. Latent and noisy uncertainty are separated.
The observed/replicated chi-square tail fraction is descriptive, not a coverage
probability. Initial-field summaries use physical coefficients sqrt(.002)*q. A companion
N16 whitened-coordinate MAP uses the same target/data, so the comparison of
parameter modes and pilot marginals does not conflate N16 and N32 targets.
The N32 result supplies only the dispersed physical initialization.

## Reproduce

```sh
make -C FML/FieldLevel/pm_tsc_posterior_validation USE_MPI=false
python3 FML/FieldLevel/pm_tsc_posterior_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --jobs 4
```

Use appropriate FFTW/compiler/launcher overrides, separate builds and one OpenMP
thread. Science chains are independent serial processes; native checks also cover
MPI slabs, pencils and sanitizers. Each science child has a bounded 1800s deadline;
process-group cleanup handles timeouts/interruption. NumPy and Matplotlib are
required for reports; C++ diagnostics use the library header. `--phase cases`
excludes analysis, and `--phase analyze` reuses saved native files.

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/analyze.py --input FML/FieldLevel/pm_tsc_posterior_validation --output /tmp/fml-tsc-posterior-report
```

The native CLI is `pm_tsc_posterior check|chain|full PREFIX DATA PHYSICAL_MAP N M
STEPS ID WARM DRAWS LEAP EPS ADAPT STRIDE`. `check` verifies the FFT basis,
transformed density/gradient/Jacobian and derivatives; `full` exercises the
untransformed full-field sampler. Counts and replicated controls are bounded.
`pm_tsc_diagnostics OUTPUT FEATURES0 FEATURES1 FEATURES2 FEATURES3` computes all
rank diagnostics. Use shared paths for MPI. Canonical outputs are bounded
validation output, not a production distributed chain-storage format.

See [RESULTS.md](RESULTS.md). Larger N32 sampling, coverage, broader refinement
and a reusable production centered-coordinate adapter remain outside this pilot.
