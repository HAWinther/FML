> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Bounded full-field PM posterior validation

This extends [the two-mode conditional campaign](../pm_hmc_validation/README.md)
with four dispersed **full-field** chains and four **joint field/spectrum/noise**
chains. It exercises the existing public `FRAMEWORK::sample()` without changing
production PM, adjoint, likelihood or HMC code. See [RESULTS.md](RESULTS.md) for
measured diagnostics, plots and remaining limits.

## Target and scope

The periodic fixture uses N=4 initial particles/cells, M=8 force/observation cells,
box length 1, flat Omega_m=0.3 and scale factors 0.2,0.3,0.4,0.5,0.6. Its initial
power is `0.02*exp(log_amplitude)` with cutoff 7; likelihood cutoff is 15. The
spectrum parameter has a Gaussian prior N(0,0.5²) truncated to [-1,1]. Log noise
power has prior N(log(0.001),0.5²) truncated to [-14,-3]. The Fourier likelihood is
normalized and its noise variance is exp(log_noise)/V.

The mock white field uses global-ID seed 91 and Gaussian mock noise seed 92.
Unlike the preceding two-mode mock, all six supported IC Fourier directions are
excited. Every chain samples **all 64 real white coordinates**. The initial cutoff
leaves six active real Fourier coordinates and 58 orthogonal null directions. The
latter have an exact independent unit-Gaussian posterior, even in joint inference;
this is a useful reference, not evidence for 64 data-constrained directions.

Initial fields use independent seeds 300..303 and scales 0.5,1,1.5,2. Joint scalar
starts are dispersed within their bounds. Sampler seeds are 9000..9003; fixed and
joint ensembles reuse these seeds but target different posteriors. Four chains per
ensemble use 256 warmup and 1024 retained draws, dual-averaged step size starting
at 0.08, target acceptance 0.85, jittered 16..32 leapfrog steps, identity white
metric and fixed scalar masses from prior scales. Mass adaptation is disabled.
Divergences count positive Hamiltonian increments above 10 after warmup. Short
pilot runs are separate artifacts and do not enter the reported chains.

In an orthogonal active/null basis the target factors as

```text
p(active, null, log_amplitude, log_noise | data)
  = p(active, log_amplitude, log_noise | data) * N(null; 0, I_58).
```

The active target still contains the nonlinear PM map and normalized likelihood.
Joint amplitude/white-field degeneracy makes scalar and squared-mode diagnostics
useful alongside individual grid coordinates. The Gaussian null reference checks
unconstrained sampling; it does not independently reference the active posterior.

## Diagnostics and predictions

- `Main.cpp` streams every retained white/scalar state, including repeated states
  after rejection. It saves latent initial/final densities every eighth retained
  draw using a separate prediction workspace. Canonical gathers are deliberately
  bounded validation output; they are not a scalable production storage design.
- `prepare.py` constructs features from all draws: 64 white coordinates, six active
  projections and their squares, 58 orthogonal null projections and their squares,
  null/full norms, potential and varying scalar parameters. QR completes a checked
  orthonormal Fourier basis; no diagnostic thinning or burn-in removal is applied
  after the sampler's explicit warmup.
- `Diagnostics.cpp` uses the existing rank-normalized split/folded Rhat and bulk/tail
  ESS. The declared finite-chain gates are Rhat <1.05, bulk ESS >400 and tail ESS
  >200 for every monitored feature, with finite states, mean acceptance >0.5 and
  no post-warmup divergences. These thresholds are diagnostics, not convergence
  proofs. The null-coordinate first/second moments use five analytic Monte Carlo
  standard errors, capping ESS at the actual sample count. A direct DFT also checks
  initial reconstruction normalization and the six-mode support.
- `analyze.py` computes pointwise latent density intervals and shell-power intervals,
  then generates independent Gaussian measurement noise for a noisy posterior
  prediction. Fourier noise uses the same exp(log_noise)/V normalization. It
  compares replicated-data and observed-data chi-square discrepancies. This
  posterior-predictive tail fraction is descriptive, not a coverage probability.

Initial density is the linear field at the spectrum normalization epoch (a=1),
not the displaced particle density at a=0.2. `density_intervals.csv` exports
pointwise 95% latent intervals; the slice plot displays latent standard deviations.

Final density slices and residuals compare the **observed Fourier band** (nonzero
k<15) on the M=8 grid. Initial density uses the supported N=4 band. Latent density
uncertainty and noisy posterior-predictive uncertainty are shown separately. Shell
powers use the box-normalized Fourier coefficients and full-Hermitian multiplicity;
initial power has only one supported shell. Plot intervals are pointwise/equal-tailed,
not simultaneous bands. Density slices are z=0.

## Reproduce

Build serial, MPI slabs or MPI pencils with normal FML configuration overrides:

```sh
make -C FML/FieldLevel/pm_full_validation pm_full pm_full_diagnostics USE_MPI=false
make -C FML/FieldLevel/pm_full_validation clean
make -C FML/FieldLevel/pm_full_validation pm_full pm_full_diagnostics USE_MPI=true USE_PENCIL_GRID=true
```

Use separate build directories or clean when changing backend/compiler flags. The
scientific Python workflow requires NumPy and Matplotlib; native diagnostics use
only the library's standalone C++17 diagnostic header.

```sh
python3 FML/FieldLevel/pm_full_validation/run.py \
  --bin-dir /path/to/serial/binaries --output /path/to/fresh/output --jobs 4
```

Optional `--mpi-bin-dir`, `--pencil-bin-dir`, `--sanitizer-bin-dir` and `--mpirun`
add short canonical state/predictive replay checks. The runner preserves an explicit
MPI launcher and arguments; choose flags appropriate for your MPI implementation.
Pencil checks use four ranks with a 2x2 decomposition. They also compare one/two
OpenMP threads; the sanitizer variant disables leak detection. Long statistical
chains run serially per process, four concurrent independent chains by default.

Native CLI (the final `1` selects fixture schema 1):

```sh
pm_full PREFIX JOINT_FLAG CHAIN_ID WARMUP DRAWS EPSILON LEAPFROG ADAPT_FLAG PREDICTIVE_STRIDE 1
pm_full_diagnostics OUTPUT.csv CHAIN1.features.csv CHAIN2.features.csv [...]
```

Counts are bounded by the driver. `--warmup`, `--draws` and `--leapfrog` configure
longer explicit campaigns. `run.py` requires a fresh science directory; native
commands overwrite their given prefix. This longer validation is explicit and is
not added to routine CI.

## Limits

One same-model mock with six active IC directions cannot establish repeated-mock
coverage, larger-field mixing, high-cutoff/nonlinear accuracy, or mesh/particle/time
convergence. Fixed flat-background dynamics, CIC assignment and identity field
metric retain their existing limitations. [Same-layout PM restart](../../../PM_CHECKPOINT.md)
is now available. Changed-layout restart, tracer/RSD/survey likelihoods, cosmological
derivatives and production performance remain extensions.
