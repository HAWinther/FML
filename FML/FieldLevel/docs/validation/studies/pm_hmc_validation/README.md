> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Bounded PM HMC validation

This campaign exercises [periodic PM inference](../../../PM_POSTERIOR.md) in two distinct
settings: a two-mode conditional posterior with a quadrature reference, and short
full-white-field/fixed or joint-spectrum/noise integration and replay checks.
[RESULTS.md](RESULTS.md) records measured evidence and limits; [summary.json (archived)](../../../ARCHIVE.md)
contains the actual gates and numerical outputs. The sampler and PM numerical
implementations are unchanged by this validation work.

## Mathematical target

The model uses 4^3 initial particles, 8^3 CIC force/observation cells, box length 1,
flat Omega_m=0.3 and scale-factor nodes 0.2,0.3,0.4,0.5,0.6. The initial power is
0.02*exp(log_amplitude) at a=1, initial cutoff 7 and Fourier likelihood cutoff 15.
Mock noise power is 0.001 with global-ID seed 92. The mock is generated with the
same PM model as the inference target; this is a small dimensionless experiment.

For conditional runs the white field is restricted to

```text
white(q) = z0 b0(q) + z1 b1(q)
b0(q) = sqrt(2/N^3) cos(2 pi q_x/N)
b1(q) = sqrt(2/N^3) sin(2 pi q_x/N), N=4
```

The code checks global orthonormality. Consequently z0,z1 have independent unit
Gaussian priors under this restriction. Spectrum, noise and all other white
coordinates are fixed (the complementary white coordinates are zero). The injected
coefficients are (1.4,-0.9). This target is a conditional posterior, not the marginal
of the full field/nuisance posterior. The HMC gradient is the PM posterior gradient
contracted with these basis vectors.

The reference integrates that same nonlinear PM target by tensor trapezoidal
quadrature, using only forward values. A 41^2 grid and 81^2 grid cover [-6,6]^2;
a 109^2 grid on [-8.1,8.1]^2 preserves fine spacing and checks range sensitivity.
Three non-grid probe values compare the value-only evaluator with the complete
posterior evaluator. Quadrature is independent of the sampler, not an independent
physical PM implementation. Earlier forward/adjoint tests supply the physics and
derivative references.

Four fixed-step chains use 64 warmup and 256 retained draws each, epsilon 0.1,
jittered 4..8 leapfrog steps and seeds 1000..1003. Two adapted chains use 128 warmup
and 256 retained draws each, initial epsilon 0.1, target acceptance 0.85 and seeds
1004..1005. Their starts are dispersed; all reject/retain decisions are recorded.
The conditional mass is identity; mass adaptation is disabled. The positive energy
increment divergence threshold is 10, and reported divergence counts are post-warmup.

Five raw moments (z0,z1,z0^2,z1^2,z0*z1) are compared with quadrature. The agreement
envelope uses the largest of batch-mean, between-chain and reference-variance/ESS
Monte Carlo error estimates, plus reference refinement differences. It is a
conservative diagnostic check, not an exact confidence interval or a coverage test.
Rank-normalized split/folded Rhat and bulk/tail ESS use the library's existing
standalone diagnostic implementation. Bulk ESS can exceed the number of draws
for negatively correlated chains; the uncertainty envelope caps it at actual draw
count rather than claiming that precision without qualification.

Deterministic leapfrog probes use trajectory length 0.8, epsilon 0.4..0.025 and
four initial positions/momenta. They report signed energy increments and reversibility,
including probes away from the high-probability region. CIC is piecewise differentiable;
no universal second-order energy-error rate is assumed across cell faces.

## Full-field integration and MPI checks

The full runs call the existing `FRAMEWORK::sample()` on all 64 white coordinates,
with fixed parameters or jointly varying log-amplitude/noise. They use 16 warmup,
16 retained draws, epsilon 0.06 and jittered 2..4 leapfrog steps, without adaptation.
They check finite values, all retained draws, parameter support, post-warmup
divergences and canonical last-state/trace replay across ranks, backends and threads.
They do not establish mixing or agreement of full-field/nuisance posterior moments.
Their target differs from the conditional quadrature target.

Scalar traces are streamed through the public sink; no full-field sample tape is
retained. The canonical final field is gathered only for validation output. All
native posterior/gradient operations remain distributed and use bounded adjoint replay.

## Build and run

```sh
make -C FML/FieldLevel/pm_hmc_validation -j2
python3 FML/FieldLevel/pm_hmc_validation/run.py \
  --bin-dir FML/FieldLevel/pm_hmc_validation --output /tmp/pm-hmc-run
```

The runner requires a fresh output directory, NumPy and Matplotlib for analysis.
It uses the existing standalone C++ diagnostics binary, without SciPy/ArviZ. Optional
`--mpi-bin-dir`, `--pencil-bin-dir`, `--sanitizer-bin-dir` and `--mpirun` add the replay
matrix. Build into separate clean directories before changing compiler/MPI/backend
flags; the runner receives prebuilt binaries and preserves an explicit MPI launcher.
Use `USE_MPI=true USE_PENCIL_GRID=true` for a pencil binary. The recorded sanitizer
matrix uses Clang 20 ASAN/UBSAN with leak detection disabled.

Individual commands:

```sh
pm_hmc reference PREFIX 81 6
pm_hmc sweep PREFIX
pm_hmc chain PREFIX CHAIN_ID EPS LEAPFROG WARMUP DRAWS ADAPT_STEP
pm_hmc full PREFIX JOINT_FLAG DRAWS
pm_hmc_diagnostics OUTPUT.csv CHAIN1.chain.csv CHAIN2.chain.csv [...]
```

The full mode's warmup, step and leapfrog settings are fixed by this validation
fixture; its draw count is bounded by the CLI. Reference extent is 4..9 and grid
size is odd, 9..201. Fixed/conditional/joint modes emit distinct CSV schemas;
metadata records actual post-warmup acceptance, divergences and step size.

The lightweight standard-library-only CI target runs eight-draw checks, with no
quadrature or statistical convergence claim:

```sh
make -C FML/FieldLevel/tests pm-hmc-smoke USE_MPI=true NP_LIST="1 2 4"
```

It is included in `FieldLevel/tests`'s `run` target. The long scientific campaign
is explicit, not part of routine module CI.

## Remaining work

Longer full-field and nuisance mixing, posterior predictive checks and multi-mock
calibration remain open. The conditional test does not establish production accuracy,
all-direction reconstruction, cosmological inference, survey/tracer/RSD likelihoods
or posterior coverage. Smoother assignment, improved metrics/optimization, PM-specific
restart identities and wider resolution/performance validation remain extensions.

The subsequent [full-field campaign](../pm_full_validation/README.md) adds dispersed
fixed/joint chains and posterior predictions for one small mock. The results here
remain the separate two-mode conditional reference and short integration milestone.
