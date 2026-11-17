> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

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

## Optional alternating kernel

`run.py --scheme blocked` uses the identical active-marginal density and initial
states. The default `--scheme joint` retains the original joint HMC code path.
The native mode is `blocked` (all other positional arguments stay unchanged).
This is a validation-only adapter for this amplitude/Gaussian-noise fixture.
It does not yet provide a general production blocked sampler.

Each outer iteration makes one conditional HMC move of all 122 physical field
coordinates, then four slice sweeps of A and N. At cached q, the conditional
negative log densities are

```text
U(A | q) = .5*exp(-A)*dot(q,q) + GaussianPrior(A) + 122*A/2
U(N | q) = .5*chi0(q)*exp(-(N-N0)) + 128*(N-N0) + GaussianPrior(N).
```

Here `chi0` is the reference-noise residual chi-square and N0=log(1e-4); the
observed band has 256 real degrees of freedom. These scalar densities are
strictly convex. The bounded slice kernel starts from the full prior support
and shrinks a connected slice; every conditional transition preserves its
conditional density. The draws are conditional Markov updates, not claimed
independent conditional samples. Composing them with Metropolis-corrected field
HMC preserves the joint density once warmup adaptation stops.

The field HMC state retains its RNG and step adaptation between trajectories.
Changing scalar parameters requires refreshing the saved potential and field
force. A rejected field trajectory restores the cache at the accepted field.
Each iteration asserts that scalar sweeps and the force refresh consume **zero
additional PM evaluations**. The independent scalar RNG uses seed 23000+chain ID;
field HMC uses 12000+chain ID. All retained states follow both blocks, including
repeated fields after HMC rejections. The acceptance and divergence diagnostics
refer to the field HMC kernel. Retained draws are never thinned.

Validate the conditional kernel against independent quadrature:

```sh
make -C FML/FieldLevel/pm_tsc_posterior_validation pm_tsc_conditionals
python3 FML/FieldLevel/pm_tsc_posterior_validation/check_conditionals.py --binary /path/to/pm_tsc_conditionals --output /tmp/conditional-check
```

Three 200,000-draw tests cover the amplitude and noise conditional forms and a
narrow Gaussian truncated near its support edge. The native `check` mode also
compares each scalar conditional to the full joint density and exercises cache
restoration. Slab and pencil MPI runs use identical replicated active
coordinates and scalar RNG decisions.

Compare two saved campaigns without relaxing the mixing criteria:

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/compare.py --baseline /joint/OUT --candidate /blocked/OUT --output /comparison/OUT
```

Both inputs need `plan.json`, `science` and `analysis/summary.json`. The comparison
checks target, seeds and budgets, then reports bulk and tail ESS divided by the
actual total forward/adjoint PM evaluations, including warmup. Predictive-only
forwards are excluded equally. Sampler elapsed time is reported separately. A speedup in
scalar mixing does not establish adequate mixing of the fields.

See [the alternating-kernel measurements](BLOCKED_RESULTS.md).

## Fixed field-curvature experiment

`--scheme preconditioned` retains the alternating scalar kernel and changes only
the field mass operator. Before the chains start, it measures the 122x122 field
Hessian of the exact centered target at the common, unperturbed physical reference
field, A=0 and N=log(1e-4). Central differences of gradients use h=1e-4. The matrix
is symmetrized, checked with four independently refined directional differences,
and given a diagonal ridge only if needed for a stable positive Cholesky factor.
The matrix is then **fixed for every warmup and retained trajectory**, shared by
all chains. It is not recomputed at sampled fields or parameters.

The HMC convention is precision-like mass: for M=LL^T, draw momentum p=L*z and
compute velocity M^-1*p by triangular solves. The existing custom-mass interface
handles the replicated active coordinates. This changes the proposal geometry;
the exact density, Metropolis correction and original 122*A/2 Jacobian remain.
No extra determinant or coordinate Jacobian is added for this fixed mass.
The production HMC engine requires no changes.

Metric setup makes 244 coordinate perturbation evaluations plus eight directional
checks, **252 forward/adjoint evaluations once for all four chains**. The comparison
includes them, along with warmup and retained sampling costs. Scalar sweeps and
force refresh still consume no additional PM evaluations. The matrix and its
hash, setup cost, ridge, derivative checks and factorization error are saved.
The analysis independently checks positivity and symmetry and preserves the
original convergence thresholds.

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --scheme preconditioned --timeout-seconds 2400
```

The native setup mode is `metric` with the same 15-argument control format
(the EPS slot supplies h); it writes PREFIX.matrix.csv and PREFIX.meta.csv.
Native `preconditioned` uses the usual sampling arguments followed by a final
metric CSV filename. Other modes reject that extra argument. Metric files must
be finite, symmetric, positive definite, and exactly 122x122 for this fixture;
input errors fail collectively under MPI. Replicated matrix values are checked
before entering the sampler.

```sh
make -C FML/FieldLevel/pm_tsc_posterior_validation pm_tsc_metric_test
/path/to/pm_tsc_metric_test
```

This independent two-dimensional correlated Gaussian test checks inverse-mass
algebra, positive factorization and HMC moments. The dense metric is deliberately
limited to this small active space, not a production full-grid preconditioner.
Its O(d^2) storage and O(d) PM setup calls would not scale to a full field.

See [the fixed-metric results](METRIC_RESULTS.md).

## Shorter trajectories at matched PM cost

The optional runner controls `--draws`, `--leapfrog` and `--predictive-stride`
retain defaults 128, 8 and 8. They do not change the native target or HMC engine.
For the next controlled test, four chains use 64 warmup +320 retained draws and
jittered 2..4 leapfrog steps (`--leapfrog 4`). The preceding metric campaign uses
64+128 iterations with 4..8 steps. Expected trajectory work is equal:
`(64+320)*3 = (64+128)*6 = 1152` force evaluations per chain. Actual totals may
differ because trajectory lengths are randomized and divergent trials can stop
before their nominal end. Every retained draw enters diagnostics.

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/run.py --bin-dir /path/to/binaries --output /fresh/SHORT --scheme preconditioned --draws 320 --leapfrog 4 --predictive-stride 20 --timeout-seconds 2400
```

The metric is built afresh from the same reference and checked to match the
preceding frozen matrix exactly. The 252 setup evaluations are charged once to
each campaign. Prediction stride 20 saves the same 16 predictions per chain as
before; this only controls predictive output, not diagnostic thinning. Warmup,
seeds, starts, target, priors, step adaptation and convergence thresholds stay
fixed. More outer iterations also mean more cheap scalar conditional updates.

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/compare.py --baseline /METRIC --candidate /SHORT --output /COMPARE --allow-trajectory-change
```

The explicit comparison flag permits different draw counts, trajectory lengths
and predictive strides. All differences are listed in the comparison JSON;
remaining target/protocol checks stay active. Both saved plans must agree with
their native metadata. Preconditioned comparisons require identical matrix
hashes, and the actual total PM budgets must lie within 5% of each other.
The default comparison still requires identical sampling schedules.

See [the shorter-trajectory results](TRAJECTORY_RESULTS.md).

See [the intermediate-trajectory comparison](INTERMEDIATE_RESULTS.md) for the maximum-6-step, 192-draw schedule and all-three-schedule plots.

## Longer fixed-schedule validation

After the three matched-budget schedules, a predeclared longer run keeps maximum
6 leapfrog steps (jittered 3..6), 64 warmup and the same four seeds/starts, with
512 retained draws per chain. The fresh metric must match the preceding matrix.
All 2048 retained states enter diagnostics. Prediction stride 32 keeps 16 saved
predictions per chain. This increases the expected total PM budget about 2.2x;
it is not a matched-budget tuning comparison.

```sh
python3 FML/FieldLevel/pm_tsc_posterior_validation/run.py --bin-dir /path/to/binaries --output /fresh/LONG --scheme preconditioned --draws 512 --leapfrog 6 --predictive-stride 32 --timeout-seconds 7200
python3 FML/FieldLevel/pm_tsc_posterior_validation/compare.py --baseline /MID --candidate /LONG --output /COMPARE --allow-trajectory-change --allow-budget-change
```

The runner accepts timeouts up to 7200s per child, keeping its original default.
The larger allowance avoids terminating this roughly hour-long case near the
old 3600s cap. The retained-draw count remains fixed in advance; the run is not
extended until a diagnostic passes. The explicit budget option requires the
trajectory-change option, records that the budgets differ, and preserves all
shared-target, metric, metadata and convergence checks. Without it, the existing
5% matched-budget guard remains active.

With unchanged seeds and starts, the pilot prefix should reproduce. The extra
draws extend the same four chains and are not an independent replication. Passing
finite-chain mixing criteria would still leave N32 mixing and repeated-mock
coverage unvalidated. No native algorithm or production default changes.

See [the longer fixed-schedule validation](LONG_VALIDATION_RESULTS.md) for the 512-draw case and its unchanged strict mixing verdict.

## Independent same-model mocks

`run_mock.py` generates a fresh N16/M16/16-step mock, fits a data-only MAP,
constructs and freezes its metric, then runs the predeclared four-chain
64-warmup/512-retained protocol. IC and measurement-noise seeds are separate
(default 191/192); both must differ from the original 91/92. The original
strict mixing criteria remain unchanged. This driver does not tune the sampler
or extend sampling in response to the verdict. New data have their own hashes;
the benchmark comparator rejects comparisons against the original dataset.

```sh
python3 run_mock.py --bin-dir /path/to/binaries --output /fresh/MOCK --ic-seed 191 --noise-seed 192 --jobs 4 --timeout-seconds 7200
```

The native posterior CLI accepts an optional trailing `IC_SEED NOISE_SEED`
pair after the existing arguments (after the metric filename for
`preconditioned`). The `mock` mode requires data and physical-MAP arguments
both `-` and writes `.data.csv`, `.truth.csv`, `.truth_modes.csv` and
`.mock_meta.csv`. The companion `pm_tsc_joint` accepts the same pair after
explicit `VARY FD` arguments. Omitting seeds preserves the legacy 91/92 data
identity and results. MPI ranks must agree on both seeds before grid setup.

The analyzer reports true-parameter interval inclusion and physical initial
field recovery separately from mixing. A single mock and its correlated
coefficient intervals cannot establish repeated-mock coverage. The original
refined-data case and a new same-model mock also have different numerical
generation targets and must not be pooled as identical realizations.

[Independent same-model N16 results and plots](INDEPENDENT_MOCK_RESULTS.md): the frozen four-chain protocol also passes on new IC/noise seeds. One mock does not establish ensemble coverage.

## Repeated same-model mock campaign

`run_mocks.py` predeclares four new IC/noise pairs (291/292, 391/392, 491/492,
591/592) before any sampling and runs `run_mock.py` once per dataset. The prior
191/192 mock is excluded from this four-dataset sample. Native binary and driver
hashes are frozen; each mock uses the unchanged four-chain 64-warmup/512-retained
protocol and strict per-feature gates. Only four native chains run concurrently.
Every failure is kept and the remaining declared mocks continue. The driver
requires a fresh output directory and never replaces failures or adds draws.

```sh
python3 run_mocks.py --bin-dir /path/to/binaries --output /fresh/REPEATED
python3 analyze_mocks.py --input /saved/REPEATED --output /fresh/ANALYSIS
```

The aggregate analyzer diagnoses datasets separately and reports every declared
case, including missing results. It shows scalar interval inclusion with Wilson and exact binomial
uncertainty across datasets, plus field recovery. A truth outside a 95% interval
is not itself a mixing failure. Four fixed-hyperparameter datasets provide only
a small repeated-data check, not prior-predictive simulation-based calibration.
Do not pool correlated field-mode counts or combine chains across datasets.
Intervals from missing/non-mixing campaigns remain explicitly unreliable.

[Repeated-mock results and plots](REPEATED_MOCK_RESULTS.md) retain every declared case and the actual strict verdict, with dataset-level interval-inclusion uncertainty.
